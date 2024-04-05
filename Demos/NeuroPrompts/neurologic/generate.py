import math
import random
import numpy as np
import torch
from torch import Tensor
import logging
from torch.nn import functional as F
from scipy.stats import rankdata
from operator import attrgetter
from typing import Dict, List, Optional, Tuple, Set, Union,Iterable
import transformers
import sys
import os
import time
# from multiprocessing import Pool
# import multiprocess as mp
from joblib import Parallel, delayed
# import asyncio
import itertools

neurologic_path = os.environ['NEUROLOGIC_PATH']
sys.path.insert(0,neurologic_path)

from lexical_constraints import ConstrainedHypothesis, ConstrainedCandidate

logger = logging.getLogger(__name__)
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('gpt2')


def topk_huggingface(timestep: int,
                     batch_size: int,
                     beam_size: int,
                     vocab_size: int,
                     pad_token_id: int,
                     diversity: bool,
                     constraint_limit,
                     sat_tolerance: float,
                     inactive: np.array,
                     scores: np.array,
                     hypotheses: List[ConstrainedHypothesis],
                     num_fill: int,
                     do_beam_sample: bool,
                     beam_temperature: float,
                     look_ahead: int,
                     max_length: int,
                     model: transformers.PreTrainedModel,
                     temp_input_ids: torch.Tensor,
                     temp_attention_mask: torch.Tensor,
                     temp_position_ids: torch.Tensor,
                     temp_past: Tuple[torch.Tensor],
                     score_history: np.array,
                     stochastic,sampler,cur_scores,
                     model_specific_kwargs
                     ) -> Tuple[np.array, np.array, List[List[Union[ConstrainedHypothesis, None]]], List[List[int]]]:
    """
    Builds a new topk list such that the beam contains hypotheses having completed different numbers of constraints.
    These items are built from three different types: (1) the best items across the whole
    scores matrix, (2) the set of words that must follow existing constraints, and (3) k-best items from each row.

    :param batch_size: The number of segments in the batch.
    :param beam_size: The length of the beam for each segment.
    :param vocab_size: The size of vocabulary.
    :param pad_token_id:
    :param prune_factor:
    :param sat_tolerance:
    :param inactive: Array listing inactive rows (shape: (batch_size, beam_size,)).
    :param scores: The scores array (shape: (batch_size, beam_size * target_vocab_size)).
    :param hypotheses: The list of hypothesis objects. (length: (batch_size * beam_size,))
    :param num_mets: The list of int how many constraints satisfied. (length: (batch_size * beam_size,))
    :param num_fill: The number of required return beam
    :return: A tuple containing the best hypothesis rows, the best hypothesis words, the scores,
        the updated constrained hypotheses, and the updated set of inactive hypotheses.
    """
    seq_scores, raw_token_idx = torch.topk(scores, beam_size, dim=1, largest=True, sorted=True)
    best_ids = (raw_token_idx // vocab_size).cpu().numpy()
    best_word_ids = (raw_token_idx % vocab_size).cpu().numpy()
    seq_scores = seq_scores.cpu().numpy()

    scores = torch.reshape(scores, [batch_size, beam_size, -1]).cpu().numpy()

    select_best_ids = np.ones((batch_size, num_fill)) * -1
    select_best_word_ids = np.ones((batch_size, num_fill)) * -1
    select_seq_scores = np.zeros((batch_size, num_fill))
    select_hypotheses = [[None] * num_fill for _ in range(batch_size)]
    select_num_mets = [[-1] * num_fill for _ in range(batch_size)]

    for sentno in range(batch_size):
        rows = slice(sentno * beam_size, sentno * beam_size + beam_size)
        idxs = torch.arange(sentno * beam_size, sentno * beam_size + beam_size)
        if all([x is None for x in hypotheses[rows]]):
            select_best_ids[sentno] = [0] * num_fill
            select_best_word_ids[sentno] = [pad_token_id] * num_fill
            select_seq_scores[sentno] = [0] * num_fill
            select_hypotheses[sentno] = [None] * num_fill
            select_num_mets[sentno] = [-1] * num_fill
            continue

        assert not any([x is None for x in hypotheses[rows]]), 'Bad state'
        is_look_ahead = [int(c.look_ahead) for c in hypotheses[rows][0].clauses]

        if not sum(is_look_ahead):
            select_best_ids[sentno], select_best_word_ids[sentno], select_seq_scores[sentno],\
                select_hypotheses[sentno], select_num_mets[sentno] = _sequential_topk(timestep,
                                                                                      sentno,
                                                                                      beam_size,
                                                                                      diversity,
                                                                                      constraint_limit,
                                                                                      sat_tolerance,
                                                                                      inactive[sentno],
                                                                                      scores[sentno],
                                                                                      hypotheses[rows],
                                                                                      best_ids[sentno],
                                                                                      best_word_ids[sentno],
                                                                                      seq_scores[sentno],
                                                                                      do_beam_sample,
                                                                                      beam_temperature,
                                                                                      num_fill=num_fill,
                                                                                              stochastic=stochastic,sampler=sampler,cur_scores=cur_scores[sentno],)
        else:
            select_best_ids[sentno], select_best_word_ids[sentno], select_seq_scores[sentno],\
                select_hypotheses[sentno], select_num_mets[sentno] = _look_ahead_topk(sentno,
                                                                                      timestep,
                                                                                      beam_size,
                                                                                      diversity,
                                                                                      sat_tolerance,
                                                                                      inactive[sentno],
                                                                                      scores[sentno],
                                                                                      hypotheses[rows],
                                                                                      best_ids[sentno],
                                                                                      best_word_ids[sentno],
                                                                                      seq_scores[sentno],
                                                                                      do_beam_sample,
                                                                                      beam_temperature,
                                                                                      num_fill,
                                                                                      look_ahead,
                                                                                      max_length,
                                                                                      model,
                                                                                      temp_input_ids[rows],
                                                                                      temp_attention_mask[rows],
                                                                                      temp_position_ids[rows],
                                                                                      _reorder_cache(temp_past, idxs),
                                                                                      score_history[rows] if score_history is not None else None,
                                                                                      model_specific_kwargs)

    select_raw_token_idx = select_best_ids * vocab_size + select_best_word_ids
    return select_seq_scores, select_raw_token_idx, select_hypotheses, select_num_mets

def topk_for_sent(sentno, timestep, beam_size, hypotheses, num_fill, pad_token_id, diversity, constraint_limit, sat_tolerance, 
                  inactive_sent, score_sent, best_ids_sent, best_word_ids_sent, seq_scores_sent, cur_scores_sent,
                  do_beam_sample, beam_temperature, stochastic, sampler):
    rows = slice(sentno * beam_size, sentno * beam_size + beam_size)
    idxs = torch.arange(sentno * beam_size, sentno * beam_size + beam_size)
    if all([x is None for x in hypotheses[rows]]):
        sent_best_ids = [0] * num_fill
        sent_best_word_ids = [pad_token_id] * num_fill
        sent_seq_scores = [0] * num_fill
        sent_hypotheses = [None] * num_fill
        sent_num_mets = [-1] * num_fill
        return [sent_best_ids, sent_best_word_ids, sent_seq_scores, sent_hypotheses, sent_num_mets]

    assert not any([x is None for x in hypotheses[rows]]), 'Bad state'
    is_look_ahead = [int(c.look_ahead) for c in hypotheses[rows][0].clauses]

    if not sum(is_look_ahead):
        sent_best_ids, sent_best_word_ids, sent_seq_scores,\
            sent_hypotheses, sent_num_mets = _sequential_topk(timestep,
                                                              sentno,
                                                              beam_size,
                                                              diversity,
                                                              constraint_limit,
                                                              sat_tolerance,
                                                              inactive_sent,
                                                              score_sent,
                                                              hypotheses[rows],
                                                              best_ids_sent,
                                                              best_word_ids_sent,
                                                              seq_scores_sent,
                                                              do_beam_sample,
                                                              beam_temperature,
                                                              num_fill=num_fill,
                                                              stochastic=stochastic,sampler=sampler,cur_scores=cur_scores_sent,)
    else:
        raise NotImplementedError
    
    return [sent_best_ids, sent_best_word_ids, sent_seq_scores, sent_hypotheses, sent_num_mets]

def topk_huggingface_parallel(timestep: int,
                     batch_size: int,
                     beam_size: int,
                     vocab_size: int,
                     pad_token_id: int,
                     diversity: bool,
                     constraint_limit,
                     sat_tolerance: float,
                     inactive: np.array,
                     scores: np.array,
                     hypotheses: List[ConstrainedHypothesis],
                     num_fill: int,
                     do_beam_sample: bool,
                     beam_temperature: float,
                     look_ahead: int,
                     max_length: int,
                     model: transformers.PreTrainedModel,
                     temp_input_ids: torch.Tensor,
                     temp_attention_mask: torch.Tensor,
                     temp_position_ids: torch.Tensor,
                     temp_past: Tuple[torch.Tensor],
                     score_history: np.array,
                     stochastic,sampler,cur_scores,
                     model_specific_kwargs
                     ) -> Tuple[np.array, np.array, List[List[Union[ConstrainedHypothesis, None]]], List[List[int]]]:
    """
    Builds a new topk list such that the beam contains hypotheses having completed different numbers of constraints.
    These items are built from three different types: (1) the best items across the whole
    scores matrix, (2) the set of words that must follow existing constraints, and (3) k-best items from each row.

    :param batch_size: The number of segments in the batch.
    :param beam_size: The length of the beam for each segment.
    :param vocab_size: The size of vocabulary.
    :param pad_token_id:
    :param prune_factor:
    :param sat_tolerance:
    :param inactive: Array listing inactive rows (shape: (batch_size, beam_size,)).
    :param scores: The scores array (shape: (batch_size, beam_size * target_vocab_size)).
    :param hypotheses: The list of hypothesis objects. (length: (batch_size * beam_size,))
    :param num_mets: The list of int how many constraints satisfied. (length: (batch_size * beam_size,))
    :param num_fill: The number of required return beam
    :return: A tuple containing the best hypothesis rows, the best hypothesis words, the scores,
        the updated constrained hypotheses, and the updated set of inactive hypotheses.
    """
    seq_scores, raw_token_idx = torch.topk(scores, beam_size, dim=1, largest=True, sorted=True)
    best_ids = (raw_token_idx // vocab_size).cpu().numpy()
    best_word_ids = (raw_token_idx % vocab_size).cpu().numpy()
    seq_scores = seq_scores.cpu().numpy()

    scores = torch.reshape(scores, [batch_size, beam_size, -1]).cpu().numpy()

    select_best_ids = np.ones((batch_size, num_fill)) * -1
    select_best_word_ids = np.ones((batch_size, num_fill)) * -1
    select_seq_scores = np.zeros((batch_size, num_fill))
    select_hypotheses = [[None] * num_fill for _ in range(batch_size)]
    select_num_mets = [[-1] * num_fill for _ in range(batch_size)]

    # with Pool(batch_size) as pool:
    #     out = pool.map(topk_for_sent, [[i, timestep, beam_size, hypotheses, num_fill, pad_token_id, diversity, constraint_limit, sat_tolerance, 
    #               inactive[i], scores[i], best_ids[i], best_word_ids[i], seq_scores[i], cur_scores[i],
    #               do_beam_sample, beam_temperature, stochastic, sampler] for i in range(batch_size)])

    # m = mp.Process(target=topk_for_sent, args=([i, timestep, beam_size, hypotheses, num_fill, pad_token_id, diversity, constraint_limit, sat_tolerance, 
    #               inactive[i], scores[i], best_ids[i], best_word_ids[i], seq_scores[i], cur_scores[i],
    #               do_beam_sample, beam_temperature, stochastic, sampler] for i in range(batch_size)))
    # m.start()

    print('Batch size: ' + str(batch_size))
    print('Beam size: ' + str(beam_size))
    out = Parallel(n_jobs=batch_size)(delayed(topk_for_sent)(i, timestep, beam_size, hypotheses, num_fill, pad_token_id, diversity, constraint_limit, sat_tolerance, 
                  inactive[i], scores[i], best_ids[i], best_word_ids[i], seq_scores[i], cur_scores[i],
                  do_beam_sample, beam_temperature, stochastic, sampler) for i in range(batch_size))

    for i in range(len(out)):
        select_best_ids[i] = out[i][0]
        select_best_word_ids[i] = out[i][1]
        select_seq_scores[i] = out[i][2]
        select_hypotheses[i] = out[i][3]
        select_num_mets[i] = out[i][4]
    
    select_raw_token_idx = select_best_ids * vocab_size + select_best_word_ids
    return select_seq_scores, select_raw_token_idx, select_hypotheses, select_num_mets

StateType = Dict[str, torch.Tensor]
class GumbelSampler():
    """
    Referenced from Allennlp
    A `Sampler` which uses the Gumbel-Top-K trick to sample without replacement. See
    [*Stochastic Beams and Where to Find Them: The Gumbel-Top-k Trick for Sampling
    Sequences Without Replacement*, W Kool, H Van Hoof and M Welling, 2010]
    (https://api.semanticscholar.org/CorpusID:76662039).
    # Parameters
    temperature : `float`, optional (default = `1.0`)
        A `temperature` below 1.0 produces a sharper probability distribution and a `temperature`
        above 1.0 produces a flatter probability distribution.
    """

    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature

    def init_state(
        self, start_class_log_probabilities: torch.Tensor, batch_size: int, num_classes: int
    ) -> StateType:
        # shape: (batch_size, num_classes)
        zeros = start_class_log_probabilities.new_zeros((batch_size, num_classes))

        # shape: (batch_size, num_classes)
        G_phi_S = self.gumbel_with_max(start_class_log_probabilities, zeros)

        return {"G_phi_S": G_phi_S}

    def sample_nodes(
        self,
        log_probs: torch.Tensor,
        per_node_beam_size: int,
        state: StateType,
    ) -> Tuple[torch.Tensor, torch.Tensor, StateType]:
        # First apply temperature coefficient:
        # shape: (batch_size * beam_size, num_classes)
        if self.temperature != 1.0:
            _log_probs = torch.nn.functional.log_softmax(log_probs / self.temperature, dim=-1)
        else:
            _log_probs = log_probs

        # shape: (group_size,)
        phi_S = state["phi_S"]

        # shape: (group_size, num_classes)
        phi_S = phi_S.unsqueeze(-1).expand_as(_log_probs)

        # shape: (group_size, num_classes)
        phi_S_new = phi_S + _log_probs

        # shape: (group_size, 1)
        G_phi_S = state["G_phi_S"].unsqueeze(-1)

        # shape: (group_size, num_classes)
        G_phi_S_new = self.gumbel_with_max(phi_S_new, G_phi_S)

        ## Replace NaNs with very negative number.
        ## shape: (group_size, num_classes)
        ##  G_phi_S_new[G_phi_S_new.isnan()] = min_value_of_dtype(G_phi_S_new.dtype)

        # shape (both): (group_size, per_node_beam_size)
        top_G_phi_S_new, top_indices = torch.topk(G_phi_S_new, per_node_beam_size, dim=-1)

        # shape: (group_size, per_node_beam_size)
        top_log_probs = log_probs.gather(1, top_indices)

        return top_log_probs, top_indices, {"G_phi_S": G_phi_S_new}
    def gumbel(self, phi) -> torch.Tensor:
        """
        Sample `Gumbel(phi)`.
        `phi` should have shape `(batch_size, num_classes)`.
        """
        return -torch.log(-torch.log(torch.rand_like(phi))) + phi

    def gumbel_with_max(self, phi, T) -> torch.Tensor:
        """
        Sample `Gumbel(phi)` conditioned on the maximum value being equal to `T`.
        `phi` should have shape `(batch_size, num_classes)` and `T` should have
        shape `(batch_size, 1)`.
        """
        # Shape: (batch_size, num_classes)
        G_phi = self.gumbel(phi)

        # Now we find the maximum from these samples.
        # Shape: (batch_size, )
        Z, _ = G_phi.max(dim=-1)

        # Shape: (batch_size, num_classes)
        v = T - G_phi + torch.log1p(-torch.exp(G_phi - Z.unsqueeze(-1)))

        # Shape: (batch_size, num_classes)
        return T - torch.nn.functional.relu(v) - torch.log1p(torch.exp(-v.abs()))

def _sequential_topk_loop(row, inactive_row, hypotheses_row, scores_row, rank_row,
                          sampling_method, beam_size, hit):
    
    finished_candidates = set(); candidates = set()
    if inactive_row:
        return [finished_candidates, candidates]

    hyp = hypotheses_row

    # (2) add all the constraints that could extend this
    if (hyp.get_pos_state().cons_type == "Comparative_Whole") and (not hyp.get_pos_state().select_positive_phrases):
        hyp.set_pos_tokens(scores_row)
    nextones = hyp.pos_allowed()
    #print(row, "|".join([tokenizer.decode(x) for x in list(nextones)]), hyp.num_met(), hyp.met_process())


    # (3) add the single-best item after this (if it's valid)
    if sampling_method == "gumbel":
        raise NotImplementedError
    else:
        best_k = np.argsort(scores_row)[::-1][:beam_size]
    for col in best_k:
        if hyp.is_valid(col):
            nextones.add(col)
    # cands = []
    # for x in [tokenizer.decode(x) for x in nextones]:
    #     cands.append(x)
    # print(cands)

    # Now, create new candidates for each of these items
    for col in nextones:
        # print(tokenizer.decode(col),col)
        if [row, col] not in hit and (rank_row[col] < 500000 and scores_row[col] > -1000):
            new_item = hyp.advance(col)
            score = scores_row[col]
            cand = ConstrainedCandidate(row, col, score, new_item)
            if cand.hypothesis.finished() and col in cand.hypothesis.eos():
                finished_candidates.add(cand)
            else:
                candidates.add(cand)
    # Add finished candidates in finished set:
    if hyp.finished():
        best_k = np.argsort(scores_row)[::-1][:int(beam_size*5)]
        for col in best_k:
            if col in hyp.eos() and scores_row[col] > -1000:
                new_item = hyp.advance(col)
                score = scores_row[col]
                cand = ConstrainedCandidate(row, col, score, new_item)
                finished_candidates.add(cand)

    return [finished_candidates, candidates]

def _sequential_topk(timestep: int,
                     batch_idx: int,
                     beam_size: int,
                     diversity: bool,
                     constraint_limit,
                     sat_tolerance: float,
                     inactive: np.array,
                     scores: np.array,
                     hypotheses: List[ConstrainedHypothesis],
                     best_ids: np.array,
                     best_word_ids: np.array,
                     sequence_scores: np.array,
                     do_beam_sample: bool,
                     beam_temperature: float,
                     num_fill: int = None,stochastic=None,sampler=None,cur_scores=None) -> Tuple[np.array, np.array, np.array,
                                                    List[ConstrainedHypothesis], List[int]]:
    """
    Builds a new topk list such that the beam contains hypotheses having completed different numbers of constraints.
    These items are built from three different types: (1) the best items across the whole
    scores matrix, (2) the set of words that must follow existing constraints, and (3) k-best items from each row.

    :param timestep: The current decoder timestep.
    :param beam_size: The length of the beam for each segment.
    :param inactive: Array listing inactive rows (shape: (beam_size,)).
    :param scores: The scores array (shape: (beam_size, target_vocab_size)).
    :param hypotheses: The list of hypothesis objects. (length: (beam_size,))
    :param best_ids: The current list of best hypotheses (shape: (beam_size,)).
    :param best_word_ids: The parallel list of best word IDs (shape: (beam_size,)).
    :param sequence_scores: (shape: (beam_size, 1)).
    :return: A tuple containing the best hypothesis rows, the best hypothesis words, the scores,
        the updated constrained hypotheses, and the updated set of inactive hypotheses.
    """

    candidates = set()
    finished_candidates = set()
    # the best item (constrained or not) in that row
    best_next = np.argmax(scores, axis=1)
    rank = rankdata(-1 * scores, method='dense').reshape(scores.shape)

    # (1) Add all of the top-k items (which were passed) in as long as they pass the constraints
    for row, col, seq_score in zip(best_ids, best_word_ids, sequence_scores):
        row, col = int(row), int(col)
        seq_score = float(seq_score)
        new_item = hypotheses[row].advance(col)
        cand = ConstrainedCandidate(row, col, seq_score, new_item)
        if cand.hypothesis.finished():
            finished_candidates.add(cand)
        elif hypotheses[row].is_valid(col) or int(best_next[row]) == col:
            candidates.add(cand)

    hit = np.stack([best_ids, best_word_ids], axis=1).tolist()
    
    # print("best_word_ids",best_word_ids.shape)
    # cands = []
    # for x in [tokenizer.decode(x) for x in best_word_ids]:
    #     cands.append(x)
    # print("best_word_ids",cands)
    # For each hypothesis, we add (2) all the constraints that could follow it and
    # (3) the best item (constrained or not) in that row
    sampling_method = "topk"
    if stochastic:
        sampling_method = "gumbel"
    if sampling_method == "gumbel":
        if timestep == 0:
            # to init, every beam is the same at timestep 0
            state = sampler.init_state(torch.tensor(scores[0:1,:]),batch_size=1, num_classes=scores.shape[1])
            G_phi_S = state["G_phi_S"] # [1,vocab]
            new_G_phi_S = G_phi_S.expand_as(sampler.state["G_phi_S"][batch_idx])
            
            # print(f"G_phi_S: {G_phi_S.shape}, {G_phi_S}")
            # print(f"new_G_phi_S: {new_G_phi_S.shape}, {new_G_phi_S}")
            new_G_phi_S[1:,:] = -1e9
            sampler.state["G_phi_S"][batch_idx] = new_G_phi_S
            # sampler.state["phi_S"][batch_idx] = torch.tensor(scores)           
            
        else:            
            state = sampler.beam_state
            cur_state = {"phi_S":state["phi_S"][batch_idx],"G_phi_S": state["G_phi_S"][batch_idx]}
            top_log_probs, top_indices, new_state = sampler.sample_nodes(cur_scores,beam_size,cur_state)
            
            sampler.state["G_phi_S"][batch_idx] = new_state["G_phi_S"]
    best_next = np.argmax(scores, axis=1)
    # t1 = time.time()
    for row in range(beam_size):
        if inactive[row]:
            continue

        hyp = hypotheses[row]

        # (2) add all the constraints that could extend this
        if (hyp.get_pos_state().cons_type == "Comparative_Whole") and (not hyp.get_pos_state().select_positive_phrases):
            hyp.set_pos_tokens(scores[row])
        nextones = hyp.pos_allowed()
        #print(row, "|".join([tokenizer.decode(x) for x in list(nextones)]), hyp.num_met(), hyp.met_process())


        # (3) add the single-best item after this (if it's valid)
        if sampling_method == "gumbel":
            best_k = []
            perturbed_scores = sampler.state["G_phi_S"][batch_idx].cpu().detach().numpy()
            best_k = np.argsort(perturbed_scores[row])[::-1][:beam_size]
            scores= perturbed_scores
        else:
            best_k = np.argsort(scores[row])[::-1][:beam_size]
        for col in best_k:
            if hyp.is_valid(col):
                nextones.add(col)
        # cands = []
        # for x in [tokenizer.decode(x) for x in nextones]:
        #     cands.append(x)
        # print(cands)

        # Now, create new candidates for each of these items
        for col in nextones:
            # print(tokenizer.decode(col),col)
            if [row, col] not in hit and (rank[row, col] < 500000 and scores[row, col] > -1000):
                new_item = hyp.advance(col)
                score = scores[row, col]
                cand = ConstrainedCandidate(row, col, score, new_item)
                if cand.hypothesis.finished() and col in cand.hypothesis.eos():
                    finished_candidates.add(cand)
                else:
                    candidates.add(cand)
        # Add finished candidates in finished set:
        if hyp.finished():
            best_k = np.argsort(scores[row])[::-1][:int(beam_size*5)]
            for col in best_k:
                if col in hyp.eos() and scores[row, col] > -1000:
                    new_item = hyp.advance(col)
                    score = scores[row, col]
                    cand = ConstrainedCandidate(row, col, score, new_item)
                    finished_candidates.add(cand)

    # out = Parallel(n_jobs=beam_size, backend='loky')(delayed(_sequential_topk_loop)(row, inactive[row], hypotheses[row], scores[row], rank[row],
    #                       sampling_method, beam_size, hit) for row in range(beam_size))
    
    # with Pool(beam_size) as pool:
    #     out = pool.map(_sequential_topk_loop, [[row, inactive[row], hypotheses[row], scores[row], rank[row],sampling_method, beam_size, hit] for row in range(beam_size)])

    # loop = asyncio.get_event_loop()
    # loop = asyncio.new_event_loop()
    # asyncio.set_event_loop(loop)
    # looper = asyncio.gather(*[_sequential_topk_loop(row, inactive[row], hypotheses[row], scores[row], rank[row],
    #                       sampling_method, beam_size, hit) for row in range(beam_size)])                 
    # out = loop.run_until_complete(looper)
    
    # finished_candidates = finished_candidates.union(itertools.chain.from_iterable([i[0] for i in out]))
    # candidates = candidates.union(itertools.chain.from_iterable([i[1] for i in out]))

    
    # t2 = time.time()
    # print('_sequential_topk loop time: ' + str(t2-t1))

    if num_fill is not None:
        assert num_fill > beam_size, "at least select number of beam candidates"
    else:
        num_fill = beam_size

    # all the sentences finish without satisfy all constraints
    if (not candidates) and (not finished_candidates):
        # print('edge case')
        for row, col, seq_score in zip(best_ids, best_word_ids, sequence_scores):
            row, col = int(row), int(col)
            seq_score = float(seq_score)
            new_item = hypotheses[row].advance(col)
            cand = ConstrainedCandidate(row, col, seq_score, new_item)
            candidates.add(cand)

    chunk_candidates = []
    
    cands = []
    # for x in [tokenizer.decode(x.col) for x in candidates]:
    #     cands.append(x)
    # print("candidates",cands)
    if candidates:
        # Sort the candidates.
        sorted_candidates = sorted(candidates, key=attrgetter('score'), reverse=True)

        max_satisfy = max([x.hypothesis.num_met() for x in sorted_candidates])
        sorted_candidates = [x for x in sorted_candidates if x.hypothesis.num_met() >= max_satisfy - sat_tolerance]

        # Bucket candidates in each group by met order
        all_orders = set([x.hypothesis.met_process() for x in sorted_candidates])

        grouped_order_candidates = []
        grouped_candidates = [[x for x in sorted_candidates if x.hypothesis.met_process() == o] for o in all_orders]
        temperature = beam_temperature if beam_temperature is not None else 1.0
        # print("----------step-----------")        
        if do_beam_sample:
            if sampling_method == "topk":
                for group in grouped_candidates:
                    chosen_ones = []
                    raw_probs = [math.exp(x.score / temperature) + 10 ** (-10) for x in group] # almost uniform sampling
                    # print(f"scores: {[x.score for x in group[:10]]} \nraw_probs: {raw_probs[:10]}")

                    while group:
                        chosen_index, chosen_one = random.choices(list(enumerate(group)), weights=raw_probs, k=1)[0]
                        group.pop(chosen_index)
                        raw_probs.pop(chosen_index)
                        chosen_ones.append(chosen_one)
                    # print("chosen_ones",[(tokenizer.decode(x.col),x.score,x.row) for x in chosen_ones])
                    grouped_order_candidates.append(_reorder_group_POS(chosen_ones, 'score') if diversity else chosen_ones)
            elif sampling_method == "gumbel":
                for group in grouped_candidates:
                    grouped_order_candidates.append(_reorder_group_POS(group, 'score') if diversity else group)
        else:
            for group in grouped_candidates:
                grouped_order_candidates.append(_reorder_group_POS(group, 'score') if diversity else group)
        # print("----------step end-----------")
        # Group the top_i candidate of each group in chunk
        chunk_candidates = []
        num_chunk = max([len(x) for x in grouped_order_candidates])
        # print(constraint_limit)
        if constraint_limit == "hard_limit":
            groups_quota = [1] * len(grouped_order_candidates)
            groups_quota = [999999999]+[timestep+1 for x in groups_quota[1:]] 
            for i in range(num_chunk):
                chunk_i = []
                for j,g in enumerate(grouped_order_candidates):
                    if len(g) > i:
                        if groups_quota[j] > 0: 
                            chunk_i.append(g[i])
                            groups_quota[j] -= 1
                chunk_candidates.append(chunk_i)
        elif constraint_limit == "softmax_limit":
            bs_rate = 0.4
            def softmax(vec, temp):
                base = sum(math.exp(x/temp) for x in vec)
                vec = [math.exp(x/temp)/base for x in vec]
                return vec
            cands_group_len = [len(group) for group in grouped_order_candidates]
            groups_quota = [1] * len(grouped_order_candidates)
            groups_quota = [999999999]+[timestep+1 for x in groups_quota[1:]]
            total_cands = sum(cands_group_len)
            cand_probs = [x/total_cands for x in cands_group_len]
            groups_quota = [999999999]+[math.ceil(y*20) for y in softmax(cand_probs,bs_rate)[1:]]
            # print(f"group_quota: {groups_quota}")
            for i in range(num_chunk):
                chunk_i = []
                for j,g in enumerate(grouped_order_candidates):
                    if len(g) > i:
                        if groups_quota[j] > 0: 
                            chunk_i.append(g[i])
                            groups_quota[j] -= 1
                chunk_candidates.append(chunk_i)
        else:
            for i in range(num_chunk):
                chunk_i = []
                for g in grouped_order_candidates:
                    if len(g) > i:
                        chunk_i.append(g[i])
                chunk_candidates.append(chunk_i)
        # Sort candidates in each chunk by score
        chunk_candidates = [sorted(x, key=attrgetter('score'), reverse=True) for x in chunk_candidates]

    # TODO: abandon candidates which cannot meet all constraints at max length
    sorted_finished_candidates = sorted(finished_candidates, key=attrgetter('score'), reverse=True)
    if diversity:
        sorted_finished_candidates = _reorder_group_POS(sorted_finished_candidates, 'score')
    pruned_candidates = sorted_finished_candidates[:(num_fill if not candidates else beam_size)]

    num_finish = len(pruned_candidates)

    for chunk in chunk_candidates:
        if len(pruned_candidates) >= num_fill:
            break

        chunk = [x for x in chunk if x not in pruned_candidates]
        if not chunk:
            continue

        pruned_candidates.extend(chunk[:num_fill - len(pruned_candidates)])

    if num_fill > beam_size:
        if candidates:
            select_num = num_finish + beam_size
            complete_candidates = sorted(pruned_candidates[:num_finish], key=attrgetter('score'), reverse=True)
            include_candidates = sorted(pruned_candidates[num_finish:select_num], key=attrgetter('score'), reverse=True)
            extra_candidates = sorted(pruned_candidates[select_num:], key=attrgetter('score'), reverse=True)
            pruned_candidates = complete_candidates + include_candidates + extra_candidates
    else:
        pruned_candidates = sorted(pruned_candidates, key=attrgetter('score'), reverse=True)

    num_pruned_candidates = len(pruned_candidates)

    inactive = np.zeros(num_fill)
    inactive[:num_pruned_candidates] = 0

    # Pad the beam so array assignment still works
    if num_pruned_candidates < num_fill:
        inactive[num_pruned_candidates:] = 1
        pruned_candidates += [pruned_candidates[num_pruned_candidates - 1]] * (num_fill - num_pruned_candidates)

    assert len(pruned_candidates) == num_fill, 'candidates number mismatch'
    # print("prunned_candidates",[x.hypothesis.num_met() for x in pruned_candidates])
    return (np.array([x.row for x in pruned_candidates]),
            np.array([x.col for x in pruned_candidates]),
            np.array([x.score for x in pruned_candidates]),
            [x.hypothesis for x in pruned_candidates],
            [x.hypothesis.num_met() for x in pruned_candidates])


alpha = 0.25


def _look_ahead_topk(sentno: int,
                     timestep: int,
                     beam_size: int,
                     diversity: bool,
                     sat_tolerance: float,
                     inactive: np.array,
                     scores: np.array,
                     hypotheses: List[ConstrainedHypothesis],
                     best_ids: np.array,
                     best_word_ids: np.array,
                     sequence_scores: np.array,
                     do_beam_sample: bool,
                     beam_temperature: float,
                     num_fill: int,
                     look_ahead: int,
                     max_length: int,
                     model: transformers.PreTrainedModel,
                     temp_input_ids: torch.Tensor,
                     temp_attention_mask: torch.Tensor,
                     temp_position_ids: torch.Tensor,
                     temp_past: Tuple[torch.Tensor],
                     score_history: np.array,
                     model_specific_kwargs,
                     chunk_size=20) -> Tuple[np.array, np.array, np.array,
                                                       List[ConstrainedHypothesis], List[int]]:
    """
    Builds a new topk list such that the beam contains hypotheses having completed different numbers of constraints.
    These items are built from three different types: (1) the best items across the whole
    scores matrix, (2) the set of words that must follow existing constraints, and (3) k-best items from each row.

    :param timestep: The current decoder timestep.
    :param beam_size: The length of the beam for each segment.
    :param inactive: Array listing inactive rows (shape: (beam_size,)).
    :param scores: The scores array (shape: (beam_size, target_vocab_size)).
    :param hypotheses: The list of hypothesis objects. (length: (beam_size,))
    :param best_ids: The current list of best hypotheses (shape: (beam_size,)).
    :param best_word_ids: The parallel list of best word IDs (shape: (beam_size,)).
    :param sequence_scores: (shape: (beam_size, 1)).
    :return: A tuple containing the best hypothesis rows, the best hypothesis words, the scores,
        the updated constrained hypotheses, and the updated set of inactive hypotheses.
    """

    candidates = set()
    finished_candidates = set()
    # the best item (constrained or not) in that row
    best_next = np.argmax(scores, axis=1)
    rank = rankdata(-1 * scores, method='dense').reshape(scores.shape)

    # (1) Add all of the top-k items (which were passed) in as long as they pass the constraints
    for row, col, seq_score in zip(best_ids, best_word_ids, sequence_scores):
        row, col = int(row), int(col)
        seq_score = float(seq_score)
        new_item = hypotheses[row].advance(col)
        cand = ConstrainedCandidate(row, col, seq_score, new_item)
        if cand.hypothesis.finished():
            finished_candidates.add(cand)
        elif hypotheses[row].is_valid(col) or int(best_next[row]) == col:
            candidates.add(cand)

    hit = np.stack([best_ids, best_word_ids], axis=1).tolist()
    # For each hypothesis, we add (2) all the constraints that could follow it and
    # (3) the best item (constrained or not) in that row
    best_next = np.argmax(scores, axis=1)
    for row in range(beam_size):
        if inactive[row]:
            continue

        hyp = hypotheses[row]

        # (2) add all the constraints that could extend this
        nextones = hyp.pos_allowed()

        # (3) add the single-best item after this (if it's valid)
        best_k = np.argsort(scores[row])[::-1][:beam_size]
        for col in best_k:
            if hyp.is_valid(col):
                nextones.add(col)

        # Now, create new candidates for each of these items
        for col in nextones:
            if [row, col] not in hit and (rank[row, col] < 500000):
                new_item = hyp.advance(col)
                score = scores[row, col]
                cand = ConstrainedCandidate(row, col, score, new_item)
                if cand.hypothesis.finished() and col in cand.hypothesis.eos():
                    finished_candidates.add(cand)
                else:
                    candidates.add(cand)

        # this is for encourage short ending for commonGen, you might not need it
        if hyp.finished():
            best_k = np.argsort(scores[row])[::-1][:int(beam_size*10)]
            for col in best_k:
                if col in hyp.eos():
                    new_item = hyp.advance(col)
                    score = scores[row, col]
                    cand = ConstrainedCandidate(row, col, score, new_item)
                    finished_candidates.add(cand)

    if num_fill is not None:
        assert num_fill > beam_size, "at least select number of beam candidates"
    else:
        num_fill = beam_size

    # all the sentences finish without satisfy all constraints
    if (not candidates) and (not finished_candidates):
        # print('edge case')
        for row, col, seq_score in zip(best_ids, best_word_ids, sequence_scores):
            row, col = int(row), int(col)
            seq_score = float(seq_score)
            new_item = hypotheses[row].advance(col)
            cand = ConstrainedCandidate(row, col, seq_score, new_item)
            candidates.add(cand)


    chunk_candidates = []
    if candidates:
        # Sort the candidates.
        all_sorted_candidates = sorted(candidates, key=attrgetter('score'), reverse=True)

        max_satisfy = max([x.hypothesis.num_met() for x in all_sorted_candidates])
        all_sorted_candidates = [x for x in all_sorted_candidates if x.hypothesis.num_met() >= max_satisfy - sat_tolerance]

        next_look_aheads, first_token_prob, second_token_prob = [], [], []
        for start in range(0, len(all_sorted_candidates), chunk_size):
            sorted_candidates = all_sorted_candidates[start: start + chunk_size]

            back_ptrs = temp_input_ids.new([x.row for x in sorted_candidates])
            curr_ids = temp_input_ids.new([x.col for x in sorted_candidates])
            input_ids = torch.cat([temp_input_ids[back_ptrs, :], curr_ids[:, None]], dim=-1)
            attention_mask = temp_attention_mask[back_ptrs, :]
            position_ids = temp_position_ids[back_ptrs, :]
            past = _reorder_cache(temp_past, back_ptrs)

            ahead_ids = None
            next_look_ahead = [c.hypothesis.next_look_ahead() for c in sorted_candidates]
            next_look_aheads += next_look_ahead
            num_ahead = [len(x) for x in next_look_ahead]
            if sum(num_ahead):
                ahead_ids = torch.cat([input_ids.new(x) for x in next_look_ahead])
                ahead_index = input_ids.new([i for i, hs in enumerate(next_look_ahead) for x in hs])

            num_candidates = len(sorted_candidates)
            output_probs = []
            for t in range(look_ahead):
                model_inputs = model.prepare_inputs_for_generation(
                    input_ids, past=past, attention_mask=attention_mask, use_cache=model.config.use_cache, **model_specific_kwargs
                )
                model_inputs["attention_mask"] = attention_mask
                model_inputs["position_ids"] = position_ids[:, -1].unsqueeze(-1)

                outputs = model(**model_inputs)  # (num_candidates, input_len, vocab_size)
                next_token_logits = outputs[0][:, -1, :]  # (num_candidates, vocab_size)

                past = outputs[1]
                if model.config.is_encoder_decoder:
                    next_token_logits = model.adjust_logits_during_generation(
                        next_token_logits, cur_len=input_ids.shape[-1], max_length=max_length
                    )

                ahead_scores = F.log_softmax(next_token_logits, dim=-1)  # (num_candidates, vocab_size)
                next_tokens = torch.argmax(ahead_scores, dim=-1)
                output_probs.append(ahead_scores)

                if ahead_ids is not None:
                    past = _reorder_cache(past, torch.arange(num_candidates))
                    input_ids = input_ids[: num_candidates]
                    attention_mask = attention_mask[: num_candidates]
                    position_ids = position_ids[: num_candidates]
                    next_tokens = next_tokens[: num_candidates]

                    ahead_past = _reorder_cache(past, ahead_index)
                    ahead_input_ids = torch.cat([input_ids[ahead_index, :], ahead_ids[:, None]], dim=-1)
                    ahead_prev_att_mask = attention_mask[ahead_index, :]
                    ahead_attention_mask = ahead_prev_att_mask if model.config.is_encoder_decoder else \
                        torch.cat([ahead_prev_att_mask, ahead_prev_att_mask.new_ones((ahead_prev_att_mask.shape[0], 1))], dim=-1)
                    ahead_prev_pos_ids = position_ids[ahead_index, :]
                    ahead_position_ids = torch.cat([ahead_prev_pos_ids, (ahead_prev_pos_ids[:, -1] + 1).unsqueeze(-1)], dim=1)

                input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
                attention_mask = attention_mask if model.config.is_encoder_decoder else \
                    torch.cat([attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1)
                position_ids = torch.cat([position_ids, (position_ids[:, -1] + 1).unsqueeze(-1)], dim=1)

                if ahead_ids is not None:
                    def concat_pasts(past,ahead_past):
                        # we are adding ahead_past after past to extend the batch
                        # past looks like [48,2,20,20,4,64] the first 2 dimensions are tuples now. the third dim is batch
                        return tuple(
                            tuple(torch.cat([past_state1,past_state2]) for past_state1,past_state2 in zip(layer_past1,layer_past2))
                            for layer_past1,layer_past2 in zip(past,ahead_past)
                        )
                    # print(past[0][0].shape,ahead_past[0][0].shape)
                    # past = tuple(torch.cat([past[i], ahead_past[i]], dim=1) for i in range(len(past)))
                    past = concat_pasts(past,ahead_past)
                    input_ids = torch.cat([input_ids, ahead_input_ids], dim=0)
                    attention_mask = torch.cat([attention_mask, ahead_attention_mask], dim=0)
                    position_ids = torch.cat([position_ids, ahead_position_ids], dim=0)

            chunk_first_token_prob = torch.cat([x[:num_candidates][:, None] for x in output_probs], dim=1)
            first_token_prob.append(chunk_first_token_prob)

            chunk_second_token_prob = [None] * chunk_size
            if ahead_ids is not None:
                chunk_second_token_prob = torch.cat([x[num_candidates:][:, None] for x in output_probs[1:]], dim=1)
                chunk_second_token_prob = torch.split(chunk_second_token_prob, num_ahead, dim=0)
            second_token_prob.extend(chunk_second_token_prob)

        first_token_prob = torch.cat(first_token_prob, dim=0)

        for i, cand in enumerate(all_sorted_candidates):
            max_prob = -10000.0
            chosen = None

            if cand.hypothesis.clause_in_process():
                for (front, back) in cand.hypothesis.literal_in_process:
                    #front_score = cand.score if len(front) == 1 else score_history[cand.row][-(len(front)-1)]
                    back_score = first_token_prob[i][0][back[0]].item()
                    #in_process_score = [front_score, back_score]
                    in_process_score = [back_score]
                    if np.mean(in_process_score) > max_prob:
                        chosen = front + back
                    max_prob = max(max_prob, np.mean(in_process_score))

            remain_clauses = [c for c in cand.hypothesis.clauses if not c.satisfy and c.look_ahead]
            remain_literals = [l for c in remain_clauses for l in c.positive]

            for literal in remain_literals:
                first_score = first_token_prob[i][:, literal[0]].tolist()
                greedy_score = torch.max(first_token_prob[i], dim=-1)[0].tolist()
                lead_score = [greedy_score[:n] for n in range(len(greedy_score))]

                if len(literal) > 1:
                    second_score = second_token_prob[i][next_look_aheads[i].index(literal[0])][:, literal[1]].tolist()
                    literal_score = [np.mean([f, s]) for (f, s) in zip(first_score[:-1], second_score)]
                else:
                    literal_score = first_score

                if max(literal_score) > max_prob:
                    chosen = literal
                max_prob = max(max_prob, max(literal_score))

            cand.rank = cand.score / (timestep + 1)
            if max_prob > -10000.0 and cand.col not in cand.hypothesis.eos():
                cand.rank += alpha * max_prob
                cand.ahead = alpha * max_prob
            # for visualization only
            cand.literal = chosen

        all_sorted_candidates = sorted(all_sorted_candidates, key=attrgetter('rank'), reverse=True)

        # Bucket candidates in each group by met order
        all_orders = set([x.hypothesis.met_order() for x in all_sorted_candidates])
        grouped_candidates = [[x for x in all_sorted_candidates if x.hypothesis.met_order() == o] for o in all_orders]

        processed_grouped_candidates = []
        for g in grouped_candidates:
            all_ahead = [c.ahead for c in g if c.ahead is not None]

            if not all_ahead:
                processed_grouped_candidates.append(g)
                continue

            for c in g:
                c.rank = c.rank if c.ahead is not None else c.rank #+ min(all_ahead)

            processed_grouped_candidates.append(sorted(g, key=attrgetter('rank'), reverse=True))

        grouped_order_candidates = []
        temperature = beam_temperature if beam_temperature is not None else 1.0

        if do_beam_sample:
            for group in processed_grouped_candidates:
                chosen_ones = []
                raw_probs = [math.exp(x.rank / temperature) + 10 ** (-10) for x in group]
                while group:
                    chosen_index, chosen_one = random.choices(list(enumerate(group)), weights=raw_probs, k=1)[0]
                    group.pop(chosen_index)
                    raw_probs.pop(chosen_index)
                    chosen_ones.append(chosen_one)

                grouped_order_candidates.append(_reorder_group_POS(chosen_ones, 'rank') if diversity else chosen_ones)
        else:
            for group in processed_grouped_candidates:
                grouped_order_candidates.append(_reorder_group_POS(group, 'rank') if diversity else group)

        # Group the top_i candidate of each group in chunk
        chunk_candidates = []
        num_chunk = max([len(x) for x in grouped_order_candidates])
        for i in range(num_chunk):
            chunk_i = []
            for g in grouped_order_candidates:
                if len(g) > i:
                    chunk_i.append(g[i])
            chunk_candidates.append(chunk_i)
        # Sort candidates in each chunk by score
        chunk_candidates = [sorted(x, key=attrgetter('rank'), reverse=True) for x in chunk_candidates]

    # TODO: abandon candidates which cannot meet all constraints at max length
    sorted_finished_candidates = sorted(finished_candidates, key=attrgetter('score'), reverse=True)
    if diversity:
        sorted_finished_candidates = _reorder_group_POS(sorted_finished_candidates, 'score')
    pruned_candidates = sorted_finished_candidates[:(num_fill if not candidates else beam_size)]

    num_finish = len(pruned_candidates)

    for chunk in chunk_candidates:
        if len(pruned_candidates) >= num_fill:
            break

        chunk = [x for x in chunk if x not in pruned_candidates]
        if not chunk:
            continue

        pruned_candidates.extend(chunk[:num_fill - len(pruned_candidates)])

    if num_fill > beam_size:
        select_num = num_finish + beam_size
        complete_candidates = sorted(pruned_candidates[:num_finish], key=attrgetter('score'), reverse=True)
        include_candidates = sorted(pruned_candidates[num_finish:select_num], key=attrgetter('score'), reverse=True)
        extra_candidates = sorted(pruned_candidates[select_num:], key=attrgetter('score'), reverse=True)
        pruned_candidates = complete_candidates + include_candidates + extra_candidates
    else:
        pruned_candidates = sorted(pruned_candidates, key=attrgetter('score'), reverse=True)

    num_pruned_candidates = len(pruned_candidates)

    inactive = np.zeros(num_fill)
    inactive[:num_pruned_candidates] = 0

    # Pad the beam so array assignment still works
    if num_pruned_candidates < num_fill:
        inactive[num_pruned_candidates:] = 1
        pruned_candidates += [pruned_candidates[num_pruned_candidates - 1]] * (num_fill - num_pruned_candidates)

    assert len(pruned_candidates) == num_fill, 'candidates number mismatch'

    return (np.array([x.row for x in pruned_candidates]),
            np.array([x.col for x in pruned_candidates]),
            np.array([x.score for x in pruned_candidates]),
            [x.hypothesis for x in pruned_candidates],
            [x.hypothesis.num_met() for x in pruned_candidates])


# def _reorder_cache(past: Tuple, beam_idx: torch.Tensor) -> Tuple[torch.Tensor]:
#     return tuple(layer_past.index_select(1, beam_idx) for layer_past in past)


def _reorder_group_POS(group, metric):
    if not group:
        return group
    POS_map = [(x, x.hypothesis.get_pos_word(POS_set=[60016, 60007], N=2)) for x in group]
    all_POS_set = set([x[1] for x in POS_map])
    grouped_items = [[x[0] for x in POS_map if x[1] == o] for o in all_POS_set]

    chunk_items = []
    num_chunk = max([len(x) for x in grouped_items])
    for i in range(num_chunk):
        chunk_i = []
        for g in grouped_items:
            if len(g) > i:
                chunk_i.append(g[i])
        chunk_items.append(chunk_i)
    # Sort candidates in each chunk by score
    chunk_items = [sorted(x, key=attrgetter(metric), reverse=True) for x in chunk_items]
    flatten = [x for ch in chunk_items for x in ch]
    return flatten


@torch.no_grad()
def generate(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    max_length: Optional[int] = None,
    min_length: Optional[int] = None,
    do_sample: Optional[bool] = None,
    early_stopping: Optional[bool] = None,
    num_beams: Optional[int] = None,
    temperature: Optional[float] = None,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    repetition_penalty: Optional[float] = None,
    bad_words_ids: Optional[Iterable[int]] = None,
    bos_token_id: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[int] = None,
    length_penalty: Optional[float] = None,
    no_repeat_ngram_size: Optional[int] = None,
    num_return_sequences: Optional[int] = None,
    num_sequences_per_return: Optional[int] = None,
    attention_mask: Optional[torch.LongTensor] = None,
    decoder_start_token_id: Optional[int] = None,
    use_cache: Optional[bool] = None,
    constraints: Optional[List[Optional[ConstrainedHypothesis]]] = None,
    diversity: Optional[bool] = None,
    constraint_limit = None,
    sat_tolerance: Optional[int] = None,
    do_beam_sample: Optional[bool] = None,
    beam_temperature: Optional[float] = None,
    look_ahead: Optional[int] = None,
    return_all=None,
    length_penalties=None,
    lp_return_size=None,
    stochastic = None,
    **model_specific_kwargs
) -> torch.LongTensor:
    r""" Generates sequences for models with a LM head. The method currently supports greedy decoding, beam-search decoding, sampling with temperature, sampling with top-k or nucleus sampling.

    Adapted in part from `Facebook's XLM beam search code`_.

    .. _`Facebook's XLM beam search code`:
       https://github.com/facebookresearch/XLM/blob/9e6f6814d17be4fe5b15f2e6c43eb2b2d76daeb4/src/model/transformer.py#L529


    Parameters:

        input_ids: (`optional`) `torch.LongTensor` of shape `(batch_size, sequence_length)`
            The sequence used as a prompt for the generation. If `None` the method initializes
            it as an empty `torch.LongTensor` of shape `(1,)`.

        max_length: (`optional`) int
            The max length of the sequence to be generated.  Between `min_length` and infinity. Default to 20.

        min_length: (`optional`) int
            The min length of the sequence to be generated.  Between 0 and infinity. Default to 0.

        do_sample: (`optional`) bool
            If set to `False` greedy decoding is used. Otherwise sampling is used. Defaults to `False` as defined in `configuration_utils.PretrainedConfig`.

        early_stopping: (`optional`) bool
            if set to `True` beam search is stopped when at least `num_beams` sentences finished per batch. Defaults to `False` as defined in `configuration_utils.PretrainedConfig`.

        num_beams: (`optional`) int
            Number of beams for beam search. Must be between 1 and infinity. 1 means no beam search. Default to 1.

        temperature: (`optional`) float
            The value used to module the next token probabilities. Must be strictly positive. Default to 1.0.

        top_k: (`optional`) int
            The number of highest probability vocabulary tokens to keep for top-k-filtering. Between 1 and infinity. Default to 50.

        top_p: (`optional`) float
            The cumulative probability of parameter highest probability vocabulary tokens to keep for nucleus sampling. Must be between 0 and 1. Default to 1.

        repetition_penalty: (`optional`) float
            The parameter for repetition penalty. Between 1.0 and infinity. 1.0 means no penalty. Default to 1.0.

        pad_token_id: (`optional`) int
            Padding token. Default to specicic model pad_token_id or None if it does not exist.

        bos_token_id: (`optional`) int
            BOS token. Defaults to `bos_token_id` as defined in the models config.

        eos_token_id: (`optional`) int
            EOS token. Defaults to `eos_token_id` as defined in the models config.

        length_penalty: (`optional`) float
            Exponential penalty to the length. Default to 1.

        no_repeat_ngram_size: (`optional`) int
            If set to int > 0, all ngrams of size `no_repeat_ngram_size` can only occur once.
        bad_words_ids: (`optional`) list of lists of int
            `bad_words_ids` contains tokens that are not allowed to be generated. In order to get the tokens of the words that should not appear in the generated text, use `tokenizer.encode(bad_word, add_prefix_space=True)`.

        num_return_sequences: (`optional`) int
            The number of independently computed returned sequences for each element in the batch. Default to 1.

        attention_mask (`optional`) obj: `torch.LongTensor` of same shape as `input_ids`
            Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
            Defaults to `None`.

            `What are attention masks? <../glossary.html#attention-mask>`__

        decoder_start_token_id=None: (`optional`) int
            If an encoder-decoder model starts decoding with a different token than BOS.
            Defaults to `None` and is changed to `BOS` later.

        use_cache: (`optional`) bool
            If `use_cache` is True, past key values are used to speed up decoding if applicable to model. Defaults to `True`.

        model_specific_kwargs: (`optional`) dict
            Additional model specific kwargs will be forwarded to the `forward` function of the model.

    Return:

        output: `torch.LongTensor` of shape `(batch_size * num_return_sequences, sequence_length)`
            sequence_length is either equal to max_length or shorter if all batches finished early due to the `eos_token_id`

    """

    # We cannot generate if the model does not have a LM head
    if self.get_output_embeddings() is None:
        raise AttributeError(
            "You tried to generate sequences with a model that does not have a LM Head."
            "Please use another model class (e.g. `OpenAIGPTLMHeadModel`, `XLNetLMHeadModel`, `GPT2LMHeadModel`, `CTRLLMHeadModel`, `T5WithLMHeadModel`, `TransfoXLLMHeadModel`, `XLMWithLMHeadModel`, `BartForConditionalGeneration` )"
        )

    max_length = max_length if max_length is not None else self.config.max_length
    min_length = min_length if min_length is not None else self.config.min_length
    do_sample = do_sample if do_sample is not None else self.config.do_sample
    early_stopping = early_stopping if early_stopping is not None else self.config.early_stopping
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    num_beams = num_beams if num_beams is not None else self.config.num_beams
    temperature = temperature if temperature is not None else self.config.temperature
    top_k = top_k if top_k is not None else self.config.top_k
    top_p = top_p if top_p is not None else self.config.top_p
    repetition_penalty = repetition_penalty if repetition_penalty is not None else self.config.repetition_penalty
    bos_token_id = bos_token_id if bos_token_id is not None else self.config.bos_token_id
    pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
    length_penalty = length_penalty if length_penalty is not None else self.config.length_penalty
    no_repeat_ngram_size = (
        no_repeat_ngram_size if no_repeat_ngram_size is not None else self.config.no_repeat_ngram_size
    )
    bad_words_ids = bad_words_ids if bad_words_ids is not None else self.config.bad_words_ids
    num_return_sequences = (
        num_return_sequences if num_return_sequences is not None else self.config.num_return_sequences
    )
    decoder_start_token_id = (
        decoder_start_token_id if decoder_start_token_id is not None else self.config.decoder_start_token_id
    )

    if input_ids is not None:
        batch_size = input_ids.shape[0]  # overriden by the input batch_size
    else:
        batch_size = 1

    assert isinstance(max_length, int) and max_length > 0, "`max_length` should be a strictly positive integer."
    assert isinstance(min_length, int) and min_length >= 0, "`min_length` should be a positive integer."
    assert isinstance(do_sample, bool), "`do_sample` should be a boolean."
    assert isinstance(early_stopping, bool), "`early_stopping` should be a boolean."
    assert isinstance(use_cache, bool), "`use_cache` should be a boolean."
    assert isinstance(num_beams, int) and num_beams > 0, "`num_beams` should be a strictly positive integer."
    assert temperature > 0, "`temperature` should be strictly positive."
    assert isinstance(top_k, int) and top_k >= 0, "`top_k` should be a positive integer."
    assert 0 <= top_p <= 1, "`top_p` should be between 0 and 1."
    assert repetition_penalty >= 1.0, "`repetition_penalty` should be >= 1."
    assert input_ids is not None or (
        isinstance(bos_token_id, int) and bos_token_id >= 0
    ), "If input_ids is not defined, `bos_token_id` should be a positive integer."
    assert pad_token_id is None or (
        isinstance(pad_token_id, int) and (pad_token_id >= 0)
    ), "`pad_token_id` should be a positive integer."
    assert (eos_token_id is None) or (
        isinstance(eos_token_id, int) and (eos_token_id >= 0)
    ), "`eos_token_id` should be a positive integer."
    assert length_penalty > 0, "`length_penalty` should be strictly positive."
    assert (
        isinstance(no_repeat_ngram_size, int) and no_repeat_ngram_size >= 0
    ), "`no_repeat_ngram_size` should be a positive integer."
    assert (
        isinstance(num_return_sequences, int) and num_return_sequences > 0
    ), "`num_return_sequences` should be a strictly positive integer."
    assert (
        bad_words_ids is None or isinstance(bad_words_ids, list) and isinstance(bad_words_ids[0], list)
    ), "`bad_words_ids` is either `None` or a list of lists of tokens that should not be generated"

    if input_ids is None:
        assert isinstance(bos_token_id, int) and bos_token_id >= 0, (
            "you should either supply a context to complete as `input_ids` input "
            "or a `bos_token_id` (integer >= 0) as a first token to start the generation."
        )
        input_ids = torch.full(
            (batch_size, 1), bos_token_id, dtype=torch.long, device=next(self.parameters()).device,
        )
    else:
        assert input_ids.dim() == 2, "Input prompt should be of shape (batch_size, sequence length)."

    # not allow to duplicate outputs when greedy decoding
    if do_sample is False:
        if num_beams == 1:
            # no_beam_search greedy generation conditions
            assert (
                num_return_sequences == 1
            ), "Greedy decoding will always produce the same output for num_beams == 1 and num_return_sequences > 1. Please set num_return_sequences = 1"

        else:
            # beam_search greedy generation conditions
            assert (
                num_beams >= num_return_sequences
            ), "Greedy beam search decoding cannot return more sequences than it has beams. Please set num_beams >= num_return_sequences"

    # create attention mask if necessary
    # TODO (PVP): this should later be handled by the forward fn() in each model in the future see PR 3140
    if (attention_mask is None) and (pad_token_id is not None) and (pad_token_id in input_ids):
        attention_mask = input_ids.ne(pad_token_id).long()
    elif attention_mask is None:
        attention_mask = input_ids.new_ones(input_ids.shape)

    # set pad_token_id to eos_token_id if not set. Important that this is done after
    # attention_mask is created
    if pad_token_id is None and eos_token_id is not None:
        logger.warning(
            "Setting `pad_token_id` to {} (first `eos_token_id`) to generate sequence".format(eos_token_id)
        )
        pad_token_id = eos_token_id

    # current position and vocab size
    if hasattr(self.config, "vocab_size"):
        vocab_size = self.config.vocab_size
    elif (
        self.config.is_encoder_decoder
        and hasattr(self.config, "decoder")
        and hasattr(self.config.decoder, "vocab_size")
    ):
        vocab_size = self.config.decoder.vocab_size

    # set effective batch size and effective batch multiplier according to do_sample
    if do_sample:
        effective_batch_size = batch_size * num_return_sequences
        effective_batch_mult = num_return_sequences
    else:
        effective_batch_size = batch_size
        effective_batch_mult = 1

    if self.config.is_encoder_decoder:
        if decoder_start_token_id is None:
            decoder_start_token_id = bos_token_id

        assert (
            decoder_start_token_id is not None
        ), "decoder_start_token_id or bos_token_id has to be defined for encoder-decoder generation"
        assert hasattr(self, "get_encoder"), "{} should have a 'get_encoder' function defined".format(self)
        assert callable(self.get_encoder), "{} should be a method".format(self.get_encoder)

        # get encoder and store encoder outputs
        encoder = self.get_encoder()

        encoder_outputs: tuple = encoder(input_ids, attention_mask=attention_mask)

    # Expand input ids if num_beams > 1 or num_return_sequences > 1
    if num_return_sequences > 1 or num_beams > 1:
        input_ids_len = input_ids.shape[-1]
        input_ids = input_ids.unsqueeze(1).expand(batch_size, effective_batch_mult * num_beams, input_ids_len)
        attention_mask = attention_mask.unsqueeze(1).expand(
            batch_size, effective_batch_mult * num_beams, input_ids_len
        )

        input_ids = input_ids.contiguous().view(
            effective_batch_size * num_beams, input_ids_len
        )  # shape: (batch_size * num_return_sequences * num_beams, cur_len)
        attention_mask = attention_mask.contiguous().view(
            effective_batch_size * num_beams, input_ids_len
        )  # shape: (batch_size * num_return_sequences * num_beams, cur_len)

    if self.config.is_encoder_decoder:
        # create empty decoder_input_ids
        input_ids = torch.full(
            (effective_batch_size * num_beams, 1),
            decoder_start_token_id,
            dtype=torch.long,
            device=next(self.parameters()).device,
        )
        cur_len = 1

        assert (
            batch_size == encoder_outputs[0].shape[0]
        ), f"expected encoder_outputs[0] to have 1st dimension bs={batch_size}, got {encoder_outputs[0].shape[0]} "

        # expand batch_idx to assign correct encoder output for expanded input_ids (due to num_beams > 1 and num_return_sequences > 1)
        expanded_batch_idxs = (
            torch.arange(batch_size)
            .view(-1, 1)
            .repeat(1, num_beams * effective_batch_mult)
            .view(-1)
            .to(input_ids.device)
        )
        # expand encoder_outputs
        encoder_outputs = (encoder_outputs[0].index_select(0, expanded_batch_idxs), *encoder_outputs[1:])

    else:
        encoder_outputs = None
        cur_len = input_ids.shape[-1]
    if num_beams > 1:
        output = _generate_beam_search(
            self,
            input_ids=input_ids,
            cur_len=cur_len,
            max_length=max_length,
            min_length=min_length,
            do_sample=do_sample,
            early_stopping=early_stopping,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            bad_words_ids=bad_words_ids,
            bos_token_id=bos_token_id,
            pad_token_id=pad_token_id,
            decoder_start_token_id=decoder_start_token_id,
            eos_token_id=eos_token_id,
            batch_size=effective_batch_size,
            num_return_sequences=num_return_sequences,
            length_penalty=length_penalty,
            num_beams=num_beams,
            vocab_size=vocab_size,
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            use_cache=use_cache,
            constraints=constraints,
            diversity=diversity,
            constraint_limit=constraint_limit,
            sat_tolerance=sat_tolerance,
            num_sequences_per_return=num_sequences_per_return,
            do_beam_sample=do_beam_sample,
            beam_temperature=beam_temperature,
            look_ahead=look_ahead,
            return_all=return_all,
            length_penalties=length_penalties,
            lp_return_size=lp_return_size,
            stochastic = stochastic,
            model_specific_kwargs=model_specific_kwargs,
        )
    else:
        raise NotImplementedError
    return output


class BeamHypotheses(object):
    def __init__(self, num_beams, max_length, length_penalty, early_stopping):
        """
        Initialize n-best list of hypotheses.
        """
        self.max_length = max_length - 1  # ignoring bos_token
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.num_beams = num_beams * 2
        self.beams = []
        self.deleted_beams = []
        self.worst_score = 1e9

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.beams)

    def add(self, hyp, sum_logprobs, num_met):
        """
        Add a new hypothesis to the list.
        """
        score = sum_logprobs / len(hyp) ** self.length_penalty
        #score = sum_logprobs / math.pow((5 + len(hyp) + 1) / 6.0, self.length_penalty)
        if len(self) < self.num_beams or score > self.worst_score:
            self.beams.append((score, hyp, num_met,sum_logprobs))
            if len(self) > self.num_beams:
                sorted_scores = sorted([(s, idx) for idx, (s, _, _,_) in enumerate(self.beams)])
                self.deleted_beams.append(self.beams[sorted_scores[0][1]])
                del self.beams[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs, cur_len=None):
        """
        If there are enough hypotheses and that none of the hypotheses being generated
        can become better than the worst one in the heap, then we are done with this sentence.
        """

        if len(self) < self.num_beams:
            return False
        elif self.early_stopping:
            return True
        else:
            if cur_len is None:
                cur_len = self.max_length
            cur_score = best_sum_logprobs / cur_len ** self.length_penalty
            #cur_score = best_sum_logprobs / math.pow((5 + cur_len + 1) / 6.0, self.length_penalty)
            ret = self.worst_score >= cur_score
            return ret


# from transformers import AutoTokenizer, AutoModelWithLMHead
# tokenizer = AutoTokenizer.from_pretrained('gpt2-large')


def _generate_beam_search(
        self,
        input_ids,
        cur_len,
        max_length,
        min_length,
        do_sample,
        early_stopping,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        no_repeat_ngram_size,
        bad_words_ids,
        bos_token_id,
        pad_token_id,
        eos_token_id,
        decoder_start_token_id,
        batch_size,
        num_return_sequences,
        length_penalty,
        num_beams,
        vocab_size,
        encoder_outputs,
        attention_mask,
        use_cache,
        constraints,
        diversity,
        constraint_limit,
        sat_tolerance,
        num_sequences_per_return,
        do_beam_sample,
        beam_temperature,
        look_ahead,
        return_all,
        length_penalties,
        lp_return_size,
        stochastic,
        model_specific_kwargs,
):
    """ Generate sequences for each example with beam search.
    """
    # end condition
    cons_eos = constraints[0].eos()

    last_non_masked_idx = (torch.sum(attention_mask, dim=1) - 1).int()
    start_idx = (last_non_masked_idx).view(-1, 1).repeat(1, self.config.vocab_size).unsqueeze(1).long()

    init_length = cur_len
    position_ids = torch.tensor([list(range(init_length)) for i in range(input_ids.shape[0])])
    for i, position_ids_slice in enumerate(position_ids):
        position_ids_slice[last_non_masked_idx[i]:] = position_ids_slice[last_non_masked_idx[i]]
    position_ids = position_ids.to(input_ids.device)

    # generated hypotheses
    generated_hyps = [
        BeamHypotheses(num_beams, max_length, length_penalty, early_stopping=early_stopping)
        for _ in range(batch_size)
    ]

    # scores for each sentence in the beam
    beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)

    # for greedy decoding it is made sure that only tokens of the first beam are considered to avoid sampling the exact same tokens three times
    if do_sample is False:
        beam_scores[:, 1:] = -1e9 # should have no effect on the 1st iteration
    beam_scores = beam_scores.view(-1)  # shape (batch_size * num_beams,)
    sampler = None
    if stochastic:
        sampler = GumbelSampler()
        phi_S = torch.zeros((batch_size,num_beams, vocab_size), dtype=torch.float, device=input_ids.device)
        G_phi_S = torch.zeros((batch_size,num_beams, vocab_size), dtype=torch.float, device=input_ids.device)
        new_state = {"phi_S":phi_S,"G_phi_S":G_phi_S}
        sampler.state = new_state
        
        beam_phi_S = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_G_phi_S = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        if do_sample is False:
            beam_phi_S[:, 1:] = -1e9
        beam_state = {"phi_S":beam_phi_S,"G_phi_S":beam_G_phi_S}
        sampler.beam_state = beam_state
        

    # cache compute states\
    past = encoder_outputs  # defined for encoder-decoder models, None for decoder-only models

    # done sentences
    done = [False for _ in range(batch_size)]

    # init number of met clauses
    num_mets = [x.num_met() for x in constraints]

    # history of token score
    score_history = None

    while cur_len < max_length:
        t3 = time.time()
        model_inputs = self.prepare_inputs_for_generation(
            input_ids, past_key_values=past, attention_mask=attention_mask, use_cache=use_cache, **model_specific_kwargs
        )
        # print("---",input_ids,tokenizer.batch_decode(input_ids))
        model_inputs["attention_mask"] = attention_mask
        model_inputs["position_ids"] = position_ids[:, -1].unsqueeze(-1) if past else position_ids

        outputs = self(**model_inputs)  # (batch_size * num_beams, cur_len, vocab_size)
        if cur_len == init_length:
            next_token_logits = outputs[0].gather(1, start_idx).squeeze(1)
        else:
            next_token_logits = outputs[0][:, -1, :]  # (batch_size * num_beams, vocab_size)

        # if model has past, then set the past variable to speed up decoding
        if _use_cache(self,outputs, use_cache):
            past = outputs[1]

        # # repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858)
        # if repetition_penalty != 1.0:
        #     self.enforce_repetition_penalty_(
        #         next_token_logits, batch_size, num_beams, input_ids, repetition_penalty,
        #     )

        if temperature != 1.0:
            next_token_logits = next_token_logits / temperature

        if self.config.is_encoder_decoder and do_sample is False:
            # TODO (PVP) still a bit hacky here - there might be a better solution
            next_token_logits = self.adjust_logits_during_generation(
                next_token_logits, cur_len=cur_len, max_length=max_length
            )

        scores = F.log_softmax(next_token_logits, dim=-1)  # (batch_size * num_beams, vocab_size)

        scores = postprocess_next_token_scores(
            scores=scores,
            input_ids=input_ids,
            no_repeat_ngram_size=no_repeat_ngram_size,
            bad_words_ids=bad_words_ids,
            cur_len=cur_len,
            min_length=min_length,
            max_length=max_length,
            eos_token_id=eos_token_id,
            repetition_penalty=repetition_penalty,
            batch_size=batch_size,
            num_beams=num_beams,
        )

        assert scores.shape == (batch_size * num_beams, vocab_size), "Shapes of scores: {} != {}".format(
            scores.shape, (batch_size * num_beams, vocab_size)
        )

        avoid_idx = []
        for i, c in enumerate(constraints):
            if c is not None:
                avoid_idx.extend([[i, x] for x in c.avoid()])
                if cur_len - init_length < min_length:
                    avoid_idx.extend([[i, x] for x in c.eos()])
        # print(input_ids.shape, scores.shape)
        if avoid_idx:
            banned_mask = torch.LongTensor(avoid_idx)
            indices = torch.ones(len(banned_mask))

            # banned_mask = torch.sparse.LongTensor(banned_mask.t(), indices, scores.size()).to(
            #     scores.device).to_dense().bool()

            banned_mask = torch.sparse.LongTensor(banned_mask.t(), indices, scores.size()).to_dense().to(
                scores.device).bool()
            
            scores.masked_fill_(banned_mask, -float("inf"))
        if do_sample:
            raise NotImplementedError
        else:
            next_scores = scores + beam_scores[:, None].expand_as(scores)  # (batch_size * num_beams, vocab_size)
            next_scores2 = next_scores
            cur_scores = scores.view(batch_size, num_beams,vocab_size)
            # re-organize to group the beam together (we are keeping top hypothesis accross beams)
            full_scores = next_scores.view(
                batch_size, num_beams * vocab_size
            )  # (batch_size, num_beams * vocab_size)

            next_scores, next_tokens = torch.topk(full_scores, 2 * num_beams, dim=1, largest=True, sorted=True)

            # prepare look ahead input
            temp_attention_mask, temp_position_ids = None, None
            if look_ahead:
                assert _use_cache(self,outputs, use_cache) and use_cache, 'model not using past'
                temp_attention_mask = attention_mask if self.config.is_encoder_decoder else \
                    torch.cat([attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1)
                temp_position_ids = torch.cat([position_ids, (position_ids[:, -1] + 1).unsqueeze(-1)], dim=1)
            
            t1 = time.time()
            pick_scores, pick_tokens, constraints, num_mets = topk_huggingface(timestep=cur_len - init_length,
                                                                               batch_size=batch_size,
                                                                               beam_size=num_beams,
                                                                               vocab_size=vocab_size,
                                                                               pad_token_id=pad_token_id,
                                                                               diversity=diversity,
                                                                               constraint_limit=constraint_limit,
                                                                               sat_tolerance=sat_tolerance,
                                                                               inactive=np.zeros((batch_size, num_beams)),
                                                                               scores=full_scores,
                                                                               hypotheses=constraints,
                                                                               num_fill=2 * num_beams,
                                                                               do_beam_sample=do_beam_sample,
                                                                               beam_temperature=beam_temperature,
                                                                               look_ahead=look_ahead,
                                                                               max_length=max_length,
                                                                               model=self,
                                                                               temp_input_ids=input_ids,
                                                                               temp_attention_mask=temp_attention_mask,
                                                                               temp_position_ids=temp_position_ids,
                                                                               temp_past=past,
                                                                               score_history=score_history.cpu().numpy() if score_history is not None else None,                                                                             stochastic=stochastic,sampler=sampler,cur_scores = cur_scores,
                                                                               model_specific_kwargs=model_specific_kwargs)
            t2 = time.time()
            # print('topk time: ' + str(t2-t1))

            next_scores = torch.tensor(pick_scores, dtype=next_scores.dtype, device=next_scores.device)
            next_tokens = torch.tensor(pick_tokens, dtype=next_tokens.dtype, device=next_tokens.device)

        assert next_scores.size() == next_tokens.size() == (batch_size, 2 * num_beams)

        # next batch beam content
        next_batch_beam = []

        # for each sentence
        t1 = time.time()
        for batch_idx in range(batch_size):

            # if we are done with this sentence, add a pad token
            if done[batch_idx] and (not return_all):
                assert (
                    len(generated_hyps[batch_idx]) >= num_beams
                ), "Batch can only be done if at least {} beams have been generated".format(num_beams)
                assert (
                    eos_token_id is not None and pad_token_id is not None
                ), "generated beams >= num_beams -> eos_token_id and pad_token have to be defined"
                next_batch_beam.extend([(0, pad_token_id, 0, None, -1,0)] * num_beams)  # pad the batch
                continue

            # next sentence beam content
            next_sent_beam = []

            # next tokens for this sentence
            for beam_token_rank, (beam_token_id, beam_token_score, constraint, num_met) in enumerate(
                zip(next_tokens[batch_idx], next_scores[batch_idx], constraints[batch_idx], num_mets[batch_idx])
            ):
                # get beam and token IDs
                beam_id = beam_token_id // vocab_size
                token_id = beam_token_id % vocab_size

                effective_beam_id = batch_idx * num_beams + beam_id
                sentence_end = token_id.item() in constraint.eos()
                # add to generated hypotheses if end of sentence or last iteration
                if ((eos_token_id is not None) and (token_id.item() == eos_token_id)) or sentence_end:
                    # if beam_token does not belong to top num_beams tokens, it should not be added
                    is_beam_token_worse_than_top_num_beams = beam_token_rank >= num_beams
                    # if is_beam_token_worse_than_top_num_beams:
                    #     continue
                    generated_hyps[batch_idx].add(
                        torch.cat((input_ids[effective_beam_id], token_id.view([1]))), beam_token_score.item(), num_met,
                    )
                else:
                    second_score = beam_token_score
                    if stochastic:
                        second_score = next_scores2[effective_beam_id, token_id]
                        # print(second_score, effective_beam_id, token_id, next_scores2.shape,next_scores2[0,:10])
                    # add next predicted token since it is not eos_token
                    next_sent_beam.append((beam_token_score, token_id, effective_beam_id, constraint, num_met,second_score))

                # once the beam for next step is full, don't add more tokens to it.
                if len(next_sent_beam) == num_beams:
                    break

            # Check if were done so that we can save a pad step if all(done)
            done[batch_idx] = done[batch_idx] or generated_hyps[batch_idx].is_done(
                next_scores[batch_idx][:beam_token_rank + 1].max().item(), cur_len=cur_len
            ) or not next_sent_beam

            if len(next_sent_beam) < num_beams:
                if next_sent_beam:
                    pad_candidate = next_sent_beam[-1]
                elif done[batch_idx]:
                    pad_candidate = (0, pad_token_id, 0, None, -1, 0)
                else:
                    raise ValueError('impossible search state')
                next_sent_beam += [pad_candidate] * (num_beams - len(next_sent_beam))

            # update next beam content
            assert len(next_sent_beam) == num_beams, "Beam should always be full"
            next_batch_beam.extend(next_sent_beam)
            assert len(next_batch_beam) == num_beams * (batch_idx + 1), "We should have added num_beams each step"
        t2 = time.time()
        # print('For each sentence loop time: ' + str(t2-t1))
        # stop when we are done with each sentence
        if all(done) and (not return_all):
            break

        # sanity check / prepare next batch
        assert len(next_batch_beam) == batch_size * num_beams
        beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
        if stochastic:
            # update phi_S, G_phi_S
            # print(f"---{cur_len}",tokenizer.batch_decode(input_ids))
            # print([len(x) for x in next_batch_beam])
            sampler.beam_state["phi_S"] = beam_scores.new([x[5] for x in next_batch_beam])
            sampler.beam_state["G_phi_S"] = beam_scores
            sampler.state["phi_S"] = full_scores.reshape(batch_size, num_beams, vocab_size)
        beam_tokens = input_ids.new([x[1] for x in next_batch_beam])
        beam_idx = input_ids.new([x[2] for x in next_batch_beam])
        constraints = [x[3] for x in next_batch_beam]
        num_mets = [x[4] for x in next_batch_beam]

        # re-order batch and update current length
        input_ids = input_ids[beam_idx, :]
        input_ids = torch.cat([input_ids, beam_tokens.unsqueeze(1)], dim=-1)
        position_ids = position_ids[beam_idx, :]
        position_ids = torch.cat([position_ids, (position_ids[:, -1] + 1).unsqueeze(-1)], dim=1)
        cur_len = cur_len + 1

        if look_ahead:
            token_probs = torch.reshape(scores, [batch_size * num_beams, -1])[beam_idx, :]
            token_score = torch.gather(token_probs, -1, beam_tokens[..., None].expand(token_probs.shape))[:, 0]

            if score_history is None:
                score_history = token_score[..., None]
            else:
                previous_history = score_history[beam_idx, :]
                score_history = torch.cat([previous_history, token_score[..., None]], dim=-1)
                score_history *= (beam_scores != 0).float()[:, None]

        # re-order internal states
        if past is not None:
            past = self._reorder_cache(past, beam_idx)

        # extend attention_mask for new generated input if only decoder
        if self.config.is_encoder_decoder is False:
            attention_mask = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
            )
        t4 = time.time()
        # print('Total token generation time: ' + str(t4-t3))

    # finalize all open beam hypotheses and add to generated hypotheses
    for batch_idx in range(batch_size):
        if done[batch_idx]:
            continue

        # test that beam scores match previously calculated scores if not eos and batch_idx not done
        if eos_token_id is not None and all(
            (token_id % vocab_size).item() not in cons_eos for token_id in next_tokens[batch_idx]
        ):
            assert torch.all(
                next_scores[batch_idx, :num_beams] == beam_scores.view(batch_size, num_beams)[batch_idx]
            ), "If batch_idx is not done, final next scores: {} have to equal to accumulated beam_scores: {}".format(
                next_scores[:, :num_beams][batch_idx], beam_scores.view(batch_size, num_beams)[batch_idx],
            )

        # need to add best num_beams hypotheses to generated hyps
        for beam_id in range(num_beams):
            effective_beam_id = batch_idx * num_beams + beam_id
            final_score = beam_scores[effective_beam_id].item()
            final_tokens = input_ids[effective_beam_id]
            final_num_met = num_mets[effective_beam_id]
            generated_hyps[batch_idx].add(final_tokens, final_score, final_num_met)

    # depending on whether greedy generation is wanted or not define different output_batch_size and output_num_return_sequences_per_batch
    output_batch_size = batch_size if do_sample else batch_size * num_sequences_per_return
    output_num_return_sequences_per_batch = num_sequences_per_return if num_sequences_per_return is not None else 1

    # select the best hypotheses
    if return_all:
        sent_lengths = []
    else:
        sent_lengths = input_ids.new(output_batch_size)
    all_best = []
    all_lp_record = []
    all_batch_hyps = []
    # retrieve best hypotheses
    for i, hypotheses in enumerate(generated_hyps):
        sorted_hyps = sorted(hypotheses.beams, key=lambda x: (x[2], x[0]), reverse=True)
        all_hyps = hypotheses.beams + hypotheses.deleted_beams
        all_batch_hyps.append(all_hyps)
        lp_record = []
        best = []
        # print([(tokenizer.decode(x[1].tolist()), x[2]) for x in hypotheses.beams])
        if return_all:
            lps = length_penalties
            cand_each_lp = lp_return_size
            for lp in lps:
                all_hyps = [(x[3] / len(x[1]) ** lp,x[1],x[2],x[3]) for x in all_hyps]
                sorted_all_hyps = sorted(all_hyps, key=lambda x: (x[2], x[0]), reverse=True)
                for j in range(cand_each_lp):
                    best_hyp = sorted_all_hyps[j][1]
                    sent_lengths.append(len(best_hyp))
                    best.append(best_hyp)
                    lp_record.append(lp)
                all_hyps = sorted_all_hyps[cand_each_lp:]
        else:
            for j in range(output_num_return_sequences_per_batch):
                effective_batch_idx = output_num_return_sequences_per_batch * i + j
                best_hyp = sorted_hyps[j][1]
                # print(tokenizer.decode(best_hyp),len(best_hyp))
                sent_lengths[effective_batch_idx] = len(best_hyp)
                best.append(best_hyp)
                lp_record.append(length_penalty)
        all_best.append(best)
        all_lp_record.append(lp_record)
    # shorter batches are padded
    # print(f"output_num_return_sequences_per_batch: {output_num_return_sequences_per_batch}")
    # print(sent_lengths,min(sent_lengths),max(sent_lengths))
    batch_decoded = []
    for i, best in enumerate(all_best):
        if min(sent_lengths) != max(sent_lengths):
            assert pad_token_id is not None, "`Pad_token_id` has to be defined"
            sent_max_len = min(max(sent_lengths) + 1, max_length)
            decoded = input_ids.new(len(best), sent_max_len).fill_(pad_token_id)
            # fill with hypothesis and eos_token_id if necessary
            for idx, hypo in enumerate(best):
                j = output_num_return_sequences_per_batch * i + idx
                decoded[idx, : sent_lengths[j]] = hypo
                if sent_lengths[j] < max_length:
                    decoded[idx, sent_lengths[j]] = eos_token_id
        else:
            # none of the hypotheses have an eos_token
            assert (len(hypo) == max_length for hypo in best)
            decoded = torch.stack(best).type(torch.long).to(next(self.parameters()).device)
        batch_decoded.append(decoded)
    outputs = {"decoded":batch_decoded, "lps":all_lp_record,"all_hyps":all_batch_hyps}
    return outputs


def _reorder_cache(past: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor) -> Tuple[Tuple[torch.Tensor]]:
    """
    This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
    [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
    beam_idx at every generation step.
    """
    return tuple(
        tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
        for layer_past in past
    )



def _use_cache(self, outputs, use_cache):
    """During generation, decide whether to pass the `past` variable to the next forward pass."""
    if len(outputs) <= 1 or use_cache is False:
        return False
    if hasattr(self.config, "mem_len") and self.config.mem_len == 0:
        return False
    return True


def calc_banned_ngram_tokens(prev_input_ids: Tensor, num_hypos: int, no_repeat_ngram_size: int, cur_len: int) -> None:
    """Copied from fairseq for no_repeat_ngram in beam_search"""
    if cur_len + 1 < no_repeat_ngram_size:
        # return no banned tokens if we haven't generated no_repeat_ngram_size tokens yet
        return [[] for _ in range(num_hypos)]
    generated_ngrams = [{} for _ in range(num_hypos)]
    for idx in range(num_hypos):
        gen_tokens = prev_input_ids[idx].tolist()
        generated_ngram = generated_ngrams[idx]
        for ngram in zip(*[gen_tokens[i:] for i in range(no_repeat_ngram_size)]):
            prev_ngram_tuple = tuple(ngram[:-1])
            generated_ngram[prev_ngram_tuple] = generated_ngram.get(prev_ngram_tuple, []) + [ngram[-1]]

    def _get_generated_ngrams(hypo_idx):
        # Before decoding the next token, prevent decoding of ngrams that have already appeared
        start_idx = cur_len + 1 - no_repeat_ngram_size
        ngram_idx = tuple(prev_input_ids[hypo_idx, start_idx:cur_len].tolist())
        return generated_ngrams[hypo_idx].get(ngram_idx, [])

    banned_tokens = [_get_generated_ngrams(hypo_idx) for hypo_idx in range(num_hypos)]
    return banned_tokens


def calc_banned_bad_words_ids(prev_input_ids: Iterable[int], bad_words_ids: Iterable[int]) -> Iterable[int]:
    banned_tokens = []

    def _tokens_match(prev_tokens, tokens):
        if len(tokens) == 0:
            # if bad word tokens is just one token always ban it
            return True
        if len(tokens) > len(prev_input_ids):
            # if bad word tokens are longer then prev input_ids they can't be equal
            return False

        if prev_tokens[-len(tokens) :] == tokens:
            # if tokens match
            return True
        else:
            return False

    for prev_input_ids_slice in prev_input_ids:
        banned_tokens_slice = []

        for banned_token_seq in bad_words_ids:
            assert len(banned_token_seq) > 0, "Banned words token sequences {} cannot have an empty list".format(
                bad_words_ids
            )

            if _tokens_match(prev_input_ids_slice.tolist(), banned_token_seq[:-1]) is False:
                # if tokens do not match continue
                continue

            banned_tokens_slice.append(banned_token_seq[-1])

        banned_tokens.append(banned_tokens_slice)

    return banned_tokens
def _use_cache(model, outputs, use_cache):
    """During generation, decide whether to pass the `past` variable to the next forward pass."""
    if len(outputs) <= 1 or use_cache is False:
        return False
    if hasattr(model.config, "mem_len") and model.config.mem_len == 0:
        return False
    return True

def postprocess_next_token_scores(
    scores,
    input_ids,
    no_repeat_ngram_size,
    bad_words_ids,
    cur_len,
    min_length,
    max_length,
    eos_token_id,
    repetition_penalty,
    batch_size,
    num_beams,
):
    # repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858)
    if repetition_penalty != 1.0:
        enforce_repetition_penalty_(
            scores, batch_size, num_beams, input_ids, repetition_penalty,
        )

    # set eos token prob to zero if min_length is not reached
    if eos_token_id is not None and cur_len < min_length:
        scores[:, eos_token_id] = -float("inf")

    if no_repeat_ngram_size > 0:
        # calculate a list of banned tokens to prevent repetitively generating the same ngrams
        num_batch_hypotheses = batch_size * num_beams
        # from fairseq: https://github.com/pytorch/fairseq/blob/a07cb6f40480928c9e0548b737aadd36ee66ac76/fairseq/sequence_generator.py#L345
        banned_batch_tokens = calc_banned_ngram_tokens(
            input_ids, num_batch_hypotheses, no_repeat_ngram_size, cur_len
        )
        for i, banned_tokens in enumerate(banned_batch_tokens):
            scores[i, banned_tokens] = -float("inf")

    if bad_words_ids is not None:
        # calculate a list of banned tokens according to bad words
        banned_tokens = calc_banned_bad_words_ids(input_ids, bad_words_ids)

        for i, banned_tokens in enumerate(banned_tokens):
            scores[i, banned_tokens] = -float("inf")

    return scores
