import math
import random
import numpy as np
import torch
from torch.nn import functional as F
from scipy.stats import rankdata
from operator import attrgetter
from typing import Dict, List, Optional, Tuple, Set, Union
import transformers

from lexical_constraints import ConstrainedHypothesis, ConstrainedCandidate

# from transformers import AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained('gpt2')


def topk_huggingface(timestep: int,
                     batch_size: int,
                     beam_size: int,
                     vocab_size: int,
                     pad_token_id: int,
                     diversity: bool,
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
        idxs = torch.arange(sentno * beam_size, sentno * beam_size + beam_size, device=temp_past[0].device)
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
                                                                                      num_fill=num_fill)
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


def _sequential_topk(timestep: int,
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
                     num_fill: int = None,) -> Tuple[np.array, np.array, np.array,
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
        nextones = hyp.positive_state.allowed() if hyp.positive_state is not None else set()

        # (3) add the single-best item after this (if it's valid)
        best_k = np.argsort(scores[row])[::-1][:beam_size]
        for col in best_k:
            if hyp.is_valid(col):
                nextones.add(col)

        # Now, create new candidates for each of these items
        for col in nextones:
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
        sorted_candidates = sorted(candidates, key=attrgetter('score'), reverse=True)

        max_satisfy = max([x.hypothesis.num_met() for x in sorted_candidates])
        sorted_candidates = [x for x in sorted_candidates if x.hypothesis.num_met() >= max_satisfy - sat_tolerance]

        # Bucket candidates in each group by met order
        all_orders = set([x.hypothesis.met_process() for x in sorted_candidates])

        grouped_order_candidates = []
        grouped_candidates = [[x for x in sorted_candidates if x.hypothesis.met_process() == o] for o in all_orders]
        temperature = beam_temperature if beam_temperature is not None else 1.0

        if do_beam_sample:
            for group in grouped_candidates:
                chosen_ones = []
                raw_probs = [math.exp(x.score / temperature) + 10 ** (-10) for x in group]
                while group:
                    chosen_index, chosen_one = random.choices(list(enumerate(group)), weights=raw_probs, k=1)[0]
                    group.pop(chosen_index)
                    raw_probs.pop(chosen_index)
                    chosen_ones.append(chosen_one)
                grouped_order_candidates.append(_reorder_group_POS(chosen_ones, 'score') if diversity else chosen_ones)
        else:
            for group in grouped_candidates:
                grouped_order_candidates.append(_reorder_group_POS(group, 'score') if diversity else group)

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
        nextones = hyp.positive_state.allowed() if hyp.positive_state is not None else set()

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
                    past = _reorder_cache(past, torch.arange(num_candidates, device=past[0].device))
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
                    past = tuple(torch.cat([past[i], ahead_past[i]], dim=1) for i in range(len(past)))
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


def _reorder_cache(past: Tuple, beam_idx: torch.Tensor) -> Tuple[torch.Tensor]:
    return tuple(layer_past.index_select(1, beam_idx) for layer_past in past)


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
