# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not
# use this file except in compliance with the License. A copy of the License
# is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

import copy
import logging
from operator import attrgetter
from typing import Dict, List, Optional, Tuple, Set, Union
import collections
import os
import utils

import numpy as np
import torch
from process import read_common_pos
from collections import defaultdict
# from flair.data import Sentence
# from flair.models import SequenceTagger
logger = logging.getLogger(__name__)

os.environ["CURL_CA_BUNDLE"]=""

Phrase = List[int]
Literal = Tuple[Phrase, bool]
# Represents a list of raw constraints for a sentence. Each constraint is a list of target-word IDs.
RawConstraintList = List[Phrase]
ClauseConstraintList = List[List[Literal]]

from transformers import AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained('gpt2-xl')
tokenizer = utils.load_tokenizer('gpt2-xl',"../models/gpt2-xl/")

digit_num = 10 ** (len(str(tokenizer.vocab_size)) - 1)
POS_start = (tokenizer.vocab_size // digit_num + 1) * digit_num

mapper = read_common_pos(os.environ['COMMON_POS_FILENAME'], POS_start)
# tagger = SequenceTagger.load("flair/pos-english")
# tagger = utils.load_tagger("flair/pos-english","../models/flair_pos_english/")


class Trie:
    """
    Represents a set of phrasal constraints for an input sentence.
    These are organized into a trie.
    """
    def __init__(self,
                 raw_phrases: Optional[RawConstraintList] = None,
                 parent_arc: int = None,
                 parent_trie: 'Trie' = None) -> None:
        self.final_ids = set()  # type: Set[int]
        self.children = {}  # type: Dict[int,'Trie']
        self.parent_arc = parent_arc
        self.parent_trie = parent_trie

        if raw_phrases:
            for phrase in raw_phrases:
                self.add_phrase(phrase)

    def add_phrase(self,
                   phrase: List[int]) -> None:
        """
        Recursively adds a phrase to this trie node.

        :param phrase: A list of word IDs to add to this trie node.
        """
        if len(phrase) == 1:
            self.final_ids.add(phrase[0])
        else:
            next_word = phrase[0]
            if next_word not in self.children:
                self.children[next_word] = Trie(parent_arc=next_word, parent_trie=self)
            self.step(next_word).add_phrase(phrase[1:])

    def delete_phrase(self,
                      phrase: List[int]) -> None:
        """
        Recursively deletes a phrase to this trie node.

        :param phrase: A list of word IDs to delete in this trie node.
        """
        if len(phrase) == 1:
            assert phrase[0] in self.final_ids, f"Trie {str(self)} \nDo not contain {phrase}"
            self.final_ids.remove(phrase[0])
        else:
            next_word = phrase[0]
            assert next_word in self.children.keys(), f"Trie {str(self)} \nDo not contain {phrase}"
            self.step(next_word).delete_phrase(phrase[1:])

        # Move the arc to an empty node to final_ids of its parent
        for arc in list(self.children):
            if len(self.children[arc]) == 0:
                self.children.pop(arc)

    def check_phrase(self,
                     phrase: List[int]) -> bool:
        """
        Check whether a phrase is in this trie.

        :param phrase: A list of word IDs to check existence.
        """
        if len(phrase) == 1:
            return phrase[0] in self.final_ids
        else:
            next_word = phrase[0]
            if next_word in self.children:
                return self.step(next_word).check_phrase(phrase[1:])
            return False

    def trace_phrase(self,
                     word_id: int) -> List[int]:
        """
        Recursively backward to get word ids in a phrase.

        :param word_id: The last word IDs in phrase.
        """
        assert word_id in self.final_ids, f"{word_id} does not in trie node {self.final_ids}"
        phrase = self.trace_arcs()
        phrase.append(word_id)
        return phrase

    def trace_arcs(self,) -> List[int]:
        """
        Recursively backward to get arc to ancestor
        """
        arcs = []
        parent_trie, parent_arc = self.parent_trie, self.parent_arc
        while parent_trie is not None:
            arcs.append(parent_arc)
            parent_arc = parent_trie.parent_arc
            parent_trie = parent_trie.parent_trie
        arcs.reverse()
        return arcs

    def __str__(self) -> str:
        s = f'({list(self.final_ids)}'
        for child_id in self.children.keys():
            s += f' -> {child_id} {self.children[child_id]}'
        s += ')'
        return s

    def __len__(self) -> int:
        """
        Returns the number of phrases represented in the trie.
        """
        phrase_count = len(self.final_ids)
        for child in self.children.values():
            phrase_count += len(child)
        return phrase_count

    def step(self, word_id: int) -> Optional['Trie']:
        """
        Returns the child node along the requested arc.

        :param word_id: requested arc.
        :return: The child node along the requested arc, or None if no such arc exists.
        """
        return self.children.get(word_id, None)

    def descend(self,
              arcs: List[int]) -> Optional['Trie']:
        pointer = self
        for arc in arcs:
            if pointer is None:
                break
            pointer = pointer.step(word_id=arc)
        return pointer

    def final(self) -> Set[int]:
        """
        Returns the set of final ids at this node.

        :return: The set of word IDs that end a constraint at this state.
        """
        return self.final_ids


class NegativeState:
    """
    Represents the state of a hypothesis in the AvoidTrie.
    The offset is used to return actual positions in the one-dimensionally-resized array that
    get set to infinity.

    :param avoid_trie: The trie containing the phrases to avoid.
    :param state: The current state (defaults to root).
    """
    def __init__(self,
                 avoid_trie: Trie,
                 state: List[Trie] = None) -> None:

        self.root = avoid_trie
        self.state = state if state else [self.root]

    def __str__(self):
        s = f'Root: {self.root}\nState: ['
        for state in self.state:
            s += f'{state}, '
        return s

    def consume(self, word_id: int) -> 'NegativeState':
        """
        Consumes a word, and updates the state based on it. Returns new objects on a state change.

        The next state for a word can be tricky. Here are the cases:
        (1) If the word is found in our set of outgoing child arcs, we take that transition.
        (2) If the word is not found, and we are not in the root state, we need to reset.
            This means we pretend we were in the root state, and see if we can take a step
        (3) Otherwise, if we are not already in the root state (i.e., we were partially through
            the trie), we need to create a new object whose state is the root state
        (4) Finally, if we couldn't advance and were already in the root state, we can reuse
            this object.

        :param word_id: The word that was just generated.
        """
        new_state = []
        for state in set(self.state + [self.root]):
            if word_id in state.children:
                new_state.append(state.step(word_id))

        if new_state:
            return NegativeState(self.root, new_state)
        else:
            if len(self.state) == 1 and self.root == self.state[0]:
                return self
            else:
                return NegativeState(self.root, [self.root])

    def avoid(self) -> Set[int]:
        """
        Returns a set of word IDs that should be avoided. This includes the set of final states from the
        root node, which are single tokens that must never be generated.

        :return: A set of integers representing words that must not be generated next by this hypothesis.
        """
        return self.root.final().union(*[state.final() for state in self.state])

    def __str__(self) -> str:
        return str(self.state)


class NegativeBatch:
    """
    Represents a set of phrasal constraints for all items in the batch.
    For each hypotheses, there is an AvoidTrie tracking its state.

    :param beam_size: The beam size.
    :param avoid_list: The list of lists (raw phrasal constraints as IDs, one for each item in the batch).
    """
    def __init__(self,
                 beam_size: int,
                 avoid_list: Optional[List[RawConstraintList]] = None) -> None:

        self.avoid_states = []  # type: List[NegativeState]

        # Store the sentence-level tries for each item in their portions of the beam
        if avoid_list is not None:
            for literal_phrases in avoid_list:
                self.avoid_states += [NegativeState(Trie(literal_phrases))] * beam_size

    def reorder(self, indices: torch.Tensor) -> None:
        """
        Reorders the avoid list according to the selected row indices.
        This can produce duplicates, but this is fixed if state changes occur in consume().

        :param indices: An mx.nd.NDArray containing indices of hypotheses to select.
        """
        if self.avoid_states:
            self.avoid_states = [self.avoid_states[x] for x in indices.numpy()]

    def consume(self, word_ids: torch.Tensor) -> None:
        """
        Consumes a word for each trie, updating respective states.

        :param word_ids: The set of word IDs.
        """
        word_ids = word_ids.numpy().tolist()
        for i, word_id in enumerate(word_ids):
            if self.avoid_states:
                self.avoid_states[i] = self.avoid_states[i].consume(word_id)

    def avoid(self) -> Tuple[Tuple[int], Tuple[int]]:
        """
        Assembles a list of per-hypothesis words to avoid. The indices are (x, y) pairs into the scores
        array, which has dimensions (beam_size, target_vocab_size). These values are then used by the caller
        to set these items to np.inf so they won't be selected. Words to be avoided are selected by
        consulting both the global trie of phrases and the sentence-specific one.

        :return: Two lists of indices: the x coordinates and y coordinates.
        """
        to_avoid = set()  # type: Set[Tuple[int, int]]
        for i, state in enumerate(self.avoid_states):
            for word_id in state.avoid():
                to_avoid.add((i, word_id))

        return tuple(zip(*to_avoid))  # type: ignore


class PositiveState:
    """
    Represents a set of words and phrases that must appear in the output.
    The offset is used to return actual positions in the one-dimensionally-resized array that
    get set to infinity.

    :param positive_trie: The trie containing the phrases to appear.
    :param state: The current state (defaults to root).
    """
    def __init__(self,
                 positive_trie: Trie,
                 state: List[Trie] = None,
                 met_phrases: RawConstraintList = None,cons_type="default") -> None:

        self.root = positive_trie
        self.state = state if state else [self.root]
        self.met_phrases = met_phrases
        self.select_positive_phrases = True
        self.cons_type = cons_type

    def __str__(self):
        s = f'Root: {self.root}\nState: ['
        for state in self.state:
            s += f'{state}, '
        s += f']\nMet_phrases: {self.met_phrases}'
        return s

    def allowed(self) -> Set[int]:
        """
        Returns the set of constrained words that could follow this one.
        For unfinished phrasal constraints, it is the next word in the phrase.
        In other cases, it is the list of all unmet constraints.
        If all constraints are met, an empty set is returned.

        :return: The ID of the next required word, or -1 if any word can follow
        """
        allow = self.root.final().union(*[state.final() for state in self.state])
        allow |= set(self.root.children.keys()).union(*[set(state.children.keys()) for state in self.state])
        legal_words = set(x for x in allow if x < tokenizer.vocab_size)
        met_phrases_check = []
        if self.met_phrases:
            for x in self.met_phrases:
                met_phrases_check.extend(x)
        return set(x for x in legal_words if x not in met_phrases_check)

    def advance(self, word_id_list: List[int]) -> 'PositiveState':
        """
        Updates the constraints object based on advancing on word_id.
        There is a complication, in that we may have started but not
        yet completed a multi-word constraint.  We need to allow constraints
        to be added as unconstrained words, so if the next word is
        invalid, we must "back out" of the current (incomplete) phrase,
        re-setting all of its words as unmet.

        :param word_id: The word ID to advance on.
        :return: A deep copy of the object, advanced on word_id.
        """
        new_state, met_phrases = [], []
        for word_id in word_id_list:
            for state in set(self.state + [self.root]):
                if word_id in state.children:
                    new_state.append(state.step(word_id))
                if word_id in state.final_ids:
                    met_phrases.append(state.trace_phrase(word_id))

        if new_state:
            return PositiveState(self.root, new_state, met_phrases if met_phrases else None)
        else:
            if len(self.state) == 1 and self.root == self.state[0] and not met_phrases:
                return self
            else:
                return PositiveState(self.root, [self.root], met_phrases if met_phrases else None)


class Clause:
    """
    Object used to hold clause.

    :param idx: The id of this clause.
    :param positive: The positive constraints in this clause.
    :param negative: The soft negative constraints in this clause.
    :param satisfy: whether this clause is satisfied
    """

    __slots__ = ('idx', 'positive', 'negative', 'count', 'satisfy', 'min_count', 'max_count', 'look_ahead', 'next_look_ahead')

    def __init__(self,
                 idx: int,
                 positive: List[Phrase],
                 negative: List[Phrase],
                 min_count: int,
                 max_count: int,
                 count: int,
                 satisfy: float,
                 look_ahead: float) -> None:
        self.idx = idx
        self.positive = positive
        self.negative = negative
        self.min_count = min_count
        self.max_count = max_count
        self.count = count
        self.satisfy = satisfy
        self.look_ahead = look_ahead

        next_look_ahead = set()
        for literal in self.positive:
            if len(literal) > 1:
                next_look_ahead.add(literal[0])
        self.next_look_ahead = list(next_look_ahead)


    def __str__(self):
        return f'clause(id={self.idx}, positive={self.positive}, negative={self.negative}, satisfy={self.satisfy},' \
               f'min_count={self.min_count}, max_count={self.max_count}, look_ahead={self.look_ahead})'


def is_prefix(pref: List[int],
              phrase: List[int]):
    if not pref:
        return False
    return pref == phrase[:len(pref)]
def get_pos(text):
    sentence = Sentence(f"It is {text} than")
    tagger.predict(sentence)
    return sentence.get_labels('pos')[2].value

class ConstrainedHypothesis:
    """
    Keep track of positive and negative constraint

    hard negative constraint will not be generated in any cases
    soft negative constraint might be generated in some case due to OR gate in clause
    positive constraints will be encourage to appear

    :param constraint_list: A list of clause constraints (each represented as a list of literals).
    """
    def __init__(self,
                 constraint_list: ClauseConstraintList,
                 eos_id: Union[int, list],ordered=False
                 ) -> None:
        self.eos_id = eos_id if isinstance(eos_id, list) else [eos_id]
        self.clauses = []  # type: List[Clause]
        self.cache = []
        self.ordered = ordered

        self.num_positive = 0
        self.positive_pool, self.neutral_pool, self.negative_pool = [], [], []
        if self.ordered:
            self.ordered_pos_clauses, self.ordered_pos_pool= [],defaultdict(lambda : defaultdict(list))
            self.cur_ind = 0
        for idx, clause in enumerate(constraint_list):
            if not clause:
                continue
            phrases, polar, min_count, max_count, con_type, look_ahead, order_index = clause
            # positive clause
            if polar:
                # need to encourage and track
                if min_count > 0:
                    self.num_positive += 1
                    self.positive_pool.extend(phrases)
                    self.neutral_pool.extend(phrases)
                    self.clauses.append(Clause(idx=idx, positive=phrases, negative=[], min_count=min_count,
                                               max_count=max_count, count=0, satisfy=False, look_ahead=look_ahead))
                    if self.ordered:
                        self.ordered_pos_pool[order_index]["phrases"].extend(phrases)
                        self.ordered_pos_pool[order_index]["clauses_idx"].append(idx)
                        self.ordered_pos_pool[order_index]["type"] = con_type
                        # self.ordered_pos_pool.append(phrases)
                        # self.ordered_pos_clauses.append(order_index)
                else:
                    # need to track only
                    self.neutral_pool.extend(phrases)
                    self.clauses.append(Clause(idx=idx, positive=phrases, negative=[], min_count=min_count,
                                               max_count=max_count, count=0, satisfy=True, look_ahead=False))
            else:
                self.negative_pool.extend(phrases)
                self.clauses.append(Clause(idx=idx, positive=[], negative=phrases, min_count=min_count,
                                           max_count=max_count, count=0, satisfy=True, look_ahead=False))
        # print("self.positive_pool", self.positive_pool)
        self.positive_state = PositiveState(Trie(self.positive_pool)) if self.positive_pool else None
        self.neutral_state = PositiveState(Trie(self.neutral_pool)) if self.neutral_pool else None
        self.hard_negative_state = NegativeState(Trie(self.negative_pool)) if self.negative_pool else None
        if ordered:
            self.max_order = max(self.ordered_pos_pool.keys())+1
            for i in range(self.max_order):
                clause_pool = self.ordered_pos_pool[i]["phrases"]
                cons_type = self.ordered_pos_pool[i]["type"]
                if cons_type == "Comparative" or cons_type == "Comparative_Whole":
                    self.ordered_pos_pool[i]["pos_state"] = ComparativeState(Trie(clause_pool),cons_type=cons_type)
                else:
                    self.ordered_pos_pool[i]["pos_state"] = PositiveState(Trie(clause_pool),cons_type=cons_type)
            self.ordered_pos_state = self.ordered_pos_pool[self.cur_ind]["pos_state"]
            
            future_negs= []
            for i in range(self.cur_ind+1,self.max_order):
                future_negs.extend(self.ordered_pos_pool[i]["phrases"])
            self.hard_negative_state = NegativeState(Trie(self.negative_pool+future_negs)) if self.negative_pool else None
        self.generation = []
        self.orders = []
        self.in_process = []
        self.past = None

    def __len__(self) -> int:
        """
        :return: The number of constraints.
        """
        return len(self.clauses)

    def __str__(self) -> str:
        return '\n'.join([str(c) for c in self.clauses])

    def pos_allowed(self) -> Set[int]:
        if self.ordered:
            return self.ordered_pos_state.allowed() if self.ordered_pos_state is not None else set()
        else:
            return self.positive_state.allowed() if self.positive_state is not None else set()
    def get_pos_state(self):
        if self.ordered:
            return self.ordered_pos_state
        else:
            return self.positive_state
    def set_pos_tokens(self, scores):
        if (self.ordered) and (self.ordered_pos_state.cons_type == "Comparative_Whole"):
            clause = self.ordered_pos_state.set_pos_tokens(scores)
            new_clauses = []
            prev_clause_ind = self.ordered_pos_pool[self.cur_ind]["clauses_idx"]
            for idx in range(len(self.clauses)):
                if idx not in prev_clause_ind:
                    new_clauses.append(self.clauses[idx])
            self.clauses = new_clauses
            index = len(self.clauses)
            clause.idx = index
            self.clauses.append(clause)
            self.ordered_pos_pool[self.cur_ind]["clauses_idx"] = [index]
        else:
            return
    def size(self) -> int:
        """
        :return: the number of constraints
        """
        return len(self.clauses)

    def num_met(self) -> int:
        """
        :return: the number of constraints that have been met.
        """
        if not self.clauses:
            return 0
        return sum([int(c.satisfy) for c in self.clauses])

    def num_positive_met(self) -> int:
        """
        :return: the number of positive constraints that have been met.
        """
        positive_met = 0
        for c in self.clauses:
            if c.min_count >= 1 and c.satisfy:
                positive_met += 1
        return positive_met

    def met_order(self) -> tuple:
        """
        :return: the number of constraints that have been met.
        """
        return tuple(sorted(self.orders)) #+ tuple(self.in_process)

    def clause_in_process(self) -> tuple:
        """
        :return: the index of clause that's in generation.
        """
        return tuple(self.in_process)

    def met_process(self) -> tuple:
        return tuple(sorted(self.orders + self.in_process))

    def num_needed(self) -> int:
        """
        :return: the number of un-met constraints.
        """
        return self.size() - self.num_met()

    def finished(self) -> bool:
        """
        Return true if all the constraints have been met.

        :return: True if all the constraints are met.
        """
        return self.num_needed() == 0

    def is_valid(self, wordid: int) -> bool:
        """
        Ensures </s> is only generated when the hypothesis is completed.

        :param wordid: The wordid to validate.
        :return: True if all constraints are already met or the word ID is not the EOS id.
        """
        return self.finished() or wordid not in self.eos_id

    def avoid(self) -> Set[int]:
        banned = self.hard_negative_state.avoid() if self.hard_negative_state is not None else set()
        banned = {x for x in banned if x < tokenizer.vocab_size}
        return banned

    def eos(self) -> list:
        """
        :return: Return EOS id.
        """
        return self.eos_id

    def next_look_ahead(self) -> list:
        """

        :return: the first token of each literal that needs looking ahead
        """
        next_look_ahead = []
        for clause in self.clauses:
            if not clause.satisfy:
                next_look_ahead.extend(clause.next_look_ahead)
        return next_look_ahead

    def is_advance(self, word_id) -> bool:
        word_piece = tokenizer.convert_ids_to_tokens([word_id])[0]
        marks = ['.', ',', ':', ';', '?', '!', "'", '-', '"', '{', '}', '(', ')', '[', ']' '|', '&', '*', '/', '~']
        marks += [f'Ġ{x}' for x in marks]
        return word_piece.startswith('Ġ') or (word_piece in marks)

    def get_new_id(self, word_id) -> List[int]:
        whole_word = tokenizer.decode([word_id]).strip()
        if whole_word in mapper:
            return mapper[whole_word]
        return [-1]

    def get_pos_word(self, POS_set, N) -> tuple:
        sentence = tokenizer.decode(self.generation)
        words = [w.strip() for w in sentence.split() if any([tag in POS_set for tag in mapper.get(w.strip().replace(',', '').replace('.', ''), [])])][:N]
        return tuple(sorted(words))

    def advance(self, word_id: int) -> 'ConstrainedHypothesis':
        """
        Updates the constraints object based on advancing on word_id.
        If one of literals in a clause is satisfied, we mark this clause as satisfied

        :param word_id: The word ID to advance on.
        """
        obj = copy.deepcopy(self)

        if obj.hard_negative_state is not None:
            obj.hard_negative_state = obj.hard_negative_state.consume(word_id)
        old_met_phrases = obj.neutral_state.met_phrases if obj.neutral_state.met_phrases else []
        obj.neutral_state = obj.neutral_state.advance([word_id] + obj.get_new_id(word_id))
        if self.ordered:
            # print(f"qooooo, {obj.ordered_pos_state.allowed()}")
            ordered_old_met_phrases = obj.ordered_pos_state.met_phrases if obj.ordered_pos_state.met_phrases else []
            obj.ordered_pos_state = obj.ordered_pos_state.advance([word_id] + obj.get_new_id(word_id))
        newly_met_clause = set()
        positive_update, negative_update = False, False
        # print(f"word_id, {tokenizer.decode(word_id)}")
        if (obj.ordered_pos_pool[obj.cur_ind]["type"] == "Comparative") or ((obj.ordered_pos_pool[obj.cur_ind]["type"] == "Comparative_Whole")):
            clauses_idx = obj.ordered_pos_pool[obj.cur_ind]["clauses_idx"]
            clause = [obj.clauses[x] for x in clauses_idx][0]
            if obj.ordered_pos_state.met_phrases:
                # print("yooooo",[tokenizer.decode(x) for x in obj.ordered_pos_state.met_phrases])
                # print(f"zooooo, {obj.ordered_pos_state.allowed()}")
                if not clause.satisfy:
                    clause.satisfy = True
                    newly_met_clause.add(clause.idx)
                    obj.positive_pool = [x for x in obj.positive_pool if x not in clause.positive]
                    positive_update = True
                    obj.negative_pool = obj.negative_pool + clause.positive
                    obj.ordered_negative_pool = obj.negative_pool
                    negative_update = True
                    obj.cur_ind += 1
                    if obj.cur_ind >= len(obj.ordered_pos_pool):
                        obj.cur_ind -= 1
                        clause_pool = []
                        obj.ordered_pos_state = ComparativeState(Trie(clause_pool),select_positive_phrases=True,cons_type=obj.ordered_pos_state.cons_type,)
                    else:
                        obj.ordered_pos_state = obj.ordered_pos_pool[obj.cur_ind]["pos_state"]
                        future_negs= []
                        for i in range(obj.cur_ind+1,self.max_order):
                            future_negs.extend(self.ordered_pos_pool[i]["phrases"])
                        obj.ordered_negative_pool = obj.negative_pool + future_negs
                        negative_update = True

        if obj.neutral_state.met_phrases is not None and ((obj.ordered_pos_pool[obj.cur_ind]["type"] != "Comparative") and (obj.ordered_pos_pool[obj.cur_ind]["type"] != "Comparative_Whole") ):
            # update the count
            if self.ordered:
                if obj.ordered_pos_state.met_phrases is not None:
                    clauses_idx = obj.ordered_pos_pool[obj.cur_ind]["clauses_idx"]
                    clauses = [obj.clauses[x] for x in clauses_idx]
                    for clause in clauses:
                        # check if current token met phrase
                        # print(f"met_phrases: {obj.ordered_pos_state.met_phrases}")
                        for phrase in obj.ordered_pos_state.met_phrases:
                            # if 1) it is positive. 2) It is newly generated
                            if (phrase in clause.positive) and (phrase not in ordered_old_met_phrases):
                                # 3) if its index in clause.positive and current pos index are equal
                                # this is equal cause we only compare to the correctly indexed clause
                                clause.count += 1

                        previous = clause.satisfy
                        clause.satisfy = clause.count >= clause.min_count
                        if not previous and clause.satisfy:
                            assert clause.idx not in obj.orders, 'clause has already satisfied, impossible state'
                            newly_met_clause.add(clause.idx)
                            obj.positive_pool = [x for x in obj.positive_pool if x not in clause.positive]
                            positive_update = True

                        if clause.count == clause.max_count:
                            obj.negative_pool = obj.negative_pool + clause.positive
                            obj.ordered_negative_pool = obj.negative_pool
                            negative_update = True
                            assert not clause.count > clause.max_count, 'max count violation'
                        if clause.satisfy:
                            not_finished_this_index = False
                            for temp_clause in clauses:
                                if not temp_clause.satisfy:
                                    not_finished_this_index = True
                            if not not_finished_this_index:
                                # finished with all clauses with this order, go to next
                                obj.cur_ind += 1
                                if obj.cur_ind >= len(obj.ordered_pos_pool):
                                    obj.cur_ind -= 1
                                else:
                                    obj.ordered_pos_state = obj.ordered_pos_pool[obj.cur_ind]["pos_state"]
                                    future_negs= []
                                    for i in range(obj.cur_ind+1,self.max_order):
                                        future_negs.extend(self.ordered_pos_pool[i]["phrases"])
                                    obj.ordered_negative_pool = obj.negative_pool + future_negs
                                    negative_update = True
                                break

            else:
                # print(f"met_phrases: {obj.neutral_state.met_phrases}")
                for clause in obj.clauses:
                    for phrase in obj.neutral_state.met_phrases:
                        if (phrase in clause.positive) and (phrase not in old_met_phrases):
                            clause.count += 1

                    previous = clause.satisfy
                    clause.satisfy = clause.count >= clause.min_count
                    if not previous and clause.satisfy:
                        assert clause.idx not in obj.orders, 'clause has already satisfied, impossible state'
                        newly_met_clause.add(clause.idx)
                        obj.positive_pool = [x for x in obj.positive_pool if x not in clause.positive]
                        positive_update = True

                    if clause.count == clause.max_count:
                        obj.negative_pool = obj.negative_pool + clause.positive
                        negative_update = True
                        assert not clause.count > clause.max_count, 'max count violation'

        obj.orders.extend(sorted(newly_met_clause))

        if positive_update:
            # reconstruct positive trie
            new_positive_state = PositiveState(Trie(obj.positive_pool)) if obj.positive_pool else None
            if new_positive_state is not None:
                new_trie_states = set()
                for state in obj.positive_state.state:
                    if state.parent_trie is None:
                        new_trie_states.add(new_positive_state.root)
                    else:
                        trace = state.trace_arcs()
                        new_state = new_positive_state.root.descend(trace)
                        if new_state is not None:
                            new_trie_states.add(new_state)
                obj.positive_state = PositiveState(positive_trie=new_positive_state.root, state=list(new_trie_states))
            else:
                obj.positive_state = None

        if negative_update:
            # reconstruct negative trie
            if self.ordered:
                new_negative_state = NegativeState(Trie(obj.ordered_negative_pool)) if obj.ordered_negative_pool else None
            else:
                new_negative_state = NegativeState(Trie(obj.negative_pool)) if obj.negative_pool else None
            if new_negative_state is not None:
                new_neg_states = set()
                if obj.hard_negative_state is not None:
                    for state in obj.hard_negative_state.state:
                        if state.parent_trie is None:
                            new_neg_states.add(new_negative_state.root)
                        else:
                            trace = state.trace_arcs()
                            new_state = new_negative_state.root.descend(trace)
                            if new_state is not None:
                                new_neg_states.add(new_state)
                obj.hard_negative_state = NegativeState(avoid_trie=new_negative_state.root, state=list(new_neg_states) if new_neg_states else None)
            else:
                obj.hard_negative_state = None

        if obj.positive_state is not None:
            # print(word_id,obj.get_new_id(word_id))
            # print(f"before: {obj.positive_state}")
            obj.positive_state = obj.positive_state.advance([word_id] + obj.get_new_id(word_id))
            # print(f"after: {obj.positive_state}")
            # assert obj.positive_state.met_phrases is None, "ill state"

            history = [s.trace_arcs() for s in obj.positive_state.state]
            newly_in_process = set()
            newly_literal_in_process = []
            for phrase in history:
                for clause in obj.clauses:
                    if not clause.satisfy: # and any([is_prefix(phrase, c) for c in clause.positive]):
                        hit_literal = [l for l in clause.positive if is_prefix(phrase, l)]
                        if hit_literal:
                            newly_literal_in_process.extend([(phrase, l[len(phrase):]) for l in hit_literal])
                            assert clause.idx not in obj.orders, 'clause has already satisfied, impossible state'
                            newly_in_process.add(clause.idx)
            obj.in_process = sorted(newly_in_process)
            obj.literal_in_process = newly_literal_in_process
        return obj


def init_batch(raw_constraints: List[ClauseConstraintList],
               beam_size: int,
               eos_id: Union[int, list],ordered=False) -> List[Optional[ConstrainedHypothesis]]:
    """
    :param raw_constraints: The list of clause constraints.
    :param beam_size: The beam size.
    :param eos_id: The target-language vocabulary ID of the EOS symbol.
    :return: A list of ConstrainedHypothesis objects (shape: (batch_size * beam_size,)).
    """
    constraints_list = [None] * (len(raw_constraints) * beam_size)  # type: List[Optional[ConstrainedHypothesis]]
    for i, raw_list in enumerate(raw_constraints):
        hyp = ConstrainedHypothesis(raw_list, eos_id,ordered=ordered)
        idx = i * beam_size
        constraints_list[idx:idx + beam_size] = [copy.deepcopy(hyp) for _ in range(beam_size)]
    return constraints_list


class ConstrainedCandidate:
    """
    Object used to hold candidates for the beam in topk().

    :param row: The row in the scores matrix.
    :param col: The column (word ID) in the scores matrix.
    :param score: the associated accumulated score.
    :param hypothesis: The ConstrainedHypothesis containing information about met constraints.
    """

    __slots__ = ('row', 'col', 'score', 'hypothesis', 'rank', 'ahead', 'literal', 'bucket', 'future')

    def __init__(self,
                 row: int,
                 col: int,
                 score: float,
                 hypothesis: ConstrainedHypothesis,
                 rank: float = None,
                 ahead: float = None,
                 literal: List[int] = None) -> None:
        self.row = row
        self.col = col
        self.score = score
        self.hypothesis = hypothesis
        self.rank = rank
        self.ahead = ahead
        self.literal = literal
        self.bucket = None
        self.future = None
        self.hypothesis.generation.append(col)

    def __hash__(self):
        return hash((self.row, self.col))

    def __eq__(self, other):
        return self.row == other.row and self.col == other.col

    def __str__(self):
        return '({}, {}, {}, {})'.format(self.row, self.col, self.score, self.hypothesis.num_met())


if __name__ == '__main__':
    clauses = [[[([3, 4, 5], True), ([3, 4], True), ([4, 5], True)], [([3, 4], True), ([6], True), ([7], True)]],
               [[([6], True), ([6, 7], True), ([6, 7, 8], True)], [([6, 9], True), ([6, 4, 9], True)]],
               [[([3, 4, 5], True)], [([3, 4], True)], [([4, 5], True)]],
               [[([3, 4], True)], [([2, 3, 5], True)], [([6, 5], True)]]]

    constraints = init_batch(raw_constraints=clauses,
                             beam_size=1,
                             eos_id=0)

    constraint = constraints[2]
    for w in [2, 3, 4, 5]:
        constraint = constraint.advance(w)
        print(constraint)
        print(constraint.positive_state)
        print(constraint.positive_state.allowed())
        print(constraint.met_order())
        print(constraint.clause_in_process())
        print()


