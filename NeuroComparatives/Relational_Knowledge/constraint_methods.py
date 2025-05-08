import requests
from flair.data import Sentence
from flair.models import SequenceTagger
import torch
import inflect
import pickle
import sys
import copy
import os
import numpy as np
sys.path.insert(0,'..')
import utils
from lexical_constraints import init_batch
from eval.metrics import _compute_ppl

# tagger = SequenceTagger.load("flair/pos-english")
tagger = utils.load_tagger("flair/pos-english","../models/flair_pos_english/")
Inits = ["the", "many", "most", "typical"]
Singular_Inits = ["a", "a typical"]
Plural_Inits = ["the", "many", "most", "typical", "some"]

singular_determiners = ["each", "every", "any"] # a/an
plural_determiners = ["the", "all", "a few", "many", "most", "a lot of", "lots of","plenty of","a great number of", "a large number of"]

singular_determiners_short = ["every"] # a/an
plural_determiners_short = ["many", "most", "a lot of", "a great number of"] # + self

possessive_nouns = ["my", "our", "his", "her", "this person's", "someone's"]

verbs = ["are", "have", "want", "are seen as"]
singular_auxiliary_verbs = ["is", "can", "has", "needs", "would", "may"]
plural_auxiliary_verbs = ["are", "can", "have", "need", "would", "may"]
combined_auxiliary_verbs = ["is", "has", "needs", "are", "have", "need", "can", "would", "may"]
singular_auxiliary_verbs_comp = ["is", "has"]
plural_auxiliary_verbs_comp = ["are", "have"]
singular_auxiliary_verbs_past = ["is", "was", "can", "could", "has", "had", "needs", "needed", "will","would"]
plural_auxiliary_verbs_past = ["are", "were", "can", "could", "have", "had", "need", "needed","will","would"]
singular_auxiliary_verbs_long = ["be","been","being","is able to","is", "was","can","could", "has", "had","has to","had better", "needs", "will","would","dares","does","did","going to", "may","might", "must","ought to","should","shall"] 
plural_auxiliary_verbs_long = ["be","been","being", "are able to","are", "were","can","could", "have", "had","have to","had better", "need", "will","would","dare","do","did","going to", "may","might", "must","ought to","should","shall"] 
auxiliary_verbs_long = ["be","been","being","is able to", "are able to","are", "is", "are", "was", "were","can","could", "has", "have","had","has to","had better", "needs", "need", "will","would","dares","dare","does","do","did","going to", "may","might", "must","ought to","should","shall"] 

adverbs_of_frequency_short = ["typically", "often", "always", "generally","normally",]
adverbs_of_frequency = ["usually","often", "always", "generally","Normally", "Almost always","Annually","Constantly","Continually","Continuously","Daily","Frequently","Nearly always","Regularly",]
adverbs_of_frequency2 = ["Occasionally","Almost never", "Eventually","Ever","Hardly ever", "Hourly", "Infrequently", "Later", "Monthly","Never","next","Nightly", "Now","Periodically","Quarterly","Rarely","Scarcely ever","Seldom","Soon","Then","Today","Tonight","Weekly","Yearly","Yesterday","yet", "sometimes",]

with open(os.environ['UNCOUNTABLE_NOUNS_FILENAME'], "rb") as f:
    non_countable_words = pickle.load(f)
def _check_plural_singular(entity):
    plural = None
    label = entity.get_label('pos').value
    print(entity)
    if label in set(["NN","NNS","NNP","NNPS"]):
        if label in set(["NN","NNP"]):
            plural = False
        elif label in set(["NNS","NNPS"]):
            plural = True
    return plural

def _first_term_determiner(sentence):
    first_entity = sentence[0]
    DT = first_entity.get_label('pos').data_point.text
    label = first_entity.get_label('pos').value
    return label
def _entity_split_singular_plural(entity, sentence, p):
    ver1_entity = None
    ver2_entity = None
    # check determiner
    label = _first_term_determiner(sentence)
    # if label in ["DT"]:
    #     entity = " ".join([x.get_label('pos').data_point.text for x in sentence[1:]])
    # check plural/singular
    last_entity = sentence[-1]
    plural = _check_plural_singular(last_entity)
    if plural:
        ver1_entity = p.singular_noun(entity)
        ver2_entity = entity
    else:
        ver1_entity = entity
        ver2_entity = p.plural_noun(entity)
    if ver1_entity == False:
        ver1_entity = ver2_entity
    elif ver2_entity == False:
        ver2_entity = ver1_entity
    return ver1_entity, ver2_entity
def _check_uncountable(entity):
    if entity in non_countable_words:
        return True
    return False
def _create_constraint_clause(clause,polarity, max_count, min_count, type,look_ahead=None):
    created = {'terms': clause, 'polarity': polarity, 'max_count': max_count, 'min_count': min_count, 'type': type}
    if look_ahead:
        created["look_ahead"] = look_ahead
    return created
def _return_plural_version(ents,p):
    new_ents = []
    for e in ents:
        if _check_uncountable(e):
            new_ents.append(e)
            continue
        else:
            main_entity_tagged = Sentence(e)
            tagger.predict(main_entity_tagged)
            ver1_entity, ver2_entity = _entity_split_singular_plural(e,main_entity_tagged, p)
            new_ents.append(ver2_entity)
    return new_ents
def constraints3_prefix_aux_plural(main_entity,beam_size, tokenizer, eos_ids,POS, use_general_constraints,general_constriants, nodes, 
prefix_as_prompt=True,entity_as_prompt=True,adv_as_prompt = True,aux_as_prompt = True, 
long_det=False, long_adv=False,long_aux = False, 
aux_num=1, divide_aux=True,divide_adv=True,divide_det=True,
sometimes_no_det_adv_in_prompt=False,second_entity=None,entity_only=False,look_ahead=None,
no_tagger_and_case = False,plural_only=False, first_ent_negative_cons=False,second_entity_prompt=False,no_prefix = False,
entity_min_count=1,other_min_count=1):
    """
    sometimes_no_det_adv_in_prompt: True means generating prompts without determiner or adverb or both.
    second_entity: Not none means generating constrastive/comparative statements aginst another entity.
    
    Return:
        input_ids: [singular + plural]
        all_constraints: [singular + plural, num_aux * num_adv]
        all_constraint_list_str: [singular + plural, num_aux * num_adv]
    """
    global singular_determiners, plural_determiners, singular_determiners_short,plural_determiners_short,singular_auxiliary_verbs_long,plural_auxiliary_verbs_long,singular_auxiliary_verbs,    plural_auxiliary_verbs, adverbs_of_frequency, adverbs_of_frequency_short
    main_entity = main_entity.lower()
    first_entity = main_entity
    if second_entity_prompt:
        main_entity = second_entity
    uncountable = _check_uncountable(main_entity)
    print(f"{main_entity} is uncountable? {uncountable}")
    p = inflect.engine()
    main_entity_tagged = Sentence(main_entity)
    tagger.predict(main_entity_tagged)
    a = p.a(main_entity).split()[0]
    if uncountable:
        a = ""
    ver1_entity = None
    ver2_entity = None
    if entity_only:
        singular_determiners_short = [];plural_determiners_short=[];Singular_Prefixes=[];plural_determiners=[]
        singular_auxiliary_verbs_long = singular_auxiliary_verbs_comp;plural_auxiliary_verbs_long=plural_auxiliary_verbs_comp;singular_auxiliary_verbs=[];plural_auxiliary_verbs=[]
        adverbs_of_frequency = [];adverbs_of_frequency_short=[]
    if plural_only:
        plural_determiners_short = []; plural_determiners = []
    if long_det:
        Singular_Prefixes = [a] + singular_determiners
        Plural_Prefixes = plural_determiners
    else:
        Singular_Prefixes = [a] + singular_determiners_short
        Plural_Prefixes = plural_determiners_short
    if uncountable or (not prefix_as_prompt):
        Singular_Prefixes = []
        Plural_Prefixes = []
    if long_aux:
        Singular_Aux_Verbs = singular_auxiliary_verbs_long
        Plural_Aux_Verbs = plural_auxiliary_verbs_long
    else:
        Singular_Aux_Verbs = singular_auxiliary_verbs
        Plural_Aux_Verbs = plural_auxiliary_verbs
    if long_adv:
        Adverbs_Frequency = adverbs_of_frequency
    else:
        Adverbs_Frequency = adverbs_of_frequency_short
    print("singular_determiners_short, Singular_Prefixes, a:", singular_determiners_short,Singular_Prefixes,a)
    
    
    # 1 Case Handling
    aux_case_plural = None
    aux_case1_plural = None
    aux_case2_plural = None
    # 1.1 Entity 1 Case
    if no_tagger_and_case:
        # use the original case
        if plural_only and (not uncountable):
            # always prefer plural
            ver1_entity, ver2_entity = _entity_split_singular_plural(main_entity,main_entity_tagged, p)
            ver1_entity = ver2_entity
        else:
            ver1_entity = main_entity; ver2_entity = main_entity; 
    else:
        # split to singular and plural cases
        ver1_entity, ver2_entity = _entity_split_singular_plural(main_entity,main_entity_tagged, p)
    # if there is only 1 case for the entity
    if ver1_entity == ver2_entity:
        aux_case1_plural = _check_plural_singular(main_entity_tagged)

    # 1.2 2nd Entity Case
    # if 2nd entity exists. Note that if second_entity_prompt=True, the variables below wouldn't be used
    # Since main_entity would be second_entity
    if second_entity:
        second_entity_tagged = Sentence(second_entity)
        tagger.predict(second_entity_tagged)
        if no_tagger_and_case:
            if plural_only and (not uncountable):
                ver1_second_entity, ver2_second_entity = _entity_split_singular_plural(main_entity,main_entity_tagged, p)
                ver1_second_entity = ver2_second_entity
            else:
                ver1_second_entity = second_entity; ver2_second_entity = second_entity; 
        else:
            ver1_second_entity, ver2_second_entity = _entity_split_singular_plural(second_entity,second_entity_tagged, p)
        if ver1_second_entity == ver2_second_entity:
            aux_case2_plural = _check_plural_singular(second_entity_tagged)

    # 1.3 Set of Entities Case
    print(f"nodes: {nodes}")
    if no_tagger_and_case:
        if plural_only:
            if len(nodes[0]) > 0:
                temp_nodes = [x.lower() for x in nodes[0]]
                nodes = [_return_plural_version(temp_nodes,p)]
                # Plural_Aux_Verbs = combined_auxiliary_verbs

    if aux_case1_plural!=None:
        no_tagger_and_case = True
        aux_case_plural = aux_case1_plural
    if aux_case2_plural!=None:
        if second_entity_prompt:
            no_tagger_and_case = True
            aux_case_plural = aux_case2_plural
    if plural_only and ((not uncountable) or (len(nodes) > 0)):
        aux_case_plural = True
    print(ver1_entity,ver2_entity)
    print(f"first entity plural case: {aux_case1_plural}, second plural entity case: {aux_case2_plural}")
    print(f"only use 1 case: {no_tagger_and_case}, use plural case: {aux_case_plural}")
    
    # 1.3 prepare prompts and constraints
    prompt_comp_singular = []
    prompt_comp_plural = []
    constraint_comp_singular = []
    constraint_comp_plural = []
    if prefix_as_prompt:
        prompt_comp_singular.append(Singular_Prefixes)
        prompt_comp_plural.append(Plural_Prefixes)
    else:
        constraint_comp_singular.append(Singular_Prefixes)
        constraint_comp_plural.append(Plural_Prefixes)
    if entity_as_prompt:
        prompt_comp_singular.append([ver1_entity])
        prompt_comp_plural.append([ver2_entity])
    else:
        if len(nodes[0]) > 0:
            pass
        else:
            constraint_comp_singular.append([ver1_entity])
            constraint_comp_plural.append([ver2_entity])

    if aux_as_prompt:
        prompt_comp_singular.append(Singular_Aux_Verbs)
        prompt_comp_plural.append(Plural_Aux_Verbs)
    else:
        constraint_comp_singular.append(Singular_Aux_Verbs)
        constraint_comp_plural.append(Plural_Aux_Verbs)
    if adv_as_prompt:
        prompt_comp_singular.append(Adverbs_Frequency)
        prompt_comp_plural.append(Adverbs_Frequency)
    else:
        constraint_comp_singular.append(Adverbs_Frequency)
        constraint_comp_plural.append(Adverbs_Frequency)
    

    # 2 Add Prompts
    def add_to_prompt(prompt, comp):
        returned = []
        if len(comp) == 1:
            if len(comp[0]) == 0:
                returned.append(prompt)
            else:
                for x in comp[0]:
                    new_prompt = prompt+" "+x.lower()
                    returned.append(new_prompt)
        else:
            if len(comp[0]) == 0:
                returned.extend(add_to_prompt(prompt,comp[1:]))
            else:
                for x in comp[0]:
                    new_prompt = prompt+" "+ x.lower()
                    returned.extend(add_to_prompt(new_prompt,comp[1:]))
        return returned
    def add_to_prompt_template(input_ids,prompt_comp):
        if len(prompt_comp) == 0:
            input_ids.append("")
        else:
            input_ids.extend(add_to_prompt("",prompt_comp))
        return input_ids
    # print(f"entities: {ver1_entity} {ver2_entity}")
    print("prefixes")
    print(Singular_Prefixes, Plural_Prefixes,prompt_comp_singular)
    single_input_ids = []
    plural_input_ids = []
    single_input_ids = add_to_prompt_template(single_input_ids,prompt_comp_singular)
    plural_input_ids = add_to_prompt_template(plural_input_ids, prompt_comp_plural)
    if len(prompt_comp_plural) > 0 and (prefix_as_prompt==True) and (uncountable==False):
        plural_input_ids.append(f"{ver2_entity.capitalize()}")
    single_input_ids = [x.lstrip().capitalize() for x in single_input_ids]
    plural_input_ids = [x.lstrip().capitalize() for x in plural_input_ids]
    print(f"single_input_ids: {single_input_ids}, plural_input_ids: {plural_input_ids}")
    if sometimes_no_det_adv_in_prompt:
        single_input_ids_no = []
        plural_input_ids_no = []
        # no det
        temp_prompt_comp_singular = prompt_comp_singular[1:]
        temp_prompt_comp_plural = prompt_comp_plural[1:]
        single_input_ids_no = add_to_prompt_template(single_input_ids_no,temp_prompt_comp_singular)
        plural_input_ids_no = add_to_prompt_template(plural_input_ids_no, temp_prompt_comp_plural)

        # no adv
        temp_prompt_comp_singular = prompt_comp_singular[:3]
        temp_prompt_comp_plural = prompt_comp_plural[:3]
        single_input_ids_no = add_to_prompt_template(single_input_ids_no,temp_prompt_comp_singular)
        plural_input_ids_no = add_to_prompt_template(plural_input_ids_no, temp_prompt_comp_plural)
        # no det & adv
        temp_prompt_comp_singular = prompt_comp_singular[1:3]
        temp_prompt_comp_plural = prompt_comp_plural[1:3]
        single_input_ids_no = add_to_prompt_template(single_input_ids_no,temp_prompt_comp_singular)
        plural_input_ids_no = add_to_prompt_template(plural_input_ids_no, temp_prompt_comp_plural)
        single_input_ids_no = [x.lstrip().capitalize() for x in single_input_ids_no]
        plural_input_ids_no = [x.lstrip().capitalize() for x in plural_input_ids_no]
        single_input_ids = single_input_ids + single_input_ids_no
        plural_input_ids = plural_input_ids + plural_input_ids_no
    single_input_ids = [torch.LongTensor(tokenizer([x])['input_ids']).to('cuda') for x in single_input_ids]
    plural_input_ids = [torch.LongTensor(tokenizer([x])['input_ids']).to('cuda') for x in plural_input_ids]

    if no_tagger_and_case:
        if not aux_case_plural:
            input_ids = single_input_ids
        else:
            input_ids = plural_input_ids
    else:
        input_ids = single_input_ids + plural_input_ids

    # 3 Add Constraints
    all_constraints_singular = []
    all_constraints_plural = []
    all_constraint_list_str_singular = []
    all_constraint_list_str_plural = []
    constraint_list_str = []
    for node in nodes:
        if len(node) !=0:
            node = [x.lower() for x in node]
            constraint_list_str.append({'terms': node, 'polarity': 1, 'max_count': 1, 'min_count': 1, 'type': 'Term'}) 
            if look_ahead:
                constraint_list_str[-1]["look_ahead"] = look_ahead
    if use_general_constraints:
        if look_ahead:
            general_constraints_new = []
            for clause in general_constriants[0]:
                if clause["polarity"] >0:
                    clause["look_ahead"] = look_ahead
                general_constraints_new.append(clause)
            constraint_list_str.extend(general_constraints_new)
        else:
            constraint_list_str.extend(general_constriants[0])
    if first_ent_negative_cons:
        cased_first_entity = [first_entity]
        first_entity_tagged = Sentence(first_entity)
        tagger.predict(first_entity_tagged)
        uncountable = _check_uncountable(first_entity)
        if uncountable:
            cased_first_entity = [first_entity]
        else:
            ver1, ver2 = _entity_split_singular_plural(first_entity, first_entity_tagged, p)
            if ver1 == ver2:
                cased_first_entity = [ver2]
            else:
                cased_first_entity = [ver1, ver2]
        constraint_list_str.append(_create_constraint_clause(cased_first_entity,0,0,0,"Term"))
        # constraint_list_str.append(_create_constraint_clause([ver2_entity],0,0,0,"Term"))
    def add_to_constraint(pre_con,comp):
        returned = []
        if len(comp) == 1:
            if len(comp[0]) ==0:
                returned.append(pre_con)
            else:
                for x in comp[0]:
                    cur_con = [_create_constraint_clause([x.lower()],1,1,other_min_count,"Term",look_ahead=look_ahead)] + pre_con
                    returned.append(cur_con)
        else:
            if len(comp[0]) == 0:
                returned.extend(add_to_constraint(pre_con,comp[1:]))
            else:
                for x in comp[0]:
                    cur_con = [_create_constraint_clause([x.lower()],1,1,other_min_count,"Term",look_ahead=look_ahead)] + pre_con
                    returned.extend(add_to_constraint(cur_con,comp[1:]))
        return returned
    print(f"constraint_comp_singular: {constraint_comp_singular}")
    cur_constraints = copy.deepcopy(constraint_list_str)
    if len(constraint_comp_singular) > 0:
        if (not divide_aux) and (not divide_adv):
            if (not aux_as_prompt) and (not adv_as_prompt):
                if (len(nodes[0]) > 1) or (entity_as_prompt):
                    entity_clause = []
                else:
                    entity_clause = [_create_constraint_clause([ver1_entity],1,1,other_min_count,"Term",look_ahead=look_ahead)]
                cur_constriant_list = [entity_clause + 
                [_create_constraint_clause(Adverbs_Frequency,1,1,other_min_count,"Term",look_ahead=look_ahead)] + [_create_constraint_clause(Singular_Aux_Verbs,1,1,other_min_count,"Term",look_ahead=look_ahead)] + cur_constraints]
        else:
            cur_constriant_list = add_to_constraint(cur_constraints,constraint_comp_singular)
        # print(cur_constriant_list)
        if second_entity and (not second_entity_prompt):
            for cid in range(len(cur_constriant_list)):
                cur_constriant_list[cid] = cur_constriant_list[cid] + [_create_constraint_clause([ver1_second_entity],1,1,1,"Term",look_ahead=look_ahead)]
        for constriant in cur_constriant_list:
            constraint_list_str = [constriant]
            all_constraint_list_str_singular.append(constraint_list_str[0])
            constraints_list = utils.tokenize_constraints(tokenizer, POS, constraint_list_str)
            constraints = init_batch(raw_constraints=constraints_list,
                                         beam_size=beam_size,
                                         eos_id=eos_ids)
            all_constraints_singular.append(constraints)
    else:
        if second_entity and (not second_entity_prompt):
            cur_constraints.append(_create_constraint_clause([ver1_second_entity],1,1,other_min_count,"Term",look_ahead=look_ahead))
        all_constraint_list_str_singular.append(cur_constraints)
        constraints_list = utils.tokenize_constraints(tokenizer, POS, [cur_constraints])
        constraints = init_batch(raw_constraints=constraints_list,
                                        beam_size=beam_size,
                                        eos_id=eos_ids)
        all_constraints_singular.append(constraints)

    if len(constraint_comp_plural) > 0:
        if (not divide_aux) and (not divide_adv):
            if (not aux_as_prompt) and (not adv_as_prompt):
                if (len(nodes[0]) > 1) or (entity_as_prompt):
                    entity_clause = []
                else:
                    entity_clause = [_create_constraint_clause([ver2_entity],1,1,other_min_count,"Term",look_ahead=look_ahead)]
                cur_constriant_list = [entity_clause + 
                [_create_constraint_clause(Adverbs_Frequency,1,1,other_min_count,"Term",look_ahead=look_ahead)] + [_create_constraint_clause(Plural_Aux_Verbs,1,1,other_min_count,"Term",look_ahead=look_ahead)] + cur_constraints]
        else:
            cur_constriant_list = add_to_constraint(cur_constraints,constraint_comp_plural)
        if second_entity and (not second_entity_prompt):
            for cid in range(len(cur_constriant_list)):
                cur_constriant_list[cid] = cur_constriant_list[cid] + [_create_constraint_clause([ver1_second_entity],1,1,other_min_count,"Term",look_ahead=look_ahead)]
        for constriant in cur_constriant_list:
            constraint_list_str = [constriant]
            all_constraint_list_str_plural.append(constraint_list_str[0])
            constraints_list = utils.tokenize_constraints(tokenizer, POS, constraint_list_str)
            constraints = init_batch(raw_constraints=constraints_list,
                                         beam_size=beam_size,
                                         eos_id=eos_ids)
            all_constraints_plural.append(constraints)
    else:
        if second_entity and (not second_entity_prompt):
            cur_constraints.append(_create_constraint_clause([ver2_second_entity],1,1,other_min_count,"Term",look_ahead=look_ahead))
        all_constraint_list_str_plural.append(cur_constraints)
        constraints_list = utils.tokenize_constraints(tokenizer, POS, [cur_constraints])
        constraints = init_batch(raw_constraints=constraints_list,
                                        beam_size=beam_size,
                                        eos_id=eos_ids)
        all_constraints_plural.append(constraints)
    if no_tagger_and_case:
        if aux_case_plural:
            all_constraints = [all_constraints_plural] * len(plural_input_ids)
            all_constraint_list_str = [all_constraint_list_str_plural] * len(plural_input_ids) 
        else:
            all_constraints = [all_constraints_singular] * len(single_input_ids)
            all_constraint_list_str = [all_constraint_list_str_singular] * len(single_input_ids) 
    else:
        all_constraints = [all_constraints_singular] * len(single_input_ids) + [all_constraints_plural] * len(plural_input_ids)
        all_constraint_list_str = [all_constraint_list_str_singular] * len(single_input_ids) + [all_constraint_list_str_plural] * len(plural_input_ids)
    return input_ids,all_constraints,all_constraint_list_str

def all_type_knowledge(model,question,main_entity,beam_size, tokenizer, eos_ids,POS, use_general_constraints,general_constriants, nodes, 
prefix_as_prompt=True,entity_as_prompt=True,adv_as_prompt = True,aux_as_prompt = True, 
long_det=False, long_adv=False,long_aux = False, aux_num=1, divide_aux=True,divide_adv=True,conditional="",second_entity=None,entity_only=False,look_ahead=None):
    '''
    Prompt: Compared to entity1, 
    Constraint: entity2, aux verb, adverb, prefix(maybe not)
    '''
    input_ids, constraints,constraint_list_str = constraints3_prefix_aux_plural(main_entity,beam_size, tokenizer, eos_ids,POS, use_general_constraints,general_constriants, nodes, 
prefix_as_prompt=prefix_as_prompt,entity_as_prompt=entity_as_prompt,adv_as_prompt = adv_as_prompt,aux_as_prompt = aux_as_prompt, 
long_det=long_det, long_adv=long_adv,long_aux = long_aux, 
aux_num=aux_num, divide_aux=divide_aux,divide_adv=divide_adv,second_entity=second_entity,entity_only=entity_only,look_ahead=look_ahead,no_tagger_and_case=True,aux_conform_2nd_ent = True, first_ent_negative_cons=True)
    # print("in here")
    # print(len(input_ids))
    # print(len(constraints), len(constraints[0]))
    # print([tokenizer.decode(x[0]) for x in input_ids[:]])
    input_ids = [""] * len(input_ids)
    if conditional != "":
        method = prompt_addons[conditional]
        input_ids = method(main_entity,input_ids,tokenizer)

    return input_ids,constraints,constraint_list_str
def all_type_knowledge2(beam_size,question,main_entity, tokenizer, eos_ids,POS, use_general_constraints,general_constriants, nodes, 
prefix_as_prompt=True,entity_as_prompt=True,adv_as_prompt = True,aux_as_prompt = True, 
long_det=False, long_adv=False,long_aux = False, aux_num=1, divide_aux=True,divide_adv=True,conditional="",second_entity=None,entity_only=False,look_ahead=None):
    '''
    Prompt: Compared to (a/an, or plural) entity1, (prefix) entity2
    Constraint: aux verb, adverb,

    Prompt: Compared to (a/an, or plural) entity1,
    Constraint: aux verb, adverb, entity2
    '''
    entity_min_count = 1
    other_min_count = 1
    input_ids, constraints,constraint_list_str = constraints3_prefix_aux_plural(main_entity,beam_size, tokenizer, eos_ids,POS, use_general_constraints,general_constriants, nodes, 
prefix_as_prompt=prefix_as_prompt,entity_as_prompt=entity_as_prompt,adv_as_prompt = adv_as_prompt,aux_as_prompt = aux_as_prompt, 
long_det=long_det, long_adv=long_adv,long_aux = long_aux, 
aux_num=aux_num, divide_aux=divide_aux,divide_adv=divide_adv,second_entity=second_entity,
entity_only=entity_only,look_ahead=look_ahead,no_tagger_and_case=True, plural_only=True,
first_ent_negative_cons=True,second_entity_prompt =True,entity_min_count = entity_min_count,other_min_count = other_min_count)
    # print("in here")
    # print(len(input_ids))
    # print(len(constraints), len(constraints[0]))
    # print([tokenizer.decode(x[0]) for x in input_ids[:]])
    if conditional != "":
        method = prompt_addons[conditional]
        decoded = [tokenizer.decode(x[0]) for x in input_ids[:]]
        input_ids = method(main_entity,decoded,tokenizer)

    return input_ids,constraints,constraint_list_str
def prompt_addon1(ent1,input_ids,tokenizer):
    context = f"Compared to {ent1.lower()},"
    new_decoded = [f"{context}{x}" for x in input_ids]
    # print(new_decoded)
    encoded = [torch.LongTensor(tokenizer([x])['input_ids']).to('cuda') for x in new_decoded]
    return encoded
def prompt_addon2(ent1,input_ids,tokenizer):
    # if uncountable, then treat it the same as plural
    p = inflect.engine()
    main_entity_tagged = Sentence(ent1)
    tagger.predict(main_entity_tagged)
    a = p.a(ent1).split()[0]
    plural = _check_plural_singular(main_entity_tagged)
    uncountable = _check_uncountable(ent1)
    if uncountable:
        plural = True

    if plural:
        context = f"Compared to {ent1.lower()},"
    else:
        context = f"Compared to {a} {ent1.lower()},"
    new_decoded = [f"{context}{x.lower()}" if x=="" else f"{context} {x.lower()}" for x in input_ids]
    # print(new_decoded)
    encoded = [torch.LongTensor(tokenizer([x])['input_ids']).to('cuda') for x in new_decoded]
    return encoded
def prompt_addon3(ent1,input_ids,tokenizer):
    # always plural
    p = inflect.engine()
    main_entity_tagged = Sentence(ent1)
    tagger.predict(main_entity_tagged)
    uncountable = _check_uncountable(ent1)
    if uncountable:
        context = f"Compared to {ent1.lower()},"
    else:
        ver1, ver2 = _entity_split_singular_plural(ent1, main_entity_tagged, p)
        context = f"Compared to {ver2.lower()},"
    new_decoded = [f"{context}{x.lower()}" if x=="" else f"{context} {x.lower()}" for x in input_ids]
    # print(new_decoded)
    encoded = [torch.LongTensor(tokenizer([x])['input_ids']).to('cuda') for x in new_decoded]
    return encoded
prompt_addons = {"compared_to":prompt_addon1,"compared_to2":prompt_addon2,"compared_to3":prompt_addon3,}




constraints_map = {"all_type_knowledge2":all_type_knowledge2,"all_type_knowledge":all_type_knowledge,
"constraints3_prefix_aux_plural":constraints3_prefix_aux_plural}
