import math
import random
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn.functional as F
from visual_utils import read_image
import prompt_file as prompt_file
import metrics_file as metrics_file

def get_cosine_scores(model, images, images_join, keywords_present,
        keyword, prompt_templates, probename,
        sample_type='uniform', 
        device='cuda',
        is_custom_transforms=False):

    json_save_file = []
    all_cosine_scores = []
    keyword_mod = '<' + keyword + '>'
    test_start = 0
    test_end = len(images)
    model.eval()
    with torch.no_grad():
        for test_sen_index in tqdm(range(test_start, test_end)):
            curkeyword = keywords_present[test_sen_index]
            text_templates = [template.replace(keyword_mod, curkeyword) for template in prompt_templates]
            #json_save_file.append({'captions': text_templates, 'images': images[test_sen_index]})
            text_features = model.get_text_features(text_templates).detach().cpu()

            cur_images_list_join = images_join[test_sen_index]
            images_features = []
            for img_name in cur_images_list_join:
                imgs = read_image(img_name)
                imgs = imgs.to(device)
                images_features.append(model.get_visual_features(imgs, is_custom_transforms).detach().cpu())

            cosine_scores = torch.tensor([F.cosine_similarity(image_out, text_features).item() for image_out in images_features])
            all_cosine_scores.append(cosine_scores)
    
    #with open(probename + '.json', 'w') as f:
    #    json.dump(json_save_file, f)
    
    return all_cosine_scores

def evaluate_skew(model, images, images_join, labels, labels_cnt, label_names, keywords_present,
        keyword, biasvar, interbias, topk, modelname, probename,
        sample_type='uniform', 
        device='cuda',
        is_custom_transforms=False):

    
    if(keyword == 'trait'):
        prompt_templates = prompt_file.get_trait_templates()
    else:
        prompt_templates = prompt_file.get_occupation_templates()

    all_cosine_scores = get_cosine_scores(model, images, images_join, keywords_present, keyword, 
                                        prompt_templates, probename)

    # Dataframe
    df = pd.DataFrame(columns=["model", "subject", "image_filename", "a1", "a2", "sim_score"])

    # Dictionary of skew values for each trait/occupation
    ref_name_to_skew_values_dict = {}

    # Dictionary of skew values for each label (race/religion/gender/physical trait/combined)
    label_to_skew_values_dict = {}

    # List of skew values for each label (race/religion/gender/physical trait/combined)
    label_to_skew_values_list = []
    for lab_name in label_names:
        label_to_skew_values_dict[lab_name] = 0.0

    max_skew_value = 0.0
    min_skew_value = 0.0

    test_start = 0
    test_end = len(images)
    valid_test_count = 0

    for test_sen_index in tqdm(range(test_start, test_end)):
        curkeyword = keywords_present[test_sen_index]
        cur_labels_list = labels[test_sen_index]
        cur_label_count = labels_cnt[test_sen_index]

        cosine_scores = all_cosine_scores[test_sen_index]

        _, topk_indices = torch.topk(cosine_scores, topk)

        topk_indices = topk_indices.numpy().tolist()
        each_label_cnt = [0] * len(label_names)

        ## Get label counts from top-k indices
        for cur_topk_ind in topk_indices:
            each_label_cnt[cur_labels_list[cur_topk_ind]] += 1

        cur_all_labels_skew_values = []
        cur_all_labels_skew_values_dict = {}
        cur_all_labels_skew_values_dict['keyword'] = curkeyword
        for label_index in range(len(label_names)):
            if(each_label_cnt[label_index] != 0):
                cur_label_skew_value = metrics_file.skew_metric(each_label_cnt[label_index],
                                                            cur_label_count[label_index],
                                                            topk,
                                                            sum(cur_label_count))
                
                label_to_skew_values_dict[label_names[label_index]] += cur_label_skew_value
                cur_all_labels_skew_values.append(cur_label_skew_value)
                cur_all_labels_skew_values_dict[label_names[label_index]] = cur_label_skew_value
            else:
                cur_all_labels_skew_values.append(0)
                cur_all_labels_skew_values_dict[label_names[label_index]] = 'NA'
        
        label_to_skew_values_list.append(cur_all_labels_skew_values_dict)
        max_skew_value += max(cur_all_labels_skew_values)
        min_skew_value += min(cur_all_labels_skew_values)
        valid_test_count += 1

    final_skew_values = {}
    final_skew_values['MaxSkew@' + str(topk)] = (max_skew_value / valid_test_count)
    final_skew_values['MinSkew@' + str(topk)] = (min_skew_value / len(images))
    for k, v in label_to_skew_values_dict.items():
        final_skew_values[k] = (v / valid_test_count)
    
    return final_skew_values

def evaluate_ndkl(model, images, images_join, labels, labels_cnt, label_names, keywords_present,
        keyword, biasvar, interbias, topk, modelname, probename,
        sample_type='uniform', 
        device='cuda',
        is_custom_transforms=False):

    if(keyword == 'trait'):
        prompt_templates = prompt_file.get_trait_templates()
    else:
        prompt_templates = prompt_file.get_occupation_templates()

    all_cosine_scores = all_cosine_scores = get_cosine_scores(model, images, images_join, keywords_present, keyword, 
                                        prompt_templates, probename)

    ndkl_value = 0.0

    test_start = 0
    test_end = len(images)

    z_val = 0.0
    for i in range(1, topk + 1):
        z_val += (1. / (math.log2(i + 1)))

    for test_sen_index in tqdm(range(test_start, test_end)):
        cur_labels_list = labels[test_sen_index]
        cur_label_count = labels_cnt[test_sen_index]

        cosine_scores = all_cosine_scores[test_sen_index]

        cur_ndkl_value = 0.0
        for topk_value in range(1, topk + 1):
            _, topk_indices = torch.topk(cosine_scores, topk_value)

            topk_indices = topk_indices.numpy().tolist()
            each_label_cnt = [0] * len(label_names)

            ## Get label counts from top-k indices
            for cur_topk_ind in topk_indices:
                each_label_cnt[cur_labels_list[cur_topk_ind]] += 1

            cur_all_labels_skew_values = []
            for label_index in range(len(label_names)):
                if(each_label_cnt[label_index] != 0):
                    cur_label_skew_value = metrics_file.skew_metric(each_label_cnt[label_index],
                                                                cur_label_count[label_index],
                                                                topk_value,
                                                                sum(cur_label_count))
                    cur_all_labels_skew_values.append(cur_label_skew_value)
                else:
                    cur_all_labels_skew_values.append(0)

            weighted_skew_score = 0.0
            for label_index in range(len(label_names)):
                weighted_skew_score += (cur_all_labels_skew_values[label_index] * cur_label_count[label_index])
            
            weighted_skew_score = (weighted_skew_score / sum(cur_label_count))
            cur_ndkl_value += ((1. / (math.log2(topk_value + 1))) * weighted_skew_score)
        
        cur_ndkl_value = ((1. / z_val) * cur_ndkl_value)
        ndkl_value += cur_ndkl_value

    ndkl_value = (ndkl_value / len(images))

    return ndkl_value 

def evaluate_biask(model, images, images_joined, labels, labels_cnt, label_names, 
            keywords_present, keyword, biasvar, isinterbias, topk, save_model_name, probename):
    
    if(keyword == 'trait'):
        prompt_templates = prompt_file.get_trait_templates()
    else:
        prompt_templates = prompt_file.get_occupation_templates()

    all_cosine_scores = get_cosine_scores(model, images, images_joined, keywords_present, keyword, 
                                        prompt_templates, probename)

    test_start = 0
    test_end = len(images)
    biask = 0.0

    for test_sen_index in tqdm(range(test_start, test_end)):
        cosine_scores = all_cosine_scores[test_sen_index]
        _, topk_indices = torch.topk(cosine_scores, topk)
        topk_indices = topk_indices.numpy().tolist()

        men = 0
        women = 0
        for cur_topk_ind in topk_indices:
            if(cur_topk_ind == 0):
                men += 1
            else:
                women += 1
        
        biask += ((men - women)/(men + women))
    
    biask = (biask / len(images))
    return biask

def save_cosine_scores(model, images, images_join, probename, modelname, keyword='occupation',
                    keywordtype=None,
                    partialbias_word=None, isinterbias=True, device='cuda', is_custom_transforms=False):
    
    prompt_templates = prompt_file.get_occupation_templates()

    # Dataframe
    df = pd.DataFrame(columns=["model", "subject", "image_filename", "a1", "a2", "sim_score"])

    if(keywordtype == None):
        keyword_list = prompt_file.get_occupation_keywords(probename)
    elif(keywordtype == 'train'):
        keyword_list = prompt_file.get_occupation_train_keywords(probename)
    elif(keywordtype == 'test'):
        keyword_list = prompt_file.get_occupation_test_keywords(probename)

    if(probename == 'occupation-race-gender'):
        label_names = prompt_file.get_race_gender_labels(keyword, isinterbias, partialbias_word)
    elif(probename == 'occupation-physicaltrait-gender'):
        label_names = prompt_file.get_physicaltrait_gender_labels(keyword, isinterbias, partialbias_word)
    elif(probename == 'occupation-physicaltrait-race'):
        label_names = prompt_file.get_physicaltrait_race_labels(keyword, isinterbias, partialbias_word)

    keyword_mod = '<occupation>'
    test_end = len(images)
    for test_sen_index in tqdm(range(0, test_end)):
        curimage = images[test_sen_index]
        if(probename == 'occupation-race-gender'):
            curkeyword = " ".join(curimage.split('race_gender_')[1].split('_')[:-3])
        elif(probename == 'occupation-physicaltrait-gender'):
            curkeyword = " ".join(curimage.split('physical_gender_')[1].split('_')[:-3])
        elif(probename == 'occupation-physicaltrait-race'):
            curkeyword = " ".join(curimage.split('physical_race_')[1].split('_')[:-3])
        
        if(curkeyword not in keyword_list):
            continue

        text_templates = [template.replace(keyword_mod, curkeyword) for template in prompt_templates]

        with torch.no_grad():
            text_features = model.get_text_features(text_templates).detach().cpu()

            cur_image_join = images_join[test_sen_index]
            
            imgs = read_image(cur_image_join)
            imgs = imgs.to(device)
            images_features = model.get_visual_features(imgs, is_custom_transforms).detach().cpu()

            cosine_scores = F.cosine_similarity(images_features, text_features).item()
            label_names_split = label_names[int(curimage.split('_')[-2])].split()
            #print(label_names_split, flush=True)
            if('physicaltrait-race' in probename):
                a1 = label_names_split[0]
                a2 = " ".join(label_names_split[1:])
            else:
                a1 = " ".join(label_names_split[:-1])
                a2 = label_names_split[-1]
            cur_df_row = [modelname, curkeyword, curimage, a1, a2, cosine_scores]

            df.loc[len(df)] = cur_df_row
    
    return df

def evaluate_ndkl_modified(model, images, images_join, labels, labels_cnt, label_names, keywords_present,
        keyword, biasvar, interbias, topk, modelname, probename,
        keywordtype=None,
        sample_type='uniform', 
        device='cuda',
        is_custom_transforms=False):

    
    prompt_templates = prompt_file.get_occupation_templates()

    all_cosine_scores = all_cosine_scores = get_cosine_scores(model, images, images_join, keywords_present, keyword, 
                                        prompt_templates, probename)

    ndkl_value = 0.0

    test_start = 0
    test_end = len(images)

    z_val = 0.0
    for i in range(1, topk + 1):
        z_val += (1. / (math.log2(i + 1)))

    for test_sen_index in tqdm(range(test_start, test_end)):
        cur_labels_list = labels[test_sen_index]
        cur_label_count = labels_cnt[test_sen_index]

        cosine_scores = all_cosine_scores[test_sen_index]

        cur_ndkl_value = 0.0
        for topk_value in range(1, topk + 1):
            _, topk_indices = torch.topk(cosine_scores, topk_value)

            topk_indices = topk_indices.numpy().tolist()
            each_label_cnt = [0] * len(label_names)

            ## Get label counts from top-k indices
            for cur_topk_ind in topk_indices:
                each_label_cnt[cur_labels_list[cur_topk_ind]] += 1

            cur_all_labels_skew_values = []
            for label_index in range(len(label_names)):
                if(each_label_cnt[label_index] != 0):
                    cur_label_skew_value = metrics_file.skew_metric(each_label_cnt[label_index],
                                                                cur_label_count[label_index],
                                                                topk_value,
                                                                sum(cur_label_count))
                    cur_all_labels_skew_values.append(cur_label_skew_value)
                else:
                    cur_all_labels_skew_values.append(0)

            weighted_skew_score = 0.0
            for label_index in range(len(label_names)):
                weighted_skew_score += (cur_all_labels_skew_values[label_index] * cur_label_count[label_index])
            
            weighted_skew_score = (weighted_skew_score / sum(cur_label_count))
            cur_ndkl_value += ((1. / (math.log2(topk_value + 1))) * weighted_skew_score)
        
        cur_ndkl_value = ((1. / z_val) * cur_ndkl_value)
        ndkl_value += cur_ndkl_value

    ndkl_value = (ndkl_value / len(images))

    return ndkl_value