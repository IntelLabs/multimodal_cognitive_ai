import os
import re
import json
import math
import pandas as pd
from sklearn.model_selection import train_test_split
from prompt_file import (
    get_race_keywords, get_religion_keywords, get_occupation_keywords,
    get_traits_keywords, get_race_gender_labels, get_religion_gender_labels,
    get_race_religion_labels, get_physicaltrait_gender_labels,
    get_physicaltrait_religion_labels, get_physicaltrait_race_labels,
    get_occupation_train_keywords, get_occupation_test_keywords,
    get_physicaltrait_race_gender_labels
)

def get_images_for_the_category(image_list, image_list_joined, category_name):
    modified_image_list = []
    modified_image_list_joined = []
    for img_name, img_name_join in zip(image_list, image_list_joined):
        #print(img_name)
        if(category_name in " ".join(img_name.split("_")[:-2])):
            modified_image_list.append(img_name)
            modified_image_list_joined.append(img_name_join)
    
    return modified_image_list, modified_image_list_joined

def load_probe_data(all_images, all_images_joined, data_path, probename, keyword, bias_word,
                    partialbias_word,  keywordtype, isinterbias, topk, minimages):
    image_list = []
    image_list_joined = []
    for img, img_join in zip(all_images, all_images_joined):
        if(img.endswith('jpg')):
            image_list.append(img)
            image_list_joined.append(img_join)
    
    if(keyword == 'trait'):
        keyword_list = get_traits_keywords()
    else:
        if(keywordtype == None):
            keyword_list = get_occupation_keywords(probename)
        elif(keywordtype == 'train'):
            keyword_list = get_occupation_train_keywords(probename)
        elif(keywordtype == 'test'):
            keyword_list = get_occupation_test_keywords(probename)

    if(isinterbias):
        if(probename == 'occupation-race-gender'):
            label_names = get_race_gender_labels(keyword, isinterbias, partialbias_word)
        elif(probename == 'occupation-physicaltrait-gender'):
            label_names = get_physicaltrait_gender_labels(keyword, isinterbias, partialbias_word)
        elif(probename == 'occupation-physicaltrait-race'):
            label_names = get_physicaltrait_race_labels(keyword, isinterbias, partialbias_word)
        elif(probename == 'occupation-physicaltrait-race-gender'):
            label_names = get_physicaltrait_race_gender_labels(keyword, isinterbias, partialbias_word)
    else:
        if(probename == 'occupation-religion-gender'):
            label_names = get_religion_gender_labels(keyword, bias_word, partialbias_word)
        elif(probename == 'occupation-race-gender'):
            label_names = get_race_gender_labels(keyword, bias_word, partialbias_word)
        elif(probename == 'occupation-race-religion'):
            label_names = get_race_religion_labels(keyword, bias_word, partialbias_word)
        elif(probename == 'occupation-physicaltrait-gender'):
            label_names = get_physicaltrait_gender_labels(keyword, bias_word, partialbias_word)
        elif(probename == 'occupation-physicaltrait-religion'):
            label_names = get_physicaltrait_religion_labels(keyword, bias_word, partialbias_word)
        elif(probename == 'occupation-physicaltrait-race'):
            label_names = get_physicaltrait_race_labels(keyword, bias_word, partialbias_word)

    label_names_to_dict = {}
    for index in range(len(label_names)):
        label_names_to_dict[label_names[index]] = index

    if('-all' in probename):
        attach_name = '-all'
    elif('-manfil' in probename):
        attach_name = '-manfil'
    else:
        attach_name = ''

    ## Load meta-data file
    if(probename == 'occupation-race-gender'):
        df = pd.read_csv(os.path.join(data_path, keyword + '-race-gender' + attach_name + '-metadata.csv'))
        captions_cnt = 12
    elif(probename == 'occupation-physicaltrait-gender'):
        df = pd.read_csv(os.path.join(data_path, keyword + '-physicaltrait-gender-metadata.csv'))
        captions_cnt = 28
    elif(probename == 'occupation-physicaltrait-race'):
        df = pd.read_csv(os.path.join(data_path, keyword + '-physicaltrait-race-metadata.csv'))
        captions_cnt = 84
    elif(probename == 'occupation-physicaltrait-race-gender'):
        df = pd.read_csv(os.path.join(data_path, keyword + '-physicaltrait-race-gender-metadata.csv'))
        captions_cnt = 60

    #each_keyword_each_image_meta_data_captions = []
    #each_keyword_each_image_meta_data_captions.append(df['caption'][0])
    #for jj in range(1, captions_cnt):
    #    each_keyword_each_image_meta_data_captions.append(df['output_' + str(jj)][0])
    each_keyword_each_image_meta_data_captions = {}
    for colname in df.columns:
        if(colname.startswith('caption')):
            each_keyword_each_image_meta_data_captions[colname.split('_')[1]] = df[colname][0]

    print(each_keyword_each_image_meta_data_captions)

    image_list_for_all_keywords = []
    image_list_for_all_keywords_joined = []
    label_list_for_all_keywords = []
    label_cnt_for_all_keywords = []
    keywords_present_in_images = []
    for each_keyword in keyword_list:
        ## Get all the images with the trait name in image name
        each_keyword_images, each_keyword_images_joined = get_images_for_the_category(image_list,
                                                                                    image_list_joined,
                                                                                    each_keyword)

        if(len(each_keyword_images) == 0):
            continue

        #each_keyword_each_image_meta_data_captions = []
        #for ii in range(len(df)):
        #    if(each_keyword_images[0].startswith(str(df['filename'][ii]))):
        #        each_keyword_each_image_meta_data_captions.append(df['caption'][ii])
        #        for jj in range(1, captions_cnt):
        #            each_keyword_each_image_meta_data_captions.append(df['output_' + str(jj)][ii])
        #        break

        label_list_for_cur_keyword = []
        label_cnt_for_cur_keyword = [0] * len(label_names)
        each_keyword_images_modified = []
        each_keyword_images_modified_joined = []
        for each_keyword_each_image, each_keyword_each_image_join in zip(each_keyword_images, each_keyword_images_joined):
            if(len(each_keyword_each_image_meta_data_captions) == 0):
                continue
            #caption_index = int(each_keyword_each_image.split('_')[-2].split('.')[0])
            caption_index = each_keyword_each_image.split('_')[-2].split('.')[0]
            #print((each_keyword_each_image, caption_index), flush=True)
            current_caption = each_keyword_each_image_meta_data_captions[caption_index]
            current_caption_words = current_caption.split()
            for ext in label_names:
                c = 0
                for x in ext.split():
                    if(x in current_caption_words):
                        c += 1
                if(c == len(ext.split())):
                    label_cnt_for_cur_keyword[label_names_to_dict[ext]] += 1
                    label_list_for_cur_keyword.append(label_names_to_dict[ext])
                    each_keyword_images_modified.append(each_keyword_each_image)
                    each_keyword_images_modified_joined.append(each_keyword_each_image_join)
                    break
        
        assert len(each_keyword_images_modified) == len(label_list_for_cur_keyword)

        if(len(each_keyword_images_modified) >= minimages):
            image_list_for_all_keywords.append(each_keyword_images_modified)
            image_list_for_all_keywords_joined.append(each_keyword_images_modified_joined)
            label_cnt_for_all_keywords.append(label_cnt_for_cur_keyword)
            label_list_for_all_keywords.append(label_list_for_cur_keyword)
            keywords_present_in_images.append(each_keyword)
            #print((len(each_keyword_images_modified)), len(each_keyword_images_modified_joined))

    return [image_list_for_all_keywords, image_list_for_all_keywords_joined, label_list_for_all_keywords, 
            label_cnt_for_all_keywords, label_names, keywords_present_in_images]

def load_downstream(datapath, datasetname, biasattr):
    
    with open(datapath + datasetname + '.json', 'r') as f:
        data = json.load(f)

    image_list_for_all_keywords = []
    image_list_for_all_keywords_joined = []
    label_list_for_all_keywords = []
    label_cnt_for_all_keywords = []
    if(biasattr == 'gender'):
        label_names = ['male', 'female']
    else:
        label_names = []
    keywords_present_in_images = []

    label_names_to_dict = {}
    for index in range(len(label_names)):
        label_names_to_dict[label_names[index]] = index

    for json_list in data:
        image_list_for_all_keywords_joined.append(json_list['images'])
        keywords_present_in_images.append(json_list['captions'][0].split()[-1])

        image_list_for_cur_keyword = []
        label_list_for_cur_keyword = []
        label_cnt_for_cur_keyword = [0] * len(label_names_to_dict)
        for img in json_list['images']:
            image_list_for_cur_keyword.append(img.split('/')[-1])

            for ext in label_names:
                if(ext in img.split('.')[0].split('_')):
                    label_cnt_for_cur_keyword[label_names_to_dict[ext]] += 1
                    label_list_for_cur_keyword.append(int(label_names_to_dict[ext]))
                    break
        
        image_list_for_all_keywords.append(image_list_for_cur_keyword)
        label_list_for_all_keywords.append(label_list_for_cur_keyword)
        label_cnt_for_all_keywords.append(label_cnt_for_cur_keyword)

    return [image_list_for_all_keywords, image_list_for_all_keywords_joined, label_list_for_all_keywords, 
            label_cnt_for_all_keywords, label_names, keywords_present_in_images]
