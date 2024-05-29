import gradio
from topic_seg_labels import (
    get_ViT_B_16_laion2b_s34b_b88k_layer8_head_anno, get_ViT_B_16_laion2b_s34b_b88k_layer9_head_anno,
    get_ViT_B_16_laion2b_s34b_b88k_layer10_head_anno, get_ViT_B_16_laion2b_s34b_b88k_layer11_head_anno,
    get_ViT_B_16_openai_layer8_head_anno, get_ViT_B_16_openai_layer9_head_anno,
    get_ViT_B_16_openai_layer10_head_anno, get_ViT_B_16_openai_layer11_head_anno,
    get_ViT_B_32_datacomp_m_s128m_b4k_layer8_head_anno, get_ViT_B_32_datacomp_m_s128m_b4k_layer9_head_anno,
    get_ViT_B_32_datacomp_m_s128m_b4k_layer10_head_anno, get_ViT_B_32_datacomp_m_s128m_b4k_layer11_head_anno,
    get_ViT_B_32_openai_layer8_head_anno, get_ViT_B_32_openai_layer9_head_anno,
    get_ViT_B_32_openai_layer10_head_anno, get_ViT_B_32_openai_layer11_head_anno,
    get_ViT_L_14_laion2b_s32b_b82k_layer20_head_anno, get_ViT_L_14_laion2b_s32b_b82k_layer21_head_anno,
    get_ViT_L_14_laion2b_s32b_b82k_layer22_head_anno, get_ViT_L_14_laion2b_s32b_b82k_layer23_head_anno,
    get_ViT_L_14_openai_layer20_head_anno, get_ViT_L_14_openai_layer21_head_anno,
    get_ViT_L_14_openai_layer22_head_anno, get_ViT_L_14_openai_layer23_head_anno
)

def get_model_stats(modelname):
    if(modelname in ['ViT-L-14_laion2b_s32b_b82k', 'ViT-L-14_openai']):
        return 23
    elif(modelname in ['ViT-B-16_laion2b_s34b_b88k', 'ViT-B-16_openai', 'ViT-B-32_datacomp_m_s128m_b4k', 'ViT-B-32_openai']):
        return 11

def contrastive_seg_model_layer(modelname):
    print(modelname)
    if(modelname in ['ViT-B-16_laion2b_s34b_b88k', 'ViT-B-16_openai', 'ViT-B-32_datacomp_m_s128m_b4k', 'ViT-B-32_openai']):
        return gradio.Dropdown(choices=["0", "1", "2", "3",
                                        "4", "5", "6", "7",
                                        "8", "9", "10", "11", "-1 (Aggregate)"], label="Layer choices")
    elif(modelname in ['ViT-L-14_laion2b_s32b_b82k', 'ViT-L-14_openai']):
        return gradio.Dropdown(choices=["0", "1", "2", "3", "4", "5",
                                        "6", "7", "8", "9", "10", "11",
                                        "12", "13", "14", "15", "16", "17",
                                        "18", "19", "20", "21", "22", "23", "-1 (Aggregate)"], label="L-14 Layer choices")
    
def contrastive_seg_model_head(modelname):
    print(modelname)
    if(modelname in ['ViT-B-16_laion2b_s34b_b88k', 'ViT-B-16_openai', 'ViT-B-32_datacomp_m_s128m_b4k', 'ViT-B-32_openai']):
        return gradio.Dropdown(choices=["0", "1", "2", "3",
                                        "4", "5", "6", "7",
                                        "8", "9", "10", "11", "-1 (Aggregate)"], label="Layer choices")
    elif(modelname in ['ViT-L-14_laion2b_s32b_b82k', 'ViT-L-14_openai']):
        return gradio.Dropdown(choices=["0", "1", "2", "3", "4", "5",
                                        "6", "7", "8", "9", "10", "11",
                                        "12", "13", "14", "15", "16", "17",
                                        "18", "19", "20", "21", "22", "23", "-1 (Aggregate)"], label="L-14 Layer choices")

def change_model_layer(modelname):
    print(modelname)
    if(modelname == 'ViT-B-16_laion2b_s34b_b88k'):
        return gradio.Dropdown(choices=["8", "9", "10", "11"], label="ViT-B-16-laion Layer choices")
    elif(modelname == 'ViT-B-16_openai'):
        return gradio.Dropdown(choices=["8", "9", "10", "11"], label="ViT-B-16-openai Layer choices")
    elif(modelname == 'ViT-B-32_datacomp_m_s128m_b4k'):
        return gradio.Dropdown(choices=["8", "9", "10", "11"], label="ViT-B-32-datacomp Layer choices")
    elif(modelname == 'ViT-B-32_openai'):
        return gradio.Dropdown(choices=["8", "9", "10", "11"], label="ViT-B-32-openai Layer choices")
    elif(modelname == 'ViT-L-14_laion2b_s32b_b82k'):
        return gradio.Dropdown(choices=["20", "21", "22", "23"], label="ViT-L-14-laion Layer choices")
    elif(modelname == 'ViT-L-14_openai'):
        return gradio.Dropdown(choices=["20", "21", "22", "23"], label="ViT-L-14-openai Layer choices")

def change_model_layer_head(modelname, k):
    k = int(k)
    print(modelname)
    print(k)
    if(modelname == 'ViT-B-16_laion2b_s34b_b88k'):
        if(k == 8):
            return gradio.Dropdown(choices=get_ViT_B_16_laion2b_s34b_b88k_layer8_head_anno(), label="Layer-8 Head choices")
        elif(k == 9):
            return gradio.Dropdown(choices=get_ViT_B_16_laion2b_s34b_b88k_layer9_head_anno(), label="Layer-9 Head choices")
        elif(k == 10):
            return gradio.Dropdown(choices=get_ViT_B_16_laion2b_s34b_b88k_layer10_head_anno(), label="Layer-10 Head choices")
        elif(k == 11):
            return gradio.Dropdown(choices=get_ViT_B_16_laion2b_s34b_b88k_layer11_head_anno(), label="Layer-11 Head choices")
    elif(modelname == 'ViT-B-16_openai'):
        if(k == 8):
            return gradio.Dropdown(choices=get_ViT_B_16_openai_layer8_head_anno(), label="Layer-8 Head choices")
        elif(k == 9):
            return gradio.Dropdown(choices=get_ViT_B_16_openai_layer9_head_anno(), label="Layer-9 Head choices")
        elif(k == 10):
            return gradio.Dropdown(choices=get_ViT_B_16_openai_layer10_head_anno(), label="Layer-10 Head choices")
        elif(k == 11):
            return gradio.Dropdown(choices=get_ViT_B_16_openai_layer11_head_anno(), label="Layer-11 Head choices")
    elif(modelname == 'ViT-B-32_datacomp_m_s128m_b4k'):
        if(k == 8):
            return gradio.Dropdown(choices=get_ViT_B_32_datacomp_m_s128m_b4k_layer8_head_anno(), label="Layer-8 Head choices")
        elif(k == 9):
            return gradio.Dropdown(choices=get_ViT_B_32_datacomp_m_s128m_b4k_layer9_head_anno(), label="Layer-9 Head choices")
        elif(k == 10):
            return gradio.Dropdown(choices=get_ViT_B_32_datacomp_m_s128m_b4k_layer10_head_anno(), label="Layer-10 Head choices")
        elif(k == 11):
            return gradio.Dropdown(choices=get_ViT_B_32_datacomp_m_s128m_b4k_layer11_head_anno(), label="Layer-11 Head choices")
    elif(modelname == 'ViT-B-32_openai'):
        if(k == 8):
            return gradio.Dropdown(choices=get_ViT_B_32_openai_layer8_head_anno(), label="Layer-8 Head choices")
        elif(k == 9):
            return gradio.Dropdown(choices=get_ViT_B_32_openai_layer9_head_anno(), label="Layer-9 Head choices")
        elif(k == 10):
            return gradio.Dropdown(choices=get_ViT_B_32_openai_layer10_head_anno(), label="Layer-10 Head choices")
        elif(k == 11):
            return gradio.Dropdown(choices=get_ViT_B_32_openai_layer11_head_anno(), label="Layer-11 Head choices")
    elif(modelname == 'ViT-L-14_laion2b_s32b_b82k'):
        if(k == 20):
            return gradio.Dropdown(choices=get_ViT_L_14_laion2b_s32b_b82k_layer20_head_anno(), label="Layer-20 Head choices")
        elif(k == 21):
            return gradio.Dropdown(choices=get_ViT_L_14_laion2b_s32b_b82k_layer21_head_anno(), label="Layer-21 Head choices")
        elif(k == 22):
            return gradio.Dropdown(choices=get_ViT_L_14_laion2b_s32b_b82k_layer22_head_anno(), label="Layer-22 Head choices")
        elif(k == 23):
            return gradio.Dropdown(choices=get_ViT_L_14_laion2b_s32b_b82k_layer23_head_anno(), label="Layer-23 Head choices")
    elif(modelname == 'ViT-L-14_openai'):
        if(k == 20):
            return gradio.Dropdown(choices=get_ViT_L_14_openai_layer20_head_anno(), label="Layer-20 Head choices")
        elif(k == 21):
            return gradio.Dropdown(choices=get_ViT_L_14_openai_layer21_head_anno(), label="Layer-21 Head choices")
        elif(k == 22):
            return gradio.Dropdown(choices=get_ViT_L_14_openai_layer22_head_anno(), label="Layer-22 Head choices")
        elif(k == 23):
            return gradio.Dropdown(choices=get_ViT_L_14_openai_layer23_head_anno(), label="Layer-23 Head choices")

def change_model_property(modelname):
    print(modelname)
    if(modelname == 'ViT-B-16_laion2b_s34b_b88k'):
        return gradio.Dropdown(choices=["animals", "locations", "art", "subject", "nature"], label="ViT-B-16-laion Layer choices")
    elif(modelname == 'ViT-B-16_openai'):
        return gradio.Dropdown(choices=["animals", "locations"], label="ViT-B-16-openai Layer choices")
    elif(modelname == 'ViT-B-32_datacomp_m_s128m_b4k'):
        return gradio.Dropdown(choices=["animals", "colors"], label="ViT-B-32-datacomp Layer choices")
    elif(modelname == 'ViT-B-32_openai'):
        return gradio.Dropdown(choices=["photography", "pattern", "locations"], label="ViT-B-32-openai Layer choices")
    elif(modelname == 'ViT-L-14_laion2b_s32b_b82k'):
        return gradio.Dropdown(choices=["colors", "locations", "environment", "objects", "photography"], 
                            label="ViT-L-14-laion Layer choices")
    elif(modelname == 'ViT-L-14_openai'):
        return gradio.Dropdown(choices=["colors", "locations", "environment", "texture", "wildlife", "birds", "clothing"], 
                            label="ViT-L-14-openai Layer choices")
