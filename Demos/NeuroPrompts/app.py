import gradio as grad
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, AutoModel
from diffusers import StableDiffusionPipeline, DiffusionPipeline
from PIL import Image
import sys
import os
from categories import styles_list, artists_list, formats_list, perspective_list, booster_list, vibe_list
import random
import pandas as pd
import json
import datetime
import socket

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)


neurologic_path = os.path.abspath('neurologic/')
os.environ['NEUROLOGIC_PATH'] = neurologic_path
sys.path.insert(0,neurologic_path)
from neurologic_pe import generate_neurologic

from aesthetic import MLP, normalized
import clip

model_name = "Intel/NeuroPrompts"
model_type = 'finetuned' # set to 'ppo' if using ppo-trained model
rand_seed = 1535471403


in_txt_global = ""
out_txt_global = ""

def load_prompter():
  prompter_model = AutoModelForCausalLM.from_pretrained(model_name).to('cuda')
  tokenizer = AutoTokenizer.from_pretrained(model_name)

  tokenizer.pad_token = tokenizer.eos_token

  return prompter_model, tokenizer

prompter_model, prompter_tokenizer = load_prompter()

stable_diff_pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
stable_diff_pipe.to("cuda")
stable_diff_pipe.unet = torch.compile(stable_diff_pipe.unet, mode="reduce-overhead", fullgraph=True)
generator = torch.Generator(device="cuda")
print('Compiling graphs')
image_temp = stable_diff_pipe(prompt='A rabbit wearing a space suit', generator=generator, num_inference_steps=25).images[0]

# aesthetic model
model_aesthetic = MLP(768)  # CLIP embedding dim is 768 for CLIP ViT L 14
aesthetic_ckpts = torch.load("sac+logos+ava1-l14-linearMSE.pth")   # load the model you trained previously or the model available in this repo
model_aesthetic.load_state_dict(aesthetic_ckpts)
model_aesthetic.to("cuda")
model_aesthetic.eval()
aesthetic_clip, aesthetic_preprocess = clip.load("ViT-L/14", device="cuda")  #RN50x64   

# PickScore
processor_pickscore = AutoProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
model_pickscore = AutoModel.from_pretrained("yuvalkirstain/PickScore_v1").eval().to("cuda")


def generate(plain_text, curr_input_artist, curr_input_style, curr_input_format, curr_input_perspective, curr_input_booster, curr_input_vibe, curr_input_negative, length_penalty, max_length, beam_size, inference_steps):
    constraints = []
    for clause in [curr_input_artist, curr_input_style, curr_input_format, curr_input_perspective, curr_input_booster, curr_input_vibe]:
      if clause is not None and len(clause) > 0:
        constraints.append([clause.lower(), clause.title()])
    print('Positive constraints:')
    print(constraints)

    neg_constraints = []
    neg_inputs = [i.strip() for i in curr_input_negative.split(',')]
    for clause in neg_inputs:
        if clause is not None and len(clause) > 0:
            neg_constraints += [clause.lower(), clause.title()]
    print('Negative constraints:')
    print(neg_constraints)

    try:
        progress=grad.Progress(track_tqdm=True)
        progress(0, desc="Starting...")
    except:
        progress = None


    res = generate_neurologic(plain_text, 
                              model=prompter_model,
                              tokenizer=prompter_tokenizer,
                              model_type=model_type, 
                              constraint_method='clusters', 
                              clusters_file='template_keywords.json',
                              user_constraints = constraints if len(constraints) > 0 else None, 
                              negative_constraints = neg_constraints if len(neg_constraints) > 0 else None, 
                              length_penalty=float(length_penalty),
                              max_tgt_length=int(max_length),
                              beam_size=int(beam_size),
                              num_return_sequences=int(beam_size),
                              ngram_size=2, 
                              n_per_cluster=1,
                              seed=None)[0][0]

    
    satysfied_count = 0
    constraints_count = 0
    for clause in [curr_input_artist, curr_input_style, curr_input_format, curr_input_perspective, curr_input_booster, curr_input_vibe]:
      if clause != "":
        constraints_count += 1
        if clause in res:
            satysfied_count += 1
    
    print("AAA")
    print(f"{satysfied_count} out of {constraints_count} constraints are satisfied. \n")
    now = datetime.datetime.now()
    curr_time = now.strftime("%Y-%m-%d %H:%M:%S")
    data_dictionary = {
                      "plain_text": plain_text,
                      "curr_input_artist": curr_input_artist,
                      "curr_input_style": curr_input_style,
                      "curr_input_format": curr_input_format,
                      "curr_input_perspective": curr_input_perspective,
                      "curr_input_booster": curr_input_booster,
                      "curr_input_vibe": curr_input_vibe,
                      "curr_input_negative": curr_input_negative,
                      "length_penalty": length_penalty,
                      "max_length": max_length,
                      "beam_size": beam_size,
                      "inference_steps": inference_steps,
                      "curr_time": curr_time
    }
  
    with open("monitor_data.txt", 'a') as file:
      json.dump(data_dictionary, file)
      file.write('\n')  # Add a newline after each dictionary

    global in_txt_global
    global out_txt_global
    in_txt_global = plain_text
    out_txt_global = res

    res = {'text' : res, 'entities' : []}
    input_constraints = [curr_input_artist, curr_input_style, curr_input_format, curr_input_perspective, curr_input_booster, curr_input_vibe]
    constraint_labels = ['Artist', 'Style', 'Format', 'Perspective', 'Booster', 'Vibe']
    for i in range(len(input_constraints)):
      if input_constraints[i] is not None and len(input_constraints[i]) > 0:
        start = res['text'].find(input_constraints[i].lower())
        if start == -1:
          start = res['text'].find(input_constraints[i].title())
        if start > -1:
          res['entities'].append({'entity' : constraint_labels[i], 'start' : start, 'end' : start+len(input_constraints[i])})

    global out_txt_highlighted_global
    out_txt_highlighted_global = res

    return res


def write_upvote(in_txt):
  with open("upvote_downvote.txt", "a") as myfile:
    myfile.write(in_txt_global + "\t" + out_txt_global + "\t" + "upvote" + "\n")


def write_downvote(in_txt):
  with open("upvote_downvote.txt", "a") as myfile:
    myfile.write(in_txt_global + "\t" + out_txt_global + "\t" + "downvote" + "\n")


def clear():
  global out_txt_global
  out_txt_global = None
  return "", "", "", "", "", "", "", "",  "", None, None, None, None


def get_aesthetic_score_and_eval(prompt_original, inference_steps):
  if out_txt_global is not None:
    img_original = generate_original(prompt_original, inference_steps)
    table_scores = eval(prompt_original, img_original, out_image_global)

    return grad.Image.update(value=img_original), grad.Textbox.update(value=table_scores)
  else:
    return None, None


def generate_optimized(inference_steps):
    if out_txt_global is not None:
      generator.manual_seed(rand_seed)
      image = stable_diff_pipe(prompt=out_txt_global, generator=generator, num_inference_steps=inference_steps).images[0]
      global out_image_global
      out_image_global = image
      return image, image
    else:
      return None, None


def generate_original(plain_text, inference_steps):
    generator.manual_seed(rand_seed)
    image = stable_diff_pipe(prompt=plain_text, generator=generator, num_inference_steps=inference_steps).images[0]
    # get_aesthetic_score(image)

    return image

def eval(prompt_original, img_original, img_optimized):
  # call get_aesthetic_score
  # call get_pickscore

  img_original_aesthetic_score = get_aesthetic_score(img_original)
  img_optimized_aesthetic_score = get_aesthetic_score(img_optimized)

  img_original_aesthetic_score = img_original_aesthetic_score[0]
  img_optimized_aesthetic_score = img_optimized_aesthetic_score[0]

  concatenated_aesthetic_score = torch.cat([img_original_aesthetic_score, img_optimized_aesthetic_score], dim=0)
  probs_aesthetic = torch.softmax(concatenated_aesthetic_score, dim=-1)


  curr_pickscore = get_pickscore(prompt_original, [img_original, img_optimized])

  img_original_aesthetic_score_final = round(probs_aesthetic[0].item(), 2)
  img_optimized_aesthetic_score_final = round(probs_aesthetic[1].item(), 2)
  img_original_pickscore_final = round(curr_pickscore[0], 2)
  img_optimized_pickscore_final = round(curr_pickscore[1], 2)

  print(f"img_original_aesthetic_score_final: {img_original_aesthetic_score_final}")
  print(f"img_optimized_aesthetic_score_final: {img_optimized_aesthetic_score_final}")
  print(f"img_original_pickscore_final: {img_original_pickscore_final}")
  print(f"img_optimized_pickscore_final: {img_optimized_pickscore_final}")

  headers = ["Prompt", "Aesthetic", "PickScore"]

  out_df = pd.DataFrame({'Image' : ['Original prompt', 'Optimized prompt'], 
                         'Aesthetics score': [round(probs_aesthetic[0].item(), 2), round(probs_aesthetic[1].item(), 2)], 
                         'PickScore' : [round(curr_pickscore[0], 2), round(curr_pickscore[1], 2)]})

  data = [
        ["Original Prompt", round(probs_aesthetic[0].item(), 2), round(curr_pickscore[0], 2)],
        ["Optimized Prompt", round(probs_aesthetic[1].item(), 2), round(curr_pickscore[1], 2)]
    ]

  table_string = get_table(headers, data)

  return out_df


def get_table(headers, data):
    ans = ""
    row_format = "{:<30} {:<20} {:<20}"
    ans += row_format.format(*headers)
    ans += '\n'
    ans += '-' * 65
    ans += '\n'

    for row in data:
        ans += row_format.format(*row)
        ans += '\n'


    return ans

  
def get_aesthetic_score(test_image):
  image = aesthetic_preprocess(test_image).unsqueeze(0).to("cuda")
  with torch.no_grad():
    image_features = aesthetic_clip.encode_image(image)

  im_emb_arr = normalized(image_features.cpu().detach().numpy() )
  prediction = model_aesthetic(torch.from_numpy(im_emb_arr).to("cuda").type(torch.cuda.FloatTensor))
  print(f"Aesthetic score predicted by the model: {prediction.item()}")
  
  return prediction


def get_pickscore(prompt, images):
  image_inputs = processor_pickscore(images=images, padding=True, truncation=True, max_length=77, return_tensors="pt").to("cuda")
  text_inputs = processor_pickscore(text=prompt, padding=True, truncation=True, max_length=77, return_tensors="pt").to("cuda")
  with torch.no_grad():
      image_embs = model_pickscore.get_image_features(**image_inputs)
      image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
      text_embs = model_pickscore.get_text_features(**text_inputs)
      text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)
      scores = model_pickscore.logit_scale.exp() * (text_embs @ image_embs.T)[0]
      print(f"scores: {scores}")
      probs = torch.softmax(scores, dim=-1)
  return probs.cpu().tolist()

def clear_previous_outputs():
  return grad.Image.update(value=None), grad.Image.update(value=None), grad.Image.update(value=None), grad.Textbox.update(value=None)


def change_tab():
  return grad.Tabs.update(selected=0)


def toggle_legend():
  if len(out_txt_highlighted_global['entities']) > 0:
    return grad.HighlightedText.update(show_legend=True)
  else:
    return grad.HighlightedText.update(show_legend=False)


examples = [["A rabbit is wearing a space suit", "Greg Rutkowski", "pop art", "photograph", "from below", 1.0, 128], 
              ["Elon Musk as punisher", "Leonardo da Vinci", "baroque", "painting", "wide angle", 1.0, 128]]




css = "#out_image {max-width: 55%; width: auto;}"

with grad.Blocks(css=css) as demo:
  grad.Markdown(
            """
            # NeuroPrompts Demo
            NeuroPrompts is an interface to Stable Diffusion which automatically optimizes a user's prompt for improved image aesthetics while maintaining stylistic control according to the user's preferences.
            """
        )
  with grad.Row():
    with grad.Column(scale=1):
      input_txt = grad.Textbox(label=" Input prompt", placeholder="Enter your prompt").style(container=False)
      with grad.Box():
        input_artist = grad.Dropdown(artists_list, label="Artist", allow_custom_value=True).style(container=False)
        input_style = grad.Dropdown(styles_list, label="Style", allow_custom_value=True).style(container=False)
        input_format = grad.Dropdown(formats_list, label="Format", allow_custom_value=True).style(container=False)
        input_perspective = grad.Dropdown(perspective_list, label="Perspective", allow_custom_value=True).style(container=False)
        input_booster= grad.Dropdown(booster_list, label="Booster", allow_custom_value=True).style(container=False)
        input_vibe = grad.Dropdown(vibe_list, label="Vibe", allow_custom_value=True).style(container=False)
      with grad.Box():
        input_negative = grad.Textbox(label="Negative constraints", info="Enter a comma-separated sequence of terms that should not appear in the optimized prompt").style(container=False)
      with grad.Accordion(label="Parameters", open=False) as og:
        input_lp = grad.Number(value=1.0, label = "Length penalty").style(container=False)
        input_max_length = grad.Number(value=77, label="Max length").style(container=False)
        input_beam_size = grad.Number(value=5, label="Beam size").style(container=False)
        input_inference_steps = grad.Number(value=25, label="Stable Diffusion inference steps").style(container=False)

      with grad.Row():
        greet_btn = grad.Button("Submit", variant="primary")
        clear_btn = grad.Button("Clear", variant="secondary")

    with grad.Column(scale=2):
      out_txt = grad.HighlightedText(label="Optimized Prompt", show_legend=True)
      with grad.Tabs() as tabs:      
        with grad.TabItem("Optimized image only", id=0):
          out_image = grad.Image(label="Optimized Image", type='pil', elem_id="out_image")#.style(height=450, width=550)
        with grad.TabItem("Side-by-side comparison", id=1) as comparison:
          with grad.Row():
            out_image_2 = grad.Image(label="Optimized Image", type='pil', elem_id="out_image")#.style(height=450, width=550)
            out_image_original_prompt = grad.Image(label="Original Image", type='pil', elem_id="out_image_original")#.style(height=450, width=550)
          out_eval = grad.Dataframe(label="Evaluation scores")
      

      with grad.Row():
        up_vote = grad.Button(value="👍 Upvote", interactive=True)
        up_vote.click(fn=write_upvote, inputs=[input_txt])

        down_vote = grad.Button(value="👎 Downvote", interactive=True)
        down_vote.click(fn=write_downvote, inputs=[input_txt])


  greet_btn.click(change_tab, None, tabs)
  greet_btn.click(fn=clear_previous_outputs,
                    inputs=[],
                    outputs=[out_image, out_image_2, out_image_original_prompt, out_eval])
  greet_btn.click(fn=generate,
                    inputs=[input_txt, input_artist, input_style, input_format, input_perspective, input_booster, input_vibe, input_negative, input_lp, input_max_length, input_beam_size, input_inference_steps],
                    outputs=[out_txt],
                    api_name="submit")
  out_txt.change(toggle_legend, None, out_txt)
  out_txt.change(generate_optimized,
                 inputs=[input_inference_steps],
                 outputs=[out_image, out_image_2])
  

  clear_btn.click(fn=clear, 
                  inputs=None, 
                  outputs=[input_txt, input_artist, input_style, input_format, input_perspective, input_booster, input_vibe, input_negative, out_txt, out_image, out_image_2, out_image_original_prompt, out_eval])

  comparison.select(fn=get_aesthetic_score_and_eval,
                     inputs=[input_txt, input_inference_steps],
                     outputs=[out_image_original_prompt, out_eval])


enable_queue = True
demo.queue(concurrency_count=3)
host = socket.getfqdn() if '.intel.com' in socket.getfqdn() else '0.0.0.0'
demo.launch(enable_queue=enable_queue, share=False, server_name=host)
