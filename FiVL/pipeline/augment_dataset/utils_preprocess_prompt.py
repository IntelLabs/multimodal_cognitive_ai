import json

def generate_segmentation_mask(prompts, prompts2ask, prompts2vec_mask, prompt2bbox):
    prompts = [(entry['turn'], entry['prompt_name'] if 'prompt_name' in entry else entry['prompt']) for entry in prompts]
    if isinstance(prompts[0], tuple):
        prompts = list(set(prompts))
        prompts.sort()
    masks = []
    for prompt in prompts:
        masks.append({
            "prompt": prompt,
            "mask_img": prompts2ask[prompt],
            "mask_rel": prompts2vec_mask[prompt],
            "bbox": prompt2bbox[prompt]
        })
    return masks

def load_json_simple(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def gather_prompts_with_turns(image_id, conv_entery, prompt_data):
    prompts_with_turns = []
    exclude_gpt_prompt = ["none", "None", "token", "visual", "N/A", "No expressions"]
    for entry in prompt_data:
        if entry['id'] == image_id:
            turn_counter = 0
            # compare that the answer is the same in conversation and prompt data
            for conv in conv_entery:
                if conv['from'] == 'human': # question
                    save_last_question = conv['value']
                if conv['from'] == 'gpt': # answer
                    turn_counter += 1
                    if entry['answer'] == conv['value'] and entry['question'] == save_last_question: 
                        prompt_entries = entry['prompts'].split(':::')
                        for prompt_entry in prompt_entries:
                            prompt_entry = prompt_entry.strip()
                            if prompt_entry != '' and prompt_entry and not any(word in prompt_entry for word in exclude_gpt_prompt): # exclude any gpt prompt with the words in the list
                                if prompt_entry in conv['value']: # check if the prompt apears in the answer
                                    prompts_with_turns.append({
                                        "turn": turn_counter,
                                        "prompt": prompt_entry,
                                        "prompt_name": prompt_entry
                                    })
                                elif len(conv['value'].split()) == 1: # checks if the answer is a single word
                                    prompts_with_turns.append({
                                        "turn": turn_counter,
                                        "prompt": prompt_entry,
                                        "prompt_name": conv['value']
                                    })
                                else:
                                    print(f'Excluded prompt entry: not part of answer for multiple words answer, for {prompt_entry}')
                            else:
                                print(f'Excluded prompt entry {prompt_entry}')
    return prompts_with_turns, turn_counter

def generate_new_json(conversation_data, prompt_data):
    new_data = []
    for entry in conversation_data:
        image_id = entry['id']
        prompts_with_turns = gather_prompts_with_turns(image_id, entry['conversation'], prompt_data)
        prompt_list = [prompt_info['prompt'] for prompt_info in prompts_with_turns]
        segmentation_masks = generate_segmentation_mask(prompt_list)
        segmentations = []
        for prompt_info in prompts_with_turns:
            prompt_segmentations = []
            for mask in segmentation_masks:
                if mask["prompt"] == prompt_info['prompt']:
                    prompt_segmentations.append({
                        "prompt": mask["prompt"],
                        "mask_img": mask["mask_img"],
                        "mask_rel": mask["mask_rel"],
                        "bbox": mask["bbox"]
                    })
            turn_entry = next((item for item in segmentations if item['turn'] == prompt_info['turn']), None)
            if turn_entry:
                turn_entry['prompts'].extend(prompt_segmentations)
            else:
                segmentations.append({
                    "turn": prompt_info['turn'],
                    "prompts": prompt_segmentations
                })
        entry['segmentations'] = segmentations
        new_data.append(entry)
    return new_data

def main(conversation_file, prompt_file, output_file):
    conversation_data = load_json_simple(conversation_file)
    prompt_data = load_json_simple(prompt_file)
    new_data = generate_new_json(conversation_data, prompt_data)
    with open(output_file, 'w') as file:
        json.dump(new_data, file, indent=4)