
import argparse
import os
import json
from utils_preprocess_image import save_process_img_for_llava
from utils_preprocess_prompt import gather_prompts_with_turns, generate_segmentation_mask
import time
import logging
from data_utils import *


def main(args):
    # load grounding_dino
    detector_id = args.detector_id 
    object_detector = load_detect(detector_id)
    # load SAM
    segmenter_id = args.segmenter_id
    segmentator, processor = load_segment(segmenter_id)
    samples_json = args.samples_json
    dataset = load_json(samples_json)
    threshold = args.threshold
    pred_iou_thresh_sam = args.pred_iou_thresh_sam
    mask_threshold_sam = args.mask_threshold_sam
    post_segments_dir = args.post_segments_dir
    mask_union = args.mask_union
    bbox_union = args.bbox_union
    add_second_adj = args.add_second_adj
    account4realtions = args.account4realtions
    should_process = args.should_process
    apply_general_mask = args.apply_general_mask
    dataset_type = args.dataset_type
    dataset_len = args.dataset_len
    start = args.start

    image_dir = args.image_dir
    
    output_dir = args.output_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    start_time = time.time()
    
    if args.prompt_gpt_file:
        prompt_data_gpt = load_json(args.prompt_gpt_file)
    
    # write to json
    samples_json_new = output_json_path(samples_json)
    with open(samples_json_new, 'w') as f:
        f.write('[')

        for item, data in enumerate(dataset):
            item += start
            target = data
            if dataset_len > 0 and item >= dataset_len:
                break
            # use lists to store the outputs via up-values
            if dataset_type == "llava_FT":
                image_id = target['id']
                # if image_id is not string
                if not isinstance(image_id, str):
                    image_id = str(image_id)
                # check if image_id is a path
                if "/" in image_id:
                    image_id = image_id.split("/")[-1]
                image_path = os.path.join(image_dir, target['image'])  
                # check if image_path exists
                if not os.path.exists(image_path):
                    image_path = os.path.join(image_dir,"coco","train2017", target['image'])           
                try:
                    if should_process:
                        processed_image = save_process_img_for_llava(image_path, output_dir)
                    else:
                        processed_path = image_path
                        processed_image = None 
                    
                    prompts_with_turns, i = gather_prompts_with_turns(target['id'], target['conversations'], prompt_data_gpt)
                    all_grounding_texts = [prompt_info['prompt'] for prompt_info in prompts_with_turns]
                    # TODO: name_labels
                    all_grounding_texts_name = [prompt_info['prompt_name'] for prompt_info in prompts_with_turns]
                    
                    # check if there were no prompts detected and skip 
                    if not all_grounding_texts:
                        print(f"No prompts for image {image_id}")
                        target['segment'] = {}
                        continue
                    # TODO: name_labels
                    image_array, detections_bbox, detections, similar_prompt_map, similar_prompt_list, labels_to_delete = grounded_segmentation(
                        image=processed_path if processed_image is None else processed_image,
                        labels=all_grounding_texts,  
                        threshold=threshold,
                        polygon_refinement=True,
                        detector_id=detector_id,
                        segmenter_id=segmenter_id,
                        object_detector=object_detector,
                        segmentator=segmentator, 
                        processor=processor,
                    )

                    if not detections:
                        print(f"No detections for image {image_id}")
                        target['segment'] = {}
                        continue
                    # delete labels with no mask from grounding_prompts
                    prompts_with_turns = [entry for entry in prompts_with_turns if entry["prompt"] not in labels_to_delete]
                    
                    #conversation['masks'] = {}
                    # if a prompt has more than a single mask then this will perform union of masks
                    bbox_dict, mask_dict = union_detection_results(detections, bbox_union=bbox_union, mask_union=mask_union, similar_prompt_list=similar_prompt_list) 
                    # change the labes to label_name and perform union of masks for the label_name, taking the turn into consideration
                    bbox_dict, mask_dict = union_bbox_masks_for_label_name(bbox_dict, mask_dict, prompts_with_turns)
                    bbox_dict = {key: [box.__dict__ for box in boxes] for key, boxes in bbox_dict.items()} #convert so we can dump to json
                    # remove masks that are similar to each other
                    mask_dict, similar_mask_map=filter_masks_based_iou(mask_dict, iou_threshold=0.95)
                    # save mask as png and save its path per each label
                    mask_img_dir = create_mask_dir_path(image_path_local=target['image'], output_dir=output_dir, image_id=image_id, post_segments_dir=post_segments_dir)
                    # save the mask image and tensor vector
                    prompt_mask_path, prompt_mask_vec_path, prompt_array_path = save_masks(mask_dict, output_dir, image_id, mask_img_dir, n_response = i, account4realtions=account4realtions)#, image_path_local=target['image'], image_dir=image_dir, post_segments_dir=post_segments_dir)
                    # use the mapping to add the mask path and the label that is maped to the mask_info (prompt_mask_path)
                    if similar_mask_map:
                        # add the lables that are mapped with the corresponding path
                        for key, value in similar_mask_map.items():
                            prompt_mask_path[key] = prompt_mask_path[value]
                            prompt_mask_vec_path[key] = prompt_mask_vec_path[value]
                    grounding_prompts = generate_segmentation_mask(prompts_with_turns, prompt_mask_path, prompt_mask_vec_path, bbox_dict)
                    # vizualizing
                    # if debug and detections_bbox:
                    #     for i, detection in enumerate(detections_bbox):
                    #         plot_detections(image_array, [detection], detection.label[:15], output_dir, image_id, index=i)

                    segmentations = []
                    for prompt_info in prompts_with_turns:
                        prompt_segmentations = []
                        for mask in grounding_prompts: #TODO: change segmentation masks
                            if isinstance(mask["prompt"],tuple):
                                for p in mask["prompt"]:
                                    if isinstance(p, str):
                                        p_name = p
                                    else:
                                        turn_name = p
                            else:
                                p_name = mask["prompt"]
                                turn_name=None
                            if turn_name:
                                if p_name == prompt_info['prompt_name'] and turn_name==prompt_info['turn']:
                                    prompt_segmentations.append({
                                        "prompt": prompt_info["prompt_name"],
                                        "mask_img": mask["mask_img"],
                                        "mask_rel": mask["mask_rel"],
                                        "bbox": mask["bbox"]
                                    })
                                    break
                            elif p_name == prompt_info['prompt_name']:
                                    prompt_segmentations.append({
                                        "prompt": prompt_info["prompt_name"],
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

                    target['segment'] = segmentations

                    if item != 0:
                        if not f.closed:
                            f.write(',')
                        else:
                            f = open(samples_json_new, 'a')
                    json.dump(target, f, indent=4)
                
                except Exception as e:
                    sample_id = target['id']  # replace with your actual sample_id
                    json_split = samples_json.split('/')[-1].split('.')[0]
                    # Write the problematic sample to a JSON file
                    problematic_jaon = os.path.join(output_dir, "failed_samples", f'problematic_sample_{json_split}.json')
                    if not os.path.exists(os.path.join(output_dir, "failed_samples")):
                        os.makedirs(os.path.join(output_dir, "failed_samples"))
                    selected_data = {
                        "id": data.get("id"),
                        "image": data.get("image"),
                        "conversations": data.get("conversations")
                    }
                    with open(problematic_jaon, 'a') as f:
                        json.dump(selected_data, f, indent=4)

                    # Log the sample_id and the error to a text file
                    logfile_error = os.path.join(output_dir, "failed_samples", f'error_log_{json_split}.txt')
                    logging.basicConfig(filename=logfile_error, level=logging.ERROR)
                    logging.error(f"An error occurred with sample_id {sample_id}, split {samples_json}: {e}")

                    print(f"An error occurred: {e}")
        
        f.write(']')

    # print statistics
    print(f"Processed {item} images.")
    print(f"Saved data to {samples_json_new}")
    print(f"time took to process all samples: {time.time()-start_time}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--segmenter_id", type=str, default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--detector_id", type=str, default="facebook/sam-vit-base") #facebook/sam-vit-huge
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--pred_iou_thresh_sam", type=float, default=0.88)
    parser.add_argument("--mask_threshold_sam", type=float, default=0.0)
    parser.add_argument("--mask_union", action='store_true')
    parser.add_argument("--bbox_union", action='store_true')
    parser.add_argument("--add_second_adj", action='store_true')
    parser.add_argument("--account4realtions", action='store_true')
    parser.add_argument("--should_process", action='store_true')
    parser.add_argument("--apply_general_mask", action='store_true')
    parser.add_argument("--dataset_type", type=str, default="llava_FT")
    parser.add_argument("--post_segments_dir", type=str, default="_segmented")
    parser.add_argument("--dataset_len", type=int, default=-1)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--image_dir", type=str, default="")
    parser.add_argument("--samples_json", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="") 
    parser.add_argument("--prompt_gpt_file", type=str, default="") 
    args = parser.parse_args()

    main(args)