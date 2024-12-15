from dataclasses import dataclass
from typing import Any, List, Dict, Optional, Union, Tuple
import cv2
import os
import torch
import requests
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from transformers import AutoModelForMaskGeneration, AutoProcessor, pipeline
import json
from collections import defaultdict
from utils_preprocess_image import reshape_mask_to_mask_vector

# Using grounded-SAM implementation from: https://github.com/NielsRogge/Transformers-Tutorials/blob/master/Grounding%20DINO/GroundingDINO_with_Segment_Anything.ipynb
# storing the results of grounding DINO
@dataclass
class BoundingBox:
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def xyxy(self) -> List[float]:
        return [self.xmin, self.ymin, self.xmax, self.ymax]

@dataclass
class DetectionResult:
    score: float
    label: str
    box: BoundingBox
    mask: Optional[np.array] = None

    @classmethod
    def from_dict(cls, detection_dict: Dict) -> 'DetectionResult':
        return cls(score=detection_dict['score'],
                   label=detection_dict['label'],
                   box=BoundingBox(xmin=detection_dict['box']['xmin'],
                                   ymin=detection_dict['box']['ymin'],
                                   xmax=detection_dict['box']['xmax'],
                                   ymax=detection_dict['box']['ymax']))

# utils

def mask_to_polygons(mask: np.ndarray) -> List[List[int]]:
    # Find contours in the binary mask
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    area_threshold = 0.1 *  cv2.contourArea(largest_contour)
    # Find the contour with the largest area
    large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > area_threshold]

    # Extract the vertices of the contour
    polygons = [cnt.reshape(-1, 2).tolist() for cnt in large_contours]

    return polygons

def polygon_to_mask(polygon: List[Tuple[int, int]], image_shape: Tuple[int, int]) -> np.ndarray:
    # convert a polygon to a segmentation mask.
    mask = np.zeros(image_shape, dtype=np.uint8)
    pts = np.array(polygon, dtype=np.int32)
    cv2.fillPoly(mask, [pts], color=(255,))
    return mask

def load_image(image_str: str) -> Image.Image:
    if image_str.startswith("http"):
        image = Image.open(requests.get(image_str, stream=True).raw).convert("RGB")
    else:
        image = Image.open(image_str).convert("RGB")
    return image

def get_boxes(results: DetectionResult) -> List[List[List[float]]]:
    boxes = []
    for result in results:
        xyxy = result.box.xyxy
        boxes.append(xyxy)

    return [boxes]

def refine_masks(masks: torch.BoolTensor, polygon_refinement: bool = False) -> List[np.ndarray]:
    masks = masks.cpu().float()
    masks = masks.permute(0, 2, 3, 1)
    masks = masks.mean(axis=-1)
    masks = (masks > 0).int()
    masks = masks.numpy().astype(np.uint8)
    masks = list(masks)

    if polygon_refinement:
        for idx, mask in enumerate(masks):
            shape = mask.shape
            polygons = mask_to_polygons(mask)
            mask = np.zeros(shape, dtype=np.uint8)
            for j, polygon in enumerate(polygons):
                mask_part = polygon_to_mask(polygon, shape)
                mask += mask_part
            masks[idx] = mask

    return masks

# # plot utils
# def annotate(image: Union[Image.Image, np.ndarray], detection_results: List[DetectionResult]) -> np.ndarray:
#     # add bounding boxes and masks
#     image_cv2 = np.array(image) if isinstance(image, Image.Image) else image
#     image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_RGB2BGR)
#     for detection in detection_results:
#         label = detection.label
#         score = detection.score
#         box = detection.box
#         mask = detection.mask
#         color = np.random.randint(0, 256, size=3)
#         cv2.rectangle(image_cv2, (box.xmin, box.ymin), (box.xmax, box.ymax), color.tolist(), 2)
#         cv2.putText(image_cv2, f'{label}: {score:.2f}', (box.xmin, box.ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color.tolist(), 2)

#         # If mask is available, apply it
#         if mask is not None:
#             # Convert mask to uint8
#             mask_uint8 = (mask * 255).astype(np.uint8)
#             contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#             cv2.drawContours(image_cv2, contours, -1, color.tolist(), 2)

#     return cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)

# def plot_detections(
#     image: Union[Image.Image, np.ndarray],
#     detections: List[DetectionResult],
#     save_name: Optional[str] = None,
#     output_dir: Optional[str] = None,
#     image_id: Optional[str] = None,
#     index: Optional[int] = None,
# ) -> None:
#     annotated_image = annotate(image, detections)
#     plt.imshow(annotated_image)
#     plt.axis('off')
#     if save_name:
#         # convert label into a valid filename
#         save_name = save_name.replace(" ", "_").replace(":", "").replace(".", "").replace(" ", "_").lower()+f"_{index}"+".png"
#         output_dir_bbox = os.path.join(output_dir+"_bbox", image_id)
#         if not os.path.exists(output_dir_bbox):
#             os.makedirs(output_dir_bbox)
#         output_file = os.path.join(output_dir_bbox, save_name)
#         plt.savefig(output_file, bbox_inches='tight')
#         plt.close()

def calculate_iou(box1, box2):
    if not isinstance(box1, np.ndarray):
        x1 = box1.xmin
        y1 = box1.ymin
        x2 = box1.xmax
        y2 = box1.ymax
        x1_p = box2.xmin
        y1_p = box2.ymin
        x2_p = box2.xmax
        y2_p = box2.ymax
        inter_x1 = max(x1, x1_p)
        inter_y1 = max(y1, y1_p)
        inter_x2 = min(x2, x2_p)
        inter_y2 = min(y2, y2_p)
        
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x2_p - x1_p) * (y2_p - y1_p)
        union_area = box1_area + box2_area - inter_area
    else:
        intersection = np.logical_and(box1, box2)
        union = np.logical_or(box1, box2)
        inter_area = np.sum(intersection)
        union_area = np.sum(union)
    
    iou = inter_area / union_area if union_area > 0 else 0
    return iou

def filter_masks_based_iou(mask_dict, iou_threshold=0.95):
    similar_mask_map = {}
    keys = list(mask_dict.keys())
    i = 0
    while i < len(keys):
        if keys[i] not in mask_dict:  # Skip if the mask has been deleted
            i += 1
            continue
        j = i + 1
        while j < len(keys):
            if keys[j] not in mask_dict:  # Skip if the mask has been deleted
                j += 1
                continue
            iou = calculate_iou(mask_dict[keys[i]][0], mask_dict[keys[j]][0])
            if iou > iou_threshold:
                similar_mask_map[keys[j]] = keys[i]
                del mask_dict[keys[j]]
            else:
                j += 1
        i += 1
    return mask_dict, similar_mask_map

# grounded sam
def load_detect(
    detector_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Use Grounding DINO to detect a set of labels in an image in a zero-shot fashion.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector_id = detector_id if detector_id is not None else "IDEA-Research/grounding-dino-tiny"
    object_detector = pipeline(model=detector_id, task="zero-shot-object-detection", device=device)
    return object_detector

def detect(
    image: Image.Image,
    labels: List[str],
    threshold: float = 0.3,
    detector_id: Optional[str] = None,
    object_detector: Optional[pipeline] = None
) -> List[Dict[str, Any]]:
    """
    Use Grounding DINO to detect a set of labels in an image in a zero-shot fashion.
    """
    labels = [label if label.endswith(".") else label+"." for label in labels]
    if labels:
        results = object_detector(image,  candidate_labels=labels, threshold=threshold)
        results = [DetectionResult.from_dict(result) for result in results]
    else:
        results = []

    return results

def load_segment(
    segmenter_id: Optional[str] = None,
) -> List[DetectionResult]:
    """
    Use Segment Anything (SAM) to generate masks given an image + a set of bounding boxes.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    segmenter_id = segmenter_id if segmenter_id is not None else "facebook/sam-vit-base"

    segmentator = AutoModelForMaskGeneration.from_pretrained(segmenter_id).to(device)
    processor = AutoProcessor.from_pretrained(segmenter_id)
    return segmentator, processor

def segment(
    image: Image.Image,
    detection_results: List[Dict[str, Any]],
    polygon_refinement: bool = False,
    segmenter_id: Optional[str] = None,
    segmentator: Optional[AutoModelForMaskGeneration] = None,
    processor: Optional[AutoProcessor] = None,
    pred_iou_thresh_sam: float =0.88,
    mask_threshold_sam: float =0.0
) -> List[DetectionResult]:
    """
    Use Segment Anything (SAM) to generate masks given an image + a set of bounding boxes.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    boxes = get_boxes(detection_results)
    inputs = processor(images=image, input_boxes=boxes, return_tensors="pt").to(device)

    outputs = segmentator(**inputs)
    masks = processor.post_process_masks(
        masks=outputs.pred_masks,
        original_sizes=inputs.original_sizes,
        reshaped_input_sizes=inputs.reshaped_input_sizes,
    )[0]
    masks = refine_masks(masks, polygon_refinement)
    for detection_result, mask in zip(detection_results, masks):
        detection_result.mask = mask

    return detection_results

def grounded_segmentation(
    image: Union[Image.Image, str],
    labels: List[str],
    threshold: float = 0.3,
    polygon_refinement: bool = False,
    detector_id: Optional[str] = None,
    segmenter_id: Optional[str] = None,
    object_detector: Optional[pipeline] = None,
    segmentator: Optional[AutoModelForMaskGeneration] = None,
    processor: Optional[AutoProcessor] = None,
    pred_iou_thresh_sam: float =0.88,
    mask_threshold_sam: float =0.0
) -> Tuple[np.ndarray, List[DetectionResult]]:
    if isinstance(image, str):
        image = load_image(image)
    detections_general = []
    similar_prompt_map = {}
    to_delete = set()
    for l in range(len(labels)):
        if isinstance(labels[l], tuple):
            gen_label = labels[l]
            for i in range(len(gen_label)):
                label = [gen_label[i]]
                detections_general_temp = detect(image, label, threshold, detector_id, object_detector)
                if len(detections_general_temp) > 0:
                    detections_general+=detections_general_temp
                    to_delete.add(l)
                    break
    # delete all general labels from labels, leave only token labels
    labels = [labels[i] for i in range(len(labels)) if i not in to_delete]
    detections_token = detect(image, labels, threshold, detector_id, object_detector) 
    detections = detections_general + detections_token
    # print the labels for which there was no bbox detected
    detected_labels = set(d.label.strip(" .") for d in detections)
    labels_to_delete = []    
    for label in labels:
        if label not in detected_labels:
            print(f"No detection for: {label}")
            labels_to_delete.append(label)     
    # combine the bbox 
    bbox_dict, _ = union_detection_results(detections, bbox_union=False, mask_union=False) 
    # check if the masks arrays in the detections_token are the same as for the general mask for some tokens and delete them
    i = 0
    similar_prompt_list=[]
    while i < len(detections):
        j = i + 1
        while j < len(detections):
            iou = calculate_iou(detections[i].box, detections[j].box)
            if iou > 0.95:
                similar_prompt_map[detections[j].label.rstrip(' .')] = detections[i].label.rstrip(' .')
                similar_prompt_list.append({"original": detections[j].label.rstrip(' .'), "mapped_to_index": i})
                del detections[j]
            else:
                j += 1
        i += 1
    if len(detections) != 0:
        if len(detections) > 150:
            def chunker(seq, num):
                avg = len(seq) // num
                last = len(seq) % num
                return (seq[i * avg + min(i, last):(i + 1) * avg + min(i + 1, last)] for i in range(num))

            # divide to smaller batches
            num_chunks = (len(detections) + 149) // 150  # Calculate the number of chunks
            detections_chunks = list(chunker(detections, num_chunks))
            detections_segment=[]
            for detection_chunk in detections_chunks:
                detections_segment_chunk = segment(image, detection_chunk, polygon_refinement, segmenter_id, segmentator, processor, pred_iou_thresh_sam, mask_threshold_sam)
                detections_segment.extend(detections_segment_chunk)
        else:
            detections_segment = segment(image, detections, polygon_refinement, segmenter_id, segmentator, processor, pred_iou_thresh_sam, mask_threshold_sam)
        # print the labels that for which there was bbox but no mask detected
        detected_labels = set(d.label.strip(" .") for d in detections)
        detections_segment_labels = set(d.label.strip(" .") for d in detections_segment)
        # TODO: name_label
        for label in detected_labels:
            if label not in detections_segment_labels:
                print(f"No detection for: {label}")
                labels_to_delete.append(label)
    else:
        detections_segment = None
 
    return np.array(image), detections, detections_segment, similar_prompt_map, similar_prompt_list, labels_to_delete

def union_bbox(bbox1, bbox2):
    if isinstance(bbox1, list) or isinstance(bbox2, list):
        bbox1 = bbox1[0]
        bbox2 = bbox2[0]
        xmin=min(bbox1["xmin"], bbox2["xmin"])
        ymin=min(bbox1["ymin"], bbox2["ymin"])
        xmax=max(bbox1["xmax"], bbox2["xmax"])
        ymax=max(bbox1["ymax"], bbox2["ymax"])
    else:
        xmin=min(bbox1.xmin, bbox2.xmin)
        ymin=min(bbox1.ymin, bbox2.ymin)
        xmax=max(bbox1.xmax, bbox2.xmax)
        ymax=max(bbox1.ymax, bbox2.ymax)
    return (xmin, ymin, xmax, ymax)

def union_of_masks(masks, mask_output_path=None, return_img=False, output_dir=""):
    if not isinstance(masks[0], np.ndarray):
        # masks is a list of png
        masks = [np.array(Image.open(mask)) if os.path.isfile(mask) else np.array(Image.open(os.path.join(output_dir, mask))) for mask in masks]
        return_img = True
    union_mask = np.sum(masks, axis=0).astype(np.uint8)
    # trim values to 1 to get the union mask
    union_mask = np.clip(union_mask, 0, 255)
    if return_img and mask_output_path is not None:
        mask_image = Image.fromarray(union_mask)
        if not os.path.exists(mask_output_path):
            mask_output_path = os.path.join(output_dir, mask_output_path)
        mask_image.save(mask_output_path)
    else:
        return union_mask

def union_detection_results(detections, bbox_union=False, mask_union=False, similar_prompt_list=[]):
    bbox_dict = {}
    mask_dict = {}
    for detection in detections:
        label = detection.label.rstrip(' .')
        box = detection.box
        if hasattr(detection, 'mask'):
            mask = detection.mask

        if label in bbox_dict:
            if bbox_union:
                bbox_dict[label] = [union_bbox(bbox_dict[label],box)]
            else:
                bbox_dict[label].append(box)
        else:
            bbox_dict[label] = [box]
        
        if hasattr(detection, 'mask'):
            if label in mask_dict:
                if mask_union:
                    mask_dict[label].append(mask)
                    mask_dict[label] = [union_of_masks(mask_dict[label])]
                else:
                    mask_dict[label].append(mask)
            else:
                mask_dict[label] = [mask]
    # go over the labels that are mapped to other mask labels
    if len(similar_prompt_list)>0: 
        # {original:,mapped_to_index:}
        for map_dict in similar_prompt_list:
            original_label = map_dict["original"]
            mapped_bbox = detections[map_dict["mapped_to_index"]].box
            if hasattr(detection, 'mask'):
                mapped_mask = detections[map_dict["mapped_to_index"]].mask
            if original_label in bbox_dict: 
                if bbox_union:
                    bbox_dict[original_label] = [union_bbox(bbox_dict[original_label], mapped_bbox)]
                else:
                    bbox_dict[original_label].append(mapped_bbox)
            else:
                bbox_dict[original_label] = [mapped_bbox]
            if hasattr(detection, 'mask'):
                if original_label in mask_dict:
                    mask_dict[original_label].append(mapped_mask)
                    mask_dict[original_label] = [union_of_masks(mask_dict[original_label])]
                else:
                    mask_dict[original_label] = [mapped_mask]
    return bbox_dict, mask_dict

def create_mask_dir_path(image_path_local, output_dir, image_id, post_segments_dir=""):
    image_dir_local, image_file = os.path.split(image_path_local)
    dirs = image_dir_local.split(os.sep)
    # add _segmented to the end of the image_dir_local
    dirs[0] += post_segments_dir 
    mask_dir = os.sep.join(dirs)
    if not os.path.exists(os.path.join(output_dir, mask_dir)):
        os.makedirs(os.path.join(output_dir, mask_dir))
    mask_img_dir = os.path.join(output_dir, mask_dir, image_id)
    if not os.path.exists(mask_img_dir):
        os.makedirs(mask_img_dir)
    return mask_img_dir

def save_mask2vec(mask_array, image_id, mask_num, mask_img_dir):
    mask_vec = reshape_mask_to_mask_vector(mask_array, num_patches_y = 24, num_patches_x = 24)
    mask_vec_file = f"{image_id}_{str(mask_num).zfill(4)}.pt"
    mask_output_path = os.path.join(mask_img_dir, mask_vec_file)
    torch.save(mask_vec, mask_output_path)
    return mask_output_path

def save_mask2img(mask_array, image_id, mask_num, mask_img_dir):
    mask_image = Image.fromarray(mask_array) 
    mask_file = f"{image_id}_{str(mask_num).zfill(4)}.png"
    mask_output_path = os.path.join(mask_img_dir, mask_file)
    mask_image.save(mask_output_path)
    return mask_output_path

def add_path2dict(masks_info, key, mask_output_path, n_response, i, account4realtions=False, output_dir=""):
    if not account4realtions and i < n_response and "_" in key:
        masks_info[f"a{i+1}"] = os.path.relpath(mask_output_path, output_dir)
    else:
        if isinstance(key, tuple):
            for k in key:
                if isinstance(k, str):
                    k=k.rstrip(' .')
            masks_info[key] = os.path.relpath(mask_output_path, output_dir)
        elif isinstance(key, str):
            masks_info[key.rstrip(' .')] = os.path.relpath(mask_output_path, output_dir)
    return masks_info

def save_masks(detections, output_dir, image_id, mask_img_dir, n_response = 1, account4realtions=False):
    #note: the masks should be computed on the processed image- the image should not be further processed within llava
    masks_info = {}
    array_info = {}
    masks_vec_info = {}
    j=0
    mask_num=0
    if isinstance(detections, dict):
        for key in detections:
            for mask_info in detections[key]:
                mask_output_path_vec = save_mask2vec(mask_info, image_id, mask_num, mask_img_dir)
                mask_output_path = save_mask2img(mask_info, image_id, mask_num,mask_img_dir)
                masks_info = add_path2dict(masks_info, key, mask_output_path, n_response, j, account4realtions=account4realtions,output_dir=output_dir)
                masks_vec_info = add_path2dict(masks_vec_info, key, mask_output_path_vec, n_response, j, account4realtions=account4realtions,output_dir=output_dir)
                mask_num += 1
            j +=1
    return masks_info,masks_vec_info, array_info

def union_bbox_masks_for_label_name(bbox_dict, mask_dict, prompts_with_turns):
    # dictionaries to store the updated bounding boxes and masks
    new_bbox_dict = defaultdict(list)
    new_mask_dict = defaultdict(list)
    for entry in prompts_with_turns:
        turn = entry['turn']
        prompt = entry['prompt']
        prompt_name = entry['prompt_name']

        # update the labels in bbox_dict and mask_dict
        if prompt in bbox_dict:
            new_bbox_dict[(turn, prompt_name)].extend(bbox_dict[prompt])
        if prompt in mask_dict:
            new_mask_dict[(turn, prompt_name)].extend(mask_dict[prompt])

    for key, masks in new_mask_dict.items():
        turn, prompt_name = key
        if len(masks) > 1:
            new_mask_dict[(turn, prompt_name)] = [union_of_masks(masks)]
    return dict(new_bbox_dict), dict(new_mask_dict)

def output_json_path(samples_json):
    # save the updated json
    dir_path = os.path.dirname(samples_json)
    file_name = os.path.basename(samples_json)
    # if splits in name of dir and json file
    if 'splits' in dir_path and 'split' in file_name:
        new_dir_path = dir_path.replace('splits', 'grounded')
        new_file_name = file_name.replace('split', 'grounded_split')
    else:
        new_dir_path = os.path.join(dir_path, 'grounded')
        new_file_name = file_name.replace('.json', '_grounded.json')
    if not os.path.exists(new_dir_path):
        os.makedirs(new_dir_path)
    samples_json_new = os.path.join(new_dir_path, new_file_name)
    return samples_json_new

def load_json(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    file_extension = os.path.splitext(file_path)[1].lower()

    if file_extension == '.json':
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
    elif file_extension == '.jsonl':
        data = []
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                data.append(json.loads(line))
    else:
        raise ValueError(f"Unsupported file extension {file_extension}. Supported extensions are .json and .jsonl.")

    return data