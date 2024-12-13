"""
instance segmentation image with sam and clip with text prompts
"""
import json
import os
import os.path as ops
import argparse
from itertools import chain
import numpy as np
import cv2
from PIL import Image
from datasets import load_dataset

from sam_clip.models import build_sam_clip_text_ins_segmentor
from sam_clip.utils import parse_config


DATASET_PATH = "output/train"

def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--insseg_cfg_path', type=str, default='./config/insseg.yaml')
    parser.add_argument('--text', type=str, default=None)
    parser.add_argument('--cls_score_thresh', type=float, default=None)
    parser.add_argument('--save_dir', type=str, default='./output/insseg')
    parser.add_argument('--save_interval', type=int, default=9)
    parser.add_argument('--use_text_prefix', action='store_true')

    return parser.parse_args()



def seg_image_in_parts(input_image, segmentor, unique_label=None, use_text_prefix=False):
    """
     Divides the input image into 4 parts, processes each part separately using seg_image,
     and returns the results for each part.

    :param input_image:
    :param segmentor:
    :param unique_label:
    :param use_text_prefix:
    :return:
    """

    # Convert PIL image to numpy array
    input_image_np = np.array(input_image)
    h, w, c = input_image_np.shape

    # Define coordinates for dividing the image into 4 parts
    parts_coords = [
        (0, 0, w // 2, h // 2),  # Top-left
        (w // 2, 0, w, h // 2),  # Top-right
        (0, h // 2, w // 2, h),  # Bottom-left
        (w // 2, h // 2, w, h)  # Bottom-right
    ]

    # Results storage
    results = []
    part_images = []
    masks_results = []

    for x1, y1, x2, y2 in parts_coords:
        # Crop part of the image
        part = input_image_np[y1:y2, x1:x2, :]
        part_image = Image.fromarray(part)  # Convert back to PIL Image for compatibility

        # merge results of all classes
        part_results, masks_dicts = {}, {}
        for label in unique_label:
            part_result, masks = segmentor.seg_image(part_image, unique_label=[label], use_text_prefix=use_text_prefix)

            part_results = {key: np.vstack([part_result[key], part_results[key]]) for key in part_result.keys() if
                            key != "source"} if part_result.keys() == part_results.keys() else part_result
            masks_dicts = {key: np.vstack([masks[key], masks_dicts[key]]) for key in
                           masks.keys()} if masks.keys() == masks_dicts.keys() else masks
        part_results.update({'source': part_result['source']})
        masks_dicts.update({'bbox_cls_names': list(chain.from_iterable(masks_dicts['bbox_cls_names'])),
                            'scores': list(chain.from_iterable(masks_dicts['scores']))})

        # Add to results
        results.append(part_results)
        masks_results.append(masks_dicts)
        part_images.append(part_image)

    return results, masks_results, part_images


def post_process(results, masks_results, images_results, i, save_dir, save_interval):
    """
    Process mask regions and only save masks with labels among the prompted ones

    :param results:
    :param masks_results:
    :param i:
    :param save_dir:
    :param save_interval:
    :return:
    """
    dataset_dict = []
    if i == 0:
        start_index = -1
    else:
        start_index = (i - save_interval + 1) * 4 - 1
    for _, (ret, mask, image) in enumerate(zip(results, masks_results, images_results)):
        start_index += 1
        seg_images = mask['segmentations']
        bboxes = mask['bboxes']
        bboxes_names = mask['bbox_cls_names']
        scores = mask['scores']
        input_image_name = str(start_index)

        seg_images_list = []
        bboxes_list = []
        labels_list = []
        scores_list = []
        for idx, _ in enumerate(bboxes):
            # Discard background segments and very large segments
            if bboxes_names[idx] == 'background' or (bboxes[idx][2] > 600 or bboxes[idx][3] > 600):
                continue
            seg_images_list.append(seg_images[idx])

            # Change the format of b-boxes to (x_min, y_min, x_max, y_max)
            bboxes_list.append([int(bboxes[idx][0]), int(bboxes[idx][1]), int(bboxes[idx][0] + bboxes[idx][2]),
                                int(bboxes[idx][1] + bboxes[idx][3])])
            labels_list.append(bboxes_names[idx])
            scores_list.append(scores[idx])

        # Discard the masks that are shared between the labels only keeping the one with the highest classification score
        result = {box: (len(indices := [i for i, b in enumerate(bboxes_list) if tuple(b) == box]), indices) for box
                  in {tuple(b) for b in bboxes_list}}
        filtered_boxes, filtered_labels, filtered_seg_images, filtered_scores = [], [], [], []
        for box, (_, indexes) in result.items():
            best_indexes = indexes[np.argmax([scores_list[i] for i in indexes])]
            filtered_boxes.append(bboxes_list[best_indexes])
            filtered_labels.append(labels_list[best_indexes])
            filtered_seg_images.append(seg_images_list[best_indexes])
            filtered_scores.append(scores_list[best_indexes])

        file_name = input_image_name + '.jpg'
        dataset_dict.append({"file_name": file_name, "segmentations": filtered_seg_images, "bboxes": filtered_boxes,
                             'labels': filtered_labels})
        # Save images
        # original_image_path = os.path.join(DATASET_PATH, file_name)
        # image.save(original_image_path)
        mask_add_save_path = ops.join(save_dir, '{:s}_insseg_add.png'.format(input_image_name))
        cv2.imwrite(mask_add_save_path, ret['ins_seg_add'])
    return dataset_dict


def main():
    """

    :return:
    """
    # init args
    args = init_args()

    # Loading the unannotated dataset of images
    ds = load_dataset("gotdairyya/plant_images")
    print("Dataset loaded!")

    save_dir = args.save_dir
    saving_interval = args.save_interval
    os.makedirs(save_dir, exist_ok=True)
    insseg_cfg_path = args.insseg_cfg_path

    insseg_cfg = parse_config.Config(config_path=insseg_cfg_path)
    if args.text is not None:
        unique_labels = args.text.split(',')
    else:
        unique_labels = None
    if args.cls_score_thresh is not None:
        insseg_cfg.INS_SEG.CLS_SCORE_THRESH = args.cls_score_thresh
    use_text_prefix = True if args.use_text_prefix else False

    print('Start initializing instance segmentor ...')
    segmentor = build_sam_clip_text_ins_segmentor(cfg=insseg_cfg)
    print('Segmentor initialized complete')

    # For better segmentation, we break each image in 4 new smaller ones and return masks and labels for all images
    results, masks_results, images_results = [], [], []
    dataset = ds['train']['image']
    print('Start to segment input images ...')
    for i, test_image in enumerate(dataset):
        ret, masks, images = seg_image_in_parts(test_image, segmentor, unique_label=unique_labels, use_text_prefix=use_text_prefix)
        print(f'segment complete on images {i * 4} - {i * 4 + 3} ')
        results.extend(ret)
        masks_results.extend(masks)
        images_results.extend(images)

        # Just tp be safe, save annotations to file every `saving_interval` steps
        if i % saving_interval == 0 or i == len(dataset) - 1:
            if i == len(dataset) - 1:
                i = len(dataset)
            dataset_dict = post_process(results, masks_results, images_results, i, save_dir, saving_interval)
            results, masks_results, images_results = [], [], []

            # Save the dataset metadata to json file
            with open(DATASET_PATH + "/metadata.jsonl", "a") as outfile:
                for data in dataset_dict:
                    file, bbox, label = data['file_name'], data["bboxes"], data["labels"]
                    entry = {"file_name": file, "bboxes": bbox, 'labels': label}
                    print(json.dumps(entry), file=outfile)

            print('Saved segments and labels result into metadata.jsonl and plots of annotated images into {:s}'.format(save_dir))

    # Save the dataset to Huggingface dataset repository "sarakarimi30/PlantMap"
    dataset = load_dataset("imagefolder", data_dir=DATASET_PATH)
    dataset.push_to_hub("PlantMap", private=False)
    return


if __name__ == '__main__':
    """
    main func
    """
    main()
