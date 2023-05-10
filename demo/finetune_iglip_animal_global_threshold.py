# from genericpath import isfile
import os 
import os.path as osp
import sys
sys.path.append('/home/kimth1/GLIP')
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import cv2 
from tqdm import tqdm
import torch
torch.set_num_threads(20)

# import requests
from PIL import Image
import numpy as np
pylab.rcParams['figure.figsize'] = 20, 12
import sys
sys.path.append('/cm/shared/kimth1/GLIP')
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.engine.predictor_glip import GLIPDemo
from torchvision.ops import nms

# nohup python demo/finetune_iglip_animal_global_threshold.py > iglip_animal_global_threshold.out &
high_threshold = 0.5
mid_threshold = 0.35
low_threshold = 0.3
sim_threshold = 0.85

DATA_DIR = '/cm/shared/kimth1/Tracking/AnimalTrack_data/test'
SAVE_DIR = './iGLIP_AnimalTrack_new/vis_before'
DROP_SAVE_DIR = './iGLIP_AnimalTrack_new/vis_after'
BOX_SAVE_DIR = './iGLIP_AnimalTrack_new/Animal_boxes'
FINAL_VIS_DIR = './iGLIP_AnimalTrack_new/vis_final'
os.makedirs(DROP_SAVE_DIR, exist_ok=True)
os.makedirs(BOX_SAVE_DIR, exist_ok=True)

config_file = "configs/pretrain/glip_Swin_L.yaml" #"configs/pretrain/glip_Swin_T_O365_GoldG.yaml"
weight_file = "MODEL/glip_large_model.pth"

def load_jpg(frame_path):
    pil_image = Image.open(frame_path).convert("RGB")
    # convert to BGR format
    image = np.array(pil_image)[:, :, [2, 1, 0]]
    return image

def imshow(img, caption, save_path: str):
    plt.imshow(img[:, :, [2, 1, 0]])
    plt.axis("off")
    plt.figtext(0.5, 0.09, caption, wrap=True, horizontalalignment='center', fontsize=20)
    plt.savefig(save_path)

def draw_box(img, bbox):
    # bbox is mode xyxy
    start_point = (int(bbox[0]), int(bbox[1]))
    end_point = (int(bbox[2]), int(bbox[3]))
    color = (255, 0, 0)
    thickness = 2
    return cv2.rectangle(img, start_point, end_point, color, thickness)

# CAPTION_MAP for high visual feature, CAPTION_MAP2 for low visual feature

CAPTION_MAP = {
    "chicken_2": "chick",
    'goose_1': 'white duck'
}

CAPTION_MAP2 = {
    "chicken_2": "chick",
    'goose_1': 'white duck'
}

specific_videos = ['chicken_2', 'duck_3', 'goose_3']
# ['chicken_2', 'dolphin_1', 'duck_3', 'goose_1', 'goose_2', 'goose_3', 'penguin_1', 'penguin_2', 'penguin_3']
run_vids = ['chicken_2', 'goose_3']# ['chicken_2', 'dolphin_1', 'duck_3', 'goose_1', 'goose_3']

def get_object_in_image(glip_demo, caption, threshold, frame_name, img_folder, save_vis_before, save_vis_after, iou, post_process=False):
    frame_path = osp.join(img_folder, frame_name)
    save_path = osp.join(save_vis_before, frame_name)
    img = load_jpg(frame_path)

    result, pred_boxes, proposal_visual_features, proposal_fused_visual_features = glip_demo.run_on_web_image(img, caption, threshold)
    assert pred_boxes.bbox.shape[0] == proposal_visual_features.shape[0]
    assert proposal_visual_features.shape == proposal_fused_visual_features.shape

    # before post-proc 
    cv2.imwrite(save_path, result)
    
    # after post-proc
    nms_output = nms(boxes=pred_boxes.bbox, scores=pred_boxes.get_field("scores"), iou_threshold=iou)
    bboxes = pred_boxes.bbox[nms_output]
    scores = pred_boxes.get_field("scores")[nms_output]
    proposal_visual_features = proposal_visual_features[nms_output]
    proposal_fused_visual_features = proposal_fused_visual_features[nms_output]

    img = cv2.imread(frame_path)

    keep = []
    for box_id in range(bboxes.shape[0]):
        xo1, yo1, xo2, yo2 = bboxes[box_id] 
        cnt = 0
        for box in bboxes:
            xi1, yi1, xi2, yi2 = box
            if cnt > 3:
                break
            if xi1 >= xo1 and yi1 >= yo1 and xi2 <= xo2 and yi2 <= yo2:
                cnt += 1
        if cnt <= 3 or post_process == False:
            img = draw_box(img, bboxes[box_id])
            keep.append(box_id)
    
    drop_save_path = osp.join(save_vis_after, frame_name)
    cv2.imwrite(drop_save_path, img)
    assert bboxes.shape[0] == proposal_visual_features.shape[0] and bboxes.shape[0] == scores.shape[0]
    assert proposal_visual_features.shape == proposal_fused_visual_features.shape
    return bboxes[keep], proposal_visual_features[keep], scores[keep], proposal_fused_visual_features[keep]

def main():
    cfg.local_rank = 0
    cfg.num_gpus = 1
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(["MODEL.WEIGHT", weight_file])
    cfg.merge_from_list(["MODEL.DEVICE", "cuda"])
    
    cfg.MODEL.ROI_HEADS.NMS = 0.7
    glip_demo = GLIPDemo(
        cfg,
        min_image_size=800,
        confidence_threshold=0.2,
        show_mask_heatmaps=False
    )
    print('Create model successfully')

    for category in tqdm(os.listdir(DATA_DIR)):
        if category not in run_vids:
            continue

        print(f'>>> Run {category}')
        img_folder = osp.join(DATA_DIR, category, 'img1')
        list_frame_names = os.listdir(img_folder)
        list_frame_names.sort()  
        choice_fnames = list_frame_names #[list_frame_names[0]] #random.choices(list_frame_names, k=10)
        
        caption = CAPTION_MAP[category] if CAPTION_MAP.get(category) is not None \
                    else category.split('_')[0]
        threshold = high_threshold if category in specific_videos else mid_threshold
        print('caption, threshold', caption, threshold)
        
        caption2 = CAPTION_MAP2[category] if CAPTION_MAP2.get(category) is not None \
                    else category.split('_')[0]
        threshold2 = low_threshold
        print('caption2, threshold2', caption2, threshold2)

        save_vis_before = osp.join(SAVE_DIR, category + '-high')
        save_vis_after = osp.join(DROP_SAVE_DIR, category + '-high')
        save_vis_before2 = osp.join(SAVE_DIR, category + '-low')
        save_vis_after2 = osp.join(DROP_SAVE_DIR, category + '-low')
        save_vis_final = osp.join(FINAL_VIS_DIR, category)
        os.makedirs(save_vis_after, exist_ok=True)
        os.makedirs(save_vis_before, exist_ok=True)
        os.makedirs(save_vis_after2, exist_ok=True)
        os.makedirs(save_vis_before2, exist_ok=True)
        os.makedirs(save_vis_final, exist_ok=True)

        final_bboxes = []
        topk = None
        
        for frame_id, frame_name in enumerate(choice_fnames, 1):
            # get features
            high_visual_bboxes, high_visual_features, high_scores, _ = get_object_in_image(glip_demo, caption, threshold, frame_name, img_folder, save_vis_before, save_vis_after, 0.5, True) 
            if high_visual_bboxes.shape[0] == 0:
                high_visual_bboxes, high_visual_features, high_scores, _ = get_object_in_image(glip_demo, caption, threshold-0.05, frame_name, img_folder, save_vis_before, save_vis_after, 0.5, True)
            if caption != caption2 or threshold != threshold2:
                low_visual_bboxes, low_visual_features, low_scores, _ = get_object_in_image(glip_demo, caption2, threshold2, frame_name, img_folder, save_vis_before2, save_vis_after2, 0.65, False) 
            else:
                low_visual_bboxes, low_visual_features, low_scores = high_visual_bboxes, high_visual_features, high_scores
            
            # calculate similarities between high_visual and low_visual
            high_visual_features_norm = high_visual_features / high_visual_features.norm(dim=1)[:, None]
            low_visual_features_norm = low_visual_features / low_visual_features.norm(dim=1)[:, None]
            cosine_scores = torch.mm(high_visual_features_norm, low_visual_features_norm.transpose(0,1))
            if len(cosine_scores) != 0:
                cosine_scores, _ = torch.max(cosine_scores, dim=0)
            else:
                continue

            threshold3 = sim_threshold
            keep = torch.nonzero(cosine_scores > threshold3).squeeze(1)
            selected_low_visual_bboxes = low_visual_bboxes[keep]
            selected_scores = cosine_scores[keep]

            # visualize
            img = cv2.imread(osp.join(img_folder, frame_name))
            for id, bbox in enumerate(selected_low_visual_bboxes):
                img = draw_box(img, bbox)
                x1, y1, x2, y2 = bbox
                w = x2 - x1 + 1
                h = y2 - y1 + 1
                score = str(round(float(selected_scores[id]), 2))
                cv2.putText(
                img, score, (int(x1), int(y1)-4), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255))
                final_bboxes.append([frame_id, -1, x1, y1, w, h, 1, 1, 1])
                
            vis_final_path = osp.join(save_vis_final, frame_name)
            try:
                cv2.imwrite(vis_final_path, img)
            except:
                print('img is empty or wrong path')
                print('img', img.shape)
                print('vis_final_path', vis_final_path)

        np.savetxt(osp.join(BOX_SAVE_DIR, f'{category}.txt'), np.array(final_bboxes), fmt='%.6f', delimiter=',')

if __name__ == '__main__':
    main()
