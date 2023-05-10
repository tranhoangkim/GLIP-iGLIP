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
from torch import nn
import statistics

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
import json

config_file = "configs/pretrain/glip_Swin_L.yaml" #"configs/pretrain/glip_Swin_T_O365_GoldG.yaml"
weight_file = "MODEL/glip_large_model.pth"
dataset_path = '/cm/shared/kimth1/SoccerNet_frames_720p'

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

def get_object_in_image(glip_demo, frame_path, caption, threshold, pre_path, post_path, is_card=True):
    # file_name = frame_path.split('/')[-1]
    # if file_name != '000275.jpg':
    #     return None, None, None, None
    img = load_jpg(frame_path)
    # height, width, _ = img.shape
    # img = cv2.resize(img, dsize=(width, height), interpolation=cv2.INTER_CUBIC)
    # print(img.shape)
    result, pred_boxes, proposal_visual_features, proposal_fused_visual_features, new_labels = glip_demo.run_on_web_image(img, caption, threshold)
    assert pred_boxes.bbox.shape[0] == proposal_visual_features.shape[0]
    assert proposal_visual_features.shape == proposal_fused_visual_features.shape
    print('proposal_visual_features', proposal_visual_features.shape)
    # before post-proc 
    cv2.imwrite(pre_path, result)
    if is_card == True:
        iou_threshold = 0.1
    else:
        iou_threshold = 0.5
    # after post-proc 
    nms_output = nms(boxes=pred_boxes.bbox, scores=pred_boxes.get_field("scores"), 
                    iou_threshold=iou_threshold)
    bboxes = pred_boxes.bbox[nms_output]
    scores = pred_boxes.get_field("scores")[nms_output]
    proposal_visual_features = proposal_visual_features[nms_output]
    proposal_fused_visual_features = proposal_fused_visual_features[nms_output]
    # new_labels = new_labels[nms_output]

    img = cv2.imread(frame_path)
    keep = []
    for box_id in range(bboxes.shape[0]):
        x1, y1, x2, y2 = bboxes[box_id] 
        w = x2 - x1 + 1
        h = y2 - y1 + 1 
        if is_card:
            if w * h > 1900:
                # print('helloo')
                continue
            if w / h > 1.5 or h / w > 3:
                # print('yeah')
                continue
        img = draw_box(img, bboxes[box_id])
        score = round(float(scores[box_id]), 2)
        label = new_labels[nms_output[box_id]]
        cv2.putText(
                img, f'{label}:{score}', (int(x1), int(y1+h+10)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1, cv2.LINE_AA
            )
        keep.append(box_id)
    
    # print(post_path, img.shape)
    cv2.imwrite(post_path, img)
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
    split = 'test'
    with open(f'/cm/shared/kimth1/spot/data/soccernetv2_p1/{split}.json') as f:
        data = json.load(f)

    red_card = []
    yellow_card = []
    yellow_red_card = []
    offside = []
    goal = []
    for video in data:
        for event in video['events']:
            if event['label'] == 'Red card':
                for i in range(-5,10):
                    if event['frame']+i > 0:
                        red_card.append(osp.join(video['video'], f"{event['frame']+i:06}.jpg"))
            if event['label'] == 'Yellow card':
                for i in range(-5,10):
                    if event['frame']+i > 0:
                        yellow_card.append(osp.join(video['video'], f"{event['frame']+i:06}.jpg"))
            if event['label'] == 'Yellow->red card':
                for i in range(-5,10):
                    if event['frame']+i > 0:
                        yellow_red_card.append(osp.join(video['video'], f"{event['frame']+i:06}.jpg"))
            if event['label'] == 'Offside':
                for i in range(-5,10):
                    if event['frame']+i > 0:
                        offside.append(osp.join(video['video'], f"{event['frame']+i:06}.jpg"))
            if event['label'] == 'Goal':
                for i in range(-5,10):
                    if event['frame']+i > 0:
                        goal.append(osp.join(video['video'], f"{event['frame']+i:06}.jpg"))
                
    # caption = 'offside flag' # red card 0.5 yellow card 0.6
    # threshold = 0.3
    # offside_pre_folder = './offside/pre'
    # offside_post_folder = './offside/post'
    # os.makedirs(offside_pre_folder, exist_ok=True)
    # os.makedirs(offside_post_folder, exist_ok=True)
    # for img_path in offside:
    #     img_path = osp.join(dataset_path, img_path)
    #     name = img_path.split('/')[-1]
    #     pre_path = osp.join(offside_pre_folder, name)
    #     post_path = osp.join(offside_post_folder, name)
    #     visual_bboxes, visual_features, scores, _ = get_object_in_image(glip_demo, img_path, caption, threshold, pre_path, post_path, False)      

    # caption = 'yellow card' # red card 0.5 yellow card 0.6
    # threshold = 0.45
    # yellow_card_pre_folder = './yellow_card_val_tmp/pre'
    # yellow_card_post_folder = './yellow_card_val_tmp/post'
    # os.makedirs(yellow_card_pre_folder, exist_ok=True)
    # os.makedirs(yellow_card_post_folder, exist_ok=True) 
    # for img_path in yellow_card:
    #     img_path = osp.join(dataset_path, img_path)
    #     name = img_path.split('/')[-1]
    #     pre_path = osp.join(yellow_card_pre_folder, name)
    #     post_path = osp.join(yellow_card_post_folder, name)
    #     visual_bboxes, visual_features, scores, _ = get_object_in_image(glip_demo, img_path, caption, threshold, pre_path, post_path, True)      

    # caption = 'red card' # red card 0.5 yellow card 0.6
    # threshold = 0.5
    # red_card_pre_folder = './red_card_test/pre'
    # red_card_post_folder = './red_card_test/post'
    # os.makedirs(red_card_pre_folder, exist_ok=True)
    # os.makedirs(red_card_post_folder, exist_ok=True)
    # for img_path in red_card:
    #     img_path = osp.join(dataset_path, img_path)
    #     name = img_path.split('/')[-1]
    #     pre_path = osp.join(red_card_pre_folder, name)
    #     post_path = osp.join(red_card_post_folder, name)
    #     visual_bboxes, visual_features, scores, _ = get_object_in_image(glip_demo, img_path, caption, threshold, pre_path, post_path, True)      

    # caption = 'red card . yellow card' # red card 0.5 yellow card 0.6
    # threshold = 0.5
    # pre_folder = f'./yellow_card_{split}/pre'
    # post_folder = f'./yellow_card_{split}/post'
    # os.makedirs(pre_folder, exist_ok=True)
    # os.makedirs(post_folder, exist_ok=True)
    
    # for img_path in yellow_card:
    #     img_path = osp.join(dataset_path, img_path)
    #     name = img_path.split('/')[-1]
    #     pre_path = osp.join(pre_folder, name)
    #     post_path = osp.join(post_folder, name)
    #     # if name != '000815.jpg': continue
    #     visual_bboxes, visual_features, scores, _ = get_object_in_image(glip_demo, img_path, caption, threshold, pre_path, post_path, True)      
    #     # import pdb; pdb.set_trace()

    # caption = 'goal' # soccer 0.5, red card 0.5, yellow card 0.5, goal 0.65
    # threshold = 0.7
    # goal_pre_folder = './goal/pre'
    # goal_post_folder = './goal/post'
    # os.makedirs(goal_pre_folder, exist_ok=True)
    # os.makedirs(goal_post_folder, exist_ok=True)
    # for img_path in goal:
    #     img_path = osp.join(dataset_path, img_path)
    #     name = img_path.split('/')[-1]
    #     pre_path = osp.join(goal_pre_folder, name)
    #     post_path = osp.join(goal_post_folder, name)
    #     visual_bboxes, visual_features, scores, _ = get_object_in_image(glip_demo, img_path, caption, threshold, pre_path, post_path, False)      
    
    caption = 'player' # red card 0.5 yellow card 0.6
    threshold = 0.5
    pre_folder = f'./player_{split}/pre'
    post_folder = f'./player_{split}/post'
    os.makedirs(pre_folder, exist_ok=True)
    os.makedirs(post_folder, exist_ok=True)
    
    player_dict = []
    raise_hand = []
    raise_hand.extend(red_card)
    raise_hand.extend(yellow_card)
    raise_hand.extend(yellow_red_card)
    assert len(raise_hand) == len(red_card) + len(yellow_red_card) + len(yellow_card)
    for img_path in raise_hand:
        img_path = osp.join(dataset_path, img_path)
        name = img_path.split('/')[-1]
        pre_path = osp.join(pre_folder, name)
        post_path = osp.join(post_folder, name)
        visual_bboxes, visual_features, scores, _ = get_object_in_image(glip_demo, img_path, caption, threshold, pre_path, post_path, False)      
        vid_info = {}
        vid_info['frame_path'] = img_path
        vid_info['boxes'] = visual_bboxes.tolist()
        vid_info['features'] = visual_features.tolist()
        player_dict.append(vid_info)
 
    # import pdb;pdb.set_trace()
    player_dict = json.dumps(player_dict)
    # import pdb;pdb.set_trace()
    with open(f"./player_{split}/player_{split}.json", "w") as outfile:
        outfile.write(player_dict)

if __name__ == '__main__':
    main()
