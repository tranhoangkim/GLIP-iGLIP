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
import json

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

config_file = "configs/pretrain/glip_Swin_L.yaml" #"configs/pretrain/glip_Swin_T_O365_GoldG.yaml"
weight_file = "MODEL/glip_large_model.pth"

category = {
    'player': 0, 
    'red card': 1,
    'yellow card': 2,
    'goal': 3
}

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

def get_object_in_image(glip_demo, frame_path, caption, threshold, is_card=True):
    img = load_jpg(frame_path)
    # height, width, _ = img.shape
    # img = cv2.resize(img, dsize=(width, height), interpolation=cv2.INTER_CUBIC)
    # print(img.shape)
    result, pred_boxes, proposal_visual_features, proposal_fused_visual_features, new_labels = glip_demo.run_on_web_image(img, caption, threshold)
    assert pred_boxes.bbox.shape[0] == proposal_visual_features.shape[0]
    assert proposal_visual_features.shape == proposal_fused_visual_features.shape
    # print('proposal_visual_features', proposal_visual_features.shape)
    # before post-proc 
    # cv2.imwrite(pre_path, result)
    
    # after post-proc
    nms_output = nms(boxes=pred_boxes.bbox, scores=pred_boxes.get_field("scores"), iou_threshold=0.7)
    bboxes = pred_boxes.bbox[nms_output]
    scores = pred_boxes.get_field("scores")[nms_output]
    proposal_visual_features = proposal_visual_features[nms_output]
    proposal_fused_visual_features = proposal_fused_visual_features[nms_output]

    # img = cv2.imread(frame_path)
    keep = []
    labels = []
    for box_id in range(bboxes.shape[0]):
        x1, y1, x2, y2 = bboxes[box_id] 
        w = x2 - x1 + 1
        h = y2 - y1 + 1 
        if is_card:
            if w * h > 2000: 
                continue
            if w / h > 3 or h / w > 3:
                continue
        # img = draw_box(img, bboxes[box_id])
        # cv2.putText(
        #         img, str(scores[box_id]), (int(x1), int(y1)-4), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 255, 0), 3, cv2.LINE_AA
        #     )
        keep.append(box_id)
        labels.append(new_labels[nms_output[box_id]])
    
    # print(post_path, img.shape)
    # cv2.imwrite(post_path, img)
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

    dataset_dir = '/cm/shared/kimth1/SoccerNet_frames_720p'
    save_dir = 'SoccerNet_frames_GLIP_features'

    for league in os.listdir(dataset_dir):
        league_dir = osp.join(dataset_dir, league)
        for season in os.listdir(league_dir):
            season_dir = osp.join(league_dir, season)
            for game in os.listdir(season_dir):
                game_dir = osp.join(season_dir, game)
                game_feat_dir = game_dir.replace(dataset_dir, save_dir)
                os.makedirs(game_feat_dir, exist_ok=True)
                for video_file in os.listdir(game_dir):
                    video_dir = osp.join(game_dir, video_file)
                    feat_vid = torch.tensor([]).cuda() # [frame_id, class_id, box, feature_vector]
                    for frame_file in os.listdir(video_dir):
                        if frame_file.endswith('.jpg') == False:
                            continue
                        feat = torch.tensor([]).cuda()
                        scores = torch.tensor([])
                        frame_path = osp.join(video_dir, frame_file)
                        # if frame_path not in event_type:
                        #     continue
                        # import pdb;pdb.set_trace()
                        frame_id = float(frame_file.split('.')[0])

                        actor_box, actor_feat, actor_score, _ = get_object_in_image(glip_demo, frame_path, 'player', 0.4, False)
                        actor_feat = torch.cat((actor_box.cuda(), actor_feat), axis=1)
                        feat = torch.cat((feat, 
                                        torch.cat((torch.tensor([frame_id, category['player']]).cuda().repeat(actor_feat.shape[0], 1), actor_feat), axis=1)),
                                        axis=0)
                        scores = torch.cat((scores, actor_score), axis=0)
                        assert len(feat) == len(scores)
                        
                        redcard_box, redcard_feat, redcard_score, _ = get_object_in_image(glip_demo, frame_path, 'red card', 0.45, True)
                        redcard_feat = torch.cat((redcard_box.cuda(), redcard_feat), axis=1)
                        feat = torch.cat((feat, 
                                        torch.cat((torch.tensor([frame_id, category['red card']]).cuda().repeat(redcard_feat.shape[0], 1), redcard_feat), axis=1)),
                                        axis=0)
                        scores = torch.cat((scores, redcard_score), axis=0)
                        assert len(feat) == len(scores)

                        yellowcard_box, yellowcard_feat, yellowcard_score, _ = get_object_in_image(glip_demo, frame_path, 'yellow card', 0.5, True)
                        yellowcard_feat = torch.cat((yellowcard_box.cuda(), yellowcard_feat), axis=1)
                        feat = torch.cat((feat, 
                                        torch.cat((torch.tensor([frame_id, category['yellow card']]).cuda().repeat(yellowcard_feat.shape[0], 1), yellowcard_feat), axis=1)),
                                        axis=0)
                        scores = torch.cat((scores, yellowcard_score), axis=0)       
                        assert len(feat) == len(scores)         

                        goal_box, goal_feat, goal_score, _ = get_object_in_image(glip_demo, frame_path, 'goal', 0.6, False)
                        goal_feat = torch.cat((goal_box.cuda(), goal_feat), axis=1)
                        feat = torch.cat((feat, 
                                        torch.cat((torch.tensor([frame_id, category['goal']]).cuda().repeat(goal_feat.shape[0], 1), goal_feat), axis=1)),
                                        axis=0)
                        scores = torch.cat((scores, goal_score), axis=0)
                        assert len(feat) == len(scores)
                        # if len(goal_box) != 0:
                        #     # import pdb;pdb.set_trace()
                        #     print('detect goal', goal_box.shape)
                        
                        # _, flag_feat, _, _ = get_object_in_image(glip_demo, frame_path, 'flag', 0.4, True)
                        # feat = torch.cat((feat, 
                        #                 torch.cat((torch.tensor([frame_id, 0]).repeat(flag_feat.shape[0], 1), flag_feat), axis=1)),
                        #                 axis=0)
                        try:
                            nms_output = nms(boxes=feat[:, 2:6], scores=scores.cuda(), iou_threshold=0.9)
                        except:
                            print(f'Error: {frame_path}')
                            import pdb;pdb.set_trace()
                        feat_vid = torch.cat((feat_vid, feat[nms_output]), axis=0)
                        # import pdb;pdb.set_trace()
                    
                    features_file = osp.join(game_feat_dir, f'{video_file}.pt')
                    torch.save(feat_vid, features_file)                                                 

if __name__ == '__main__':
    main()
