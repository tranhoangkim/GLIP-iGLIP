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

# DATA_DIR = '/cm/archive/kimth1/chicken_data/test'
DATA_DIR = '/cm/archive/kimth1/MOT_data/GMOT_40/test'
# DATA_DIR = '/cm/archive/kimth1/MOT_data/MOT17/train'
# DATA_DIR = '/cm/archive/kimth1/MOT_data/GMOT_40/GenericMOT_JPEG_Sequence/'
SAVE_DIR = './results_swinT/vis_before'
DROP_SAVE_DIR = './results_swinT/vis_after'
BOX_SAVE_DIR = './results_swinT/GMOT_40_boxes'
os.makedirs(DROP_SAVE_DIR, exist_ok=True)
os.makedirs(BOX_SAVE_DIR, exist_ok=True)

config_file = "configs/pretrain/glip_Swin_T_O365_GoldG.yaml" # "configs/pretrain/glip_Swin_L.yaml" 
weight_file = "MODEL/glip_tiny_model_o365_goldg_cc_sbu.pth" # "MODEL/glip_large_model.pth"


def load_jpg(fpath):
    pil_image = Image.open(fpath).convert("RGB")
    # convert to BGR format
    image = np.array(pil_image)[:, :, [2, 1, 0]]
    return image


def imshow(img, caption, save_path: str):
    plt.imshow(img[:, :, [2, 1, 0]])
    plt.axis("off")
    plt.figtext(0.5, 0.09, caption, wrap=True, horizontalalignment='center', fontsize=20)
    plt.savefig(save_path)


FINETUNE_ONLY=True
CAPTION_MAP = {
    # 'airplane-0': 'black airplane. black helicopter', # 0.3
    # 'airplane-1': 'black airplane. black helicopter', # 0.3

    # 'airplane-2': 'black propeller plane. monoplane', # 0.2
    # 'balloon-1': 'red ballon. white ballon. red heart. white heart. green balloon. black balloon.', # 0.2
    # 'bird-0': 'black bird. black swallow', # 0.15
    # 'boat-2': 'white boat. white ship. cano', # 0.15
    
    # 'ball-0': 'red object . red ball',
    # 'ball-1': 'small green yellow ball. tennis ball',
    ##  'ball-2': 'green ball. tennis ball',
    'ball-3': 'tennis ball'
    
    # 'balloon-0': 'red balloon. blue balloon. orange balloon. purple balloon. pink balloon. green balloon. red ball. blue ball. orange ball. purple ball. green ball. pink balloon',
    #'balloon-1': 'Balloon. Colorful Balloon. Parachute. red heart. red balloon',
    # 'balloon-1': 'red heart. red balloon. balloon. red object. Colorful Balloon. Parachute. '
    # 'balloon-1': 'red balloon . red heart . Colorful Balloon'
    # 'balloon-1': 'Red heart. Red balloon. green balloon. black balloon'
    ## 'boat-1': 'sailboat'

    # 'balloon-2': 'yellow lantern. yellow lamp',
    # 'balloon-3': 'red ballon. white ballon. red heart. white heart',
    # "bird-0": "black bird. black swallow",
    # "insect-0": "yellow bee. carpenter bee. honey bee",
    # "person-3": "red person. white person"

    # 'car-2': 'car . black van. bus', # 0.3
    # 'insect-0': 'yellow bee',
    # 'insect-1': 'yellow ant. black ant. orange ant', # 0.55
    # 'person-0': 'bird',
    # 'person-1': 'black person.',
    # 'person-2': 'red person. red human, red athlete, red uniform', # 0.3
    # 'person-3': 'person',
    
    # 'stock-0': 'brown cow', #'white cow. black cow. brown cow',
    # 'stock-1': 'white sheep. white goat', # 0.3
    # 'stock-2': 'white goat'# white sheep. white antelope. white horse. white deer', 0.3
    # 'stock-3': 'gray wolf. gray dog' # 0.5

}

THRES_MAP = {
    'ball-0': 0.7,
    'ball-1': 0.3,
    'ball-2': 0.5,
    'balloon-0': 0.3,
    'balloon-1': 0.41,
    'balloon-2': 0.3,
    'balloon-3':0.3,
    "bird-0": 0.3,
    "insect-0": 0.3,
    "person-1": 0.5,
    "person-2": 0.2,
    "person-3": 0.3,
    'airplane-1': 0.3,
    'airplane-2': 0.2,

    'bird-0': 0.15,
    'boat-1': 0.6,
    'boat-2': 0.15,

}

COND = {
    'ball-0': True, 
    # 'ball-1': 'small green yellow ball. tennis ball',
    'ball-2': True,
    
    # 'balloon-0': 'red balloon. blue balloon. orange balloon. purple balloon. pink balloon. green balloon. red ball. blue ball. orange ball. purple ball. green ball. pink balloon',
    'balloon-1': True,
    'boat-1': True,
    'car-2': True,
}

result = {}
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
        print(f'>>> Run {category}')

        img_folder = osp.join(DATA_DIR, category, 'img1')
        list_fnames = os.listdir(img_folder)
        list_fnames.sort()  
        choice_fanmes = list_fnames #[list_fnames[0]] #random.choices(list_fnames, k=10)
        caption = category.split('-')[0]
        if CAPTION_MAP.get(category) is not None:
            caption = CAPTION_MAP[category]
        elif FINETUNE_ONLY:
            continue

        save_cat = osp.join(SAVE_DIR, category)
        vis_save_cat = osp.join(DROP_SAVE_DIR, category)
        os.makedirs(vis_save_cat, exist_ok=True)
        os.makedirs(save_cat, exist_ok=True)

        final_bboxes = []
        for frame_id, fname in enumerate(choice_fanmes):
            # box_save_path = osp.join(box_cat_save_dir, f'{fname}').replace('.jpg', '.npy')
            # if osp.isfile(box_save_path):
            #     continue
            fpath = osp.join(img_folder, fname)
            # print(fpath)
            save_path = osp.join(save_cat, fname)

            img = load_jpg(fpath)
            result, pred_boxes, proposal_visual_features = glip_demo.run_on_web_image(img, caption, THRES_MAP[category]) 
            
            # from box_proc import remove_large_boxes
            # new_boxes = remove_large_boxes(pred_boxes)

            # before post-proc 
            cv2.imwrite(save_path, result)
            # print('pred_boxes.bbox', pred_boxes.bbox)
            
            # after post-proc
            nms_output = nms(boxes=pred_boxes.bbox, scores=pred_boxes.get_field("scores"), iou_threshold=0.4)
            bboxes = pred_boxes.bbox[nms_output]
            proposal_visual_features = proposal_visual_features[nms_output]
            # scores = pred_boxes.get_field("scores")[nms_output]
            img = cv2.imread(fpath)
            
            areas = []
            for box in bboxes:
                x1, y1, x2, y2 = box 
                areas.append((x2-x1+1)*(y2-y1+1))
            if len(areas) != 0:
                median_area = statistics.median(areas)
            else:
                print('No box is found')
                median_area = 100000000

            keep = []
            for box_id in range(bboxes.shape[0]):
                x1, y1, x2, y2 = bboxes[box_id] 
                start_point = (int(x1), int(y1))
                end_point = (int(x2), int(y2))
                w = x2 - x1 + 1
                h = y2 - y1 + 1
                # if w * h > img.shape[0] * img.shape[1] / 4: 
                #     continue
                if COND[category] and w * h > 7 * median_area:
                    continue

                color = (255, 0, 0)
                thickness = 2
                final_bboxes.append([frame_id, -1, x1, y1, w, h, 1, 1, 1])
                img = cv2.rectangle(img, start_point, end_point, color, thickness)
                keep.append(box_id)
            proposal_visual_features = proposal_visual_features[keep]
            # import pdb; pdb.set_trace()
            drop_save_path = osp.join(vis_save_cat, fname)
            cv2.imwrite(drop_save_path, img)
            # np.save(box_save_path, new_boxes)
            import pdb; pdb.set_trace()
            break
        np.savetxt(osp.join(BOX_SAVE_DIR, f'{category}.txt'), np.array(final_bboxes), fmt='%.6f', delimiter=',')

    pass

if __name__ == '__main__':
    main()
