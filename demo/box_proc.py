import enum
import numpy as np
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_nms
from maskrcnn_benchmark.structures.bounding_box import BoxList

def remove_large_boxes(pred_boxes: BoxList):
    # boxes = pred_boxes.bbox
    list_area = pred_boxes.area().numpy()
    img_area = pred_boxes.size[0]*pred_boxes.size[1]
    avg_area = np.mean(list_area)
    ratio_area = list_area/img_area

    choose_ids = (list_area < 5*avg_area) & (ratio_area < 0.3*img_area)
    print(f'Remove {np.sum(~choose_ids)} boxes')
    choose_boxes = pred_boxes.bbox.numpy()[choose_ids]
    return choose_boxes