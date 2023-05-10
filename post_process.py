import numpy as np
import os
import os.path as osp
import statistics

before_dir = '/home/kimth1/GLIP/iGLIP_GMOT40/before_high_threshold_0.35/GMOT_40_boxes' 
after_dir = '/home/kimth1/GLIP/iGLIP_GMOT40/after_high_threshold_0.35/GMOT_40_boxes'

def intersection(bo, bi):  # returns None if rectangles don't intersect
    xo1, yo1, xo2, yo2 = bo
    xi1, yi1, xi2, yi2 = bi
    dx = min(xo2, xi2) - max(xo1, xi1)
    dy = min(yo2, yi2) - max(yo1, yi1)
    if (dx>=0) and (dy>=0):
        return dx*dy
    return 0

file_list = os.listdir(before_dir)
file_list.sort()
for file_name in file_list:
    track_list = np.loadtxt(osp.join(before_dir, file_name), delimiter=',', dtype=float)
    frame_max = np.max(track_list[:, 0])
    result = []
    cnt1 = 0
    cnt2 = 0
    remove_in_frames = []
    for frame_id in range(int(frame_max) + 1):
        tracks = track_list[track_list[:,0] == frame_id]

        areas = tracks[:,4] * tracks[:,5]
        if areas.shape[0] != 0:
            median_area = statistics.median(areas)
        else:
            median_area = 100000000
        
        big_box = False
        K=15
        if np.count_nonzero(areas > K*median_area) == 1:
            big_box = True

        for track_out in tracks:
            bo = track_out[2:6]
            if (big_box == True) and (bo[2]*bo[3] > K*median_area):
                result.append(np.append(track_out, -1))
                cnt1 += 1
                remove_in_frames.append(frame_id)
                continue
            result.append(np.append(track_out, 1))
            # FP = False
            # cnt = 0
            # for track_in in tracks:
            #     bi = track_in[2:6]
            #     if (bo[2]*bo[3] > bi[2]*bi[3]) and \
            #     intersection([bo[0], bo[1], bo[0]+bo[2]-1, bo[1]+bo[3]-1], [bi[0], bi[1], bi[0]+bi[2]-1, bi[1]+bi[3]-1]) / (bi[2]*bi[3]) > 0.85:
            #         cnt += 1
            #     if cnt > 0:
            #         FP = True

            # if FP == False:
            #     result.append(np.append(track_out, 1))
            #     # result.append(track)
            # else:
            #     result.append(np.append(track_out, -1))
            #     cnt2 += 1
    assert len(track_list) == len(result)
    if len(remove_in_frames) > 3:
        print(file_name, len(track_list), cnt1, remove_in_frames[:3])
    else:
        print(file_name, len(track_list), cnt1)
    np.savetxt(osp.join(after_dir, file_name), np.array(result), fmt='%.6f', delimiter=',')

