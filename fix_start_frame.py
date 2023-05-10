import os
import numpy as np
import os.path as osp

data_dir = '/cm/shared/kimth1/Tracking/DanceTrack/val'
result_dir = '/home/kimth1/GLIP/dancetrack_0.4/val/boxes'
fixed_result_dir = '/home/kimth1/GLIP/dancetrack_0.4/val/boxes_fixed'
os.makedirs(fixed_result_dir, exist_ok=True)
for vid_name in os.listdir(data_dir):
    track_path = osp.join(result_dir, f'{vid_name}.txt')
    vid_path = osp.join(data_dir, vid_name, 'img1')
    list_frame = os.listdir(vid_path)
    list_frame.sort()
    first_frame = list_frame[0]
    # import pdb;pdb.set_trace()
    first_frame_id = int(first_frame.split('.')[0].replace('frame',''))
    print('first_frame_id', first_frame_id)
    a = np.loadtxt(track_path, delimiter=',', dtype=float)
    a[:,0] += first_frame_id
    new_track_path = osp.join(fixed_result_dir, f'{vid_name}.txt')
    np.savetxt(new_track_path, a, fmt='%.6f', delimiter=',')

