import cv2
import numpy as np
import os
import os.path as osp

# def get_color(idx):
#     idx = idx * 3
#     color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
#     return color

def plot_tracking(image, tlwhs, obj_ids, flags, frame_id=0, ids2=None):
    im = np.ascontiguousarray(np.copy(image))
    im_h, im_w = im.shape[:2]

    #text_scale = max(1, image.shape[1] / 1600.)
    #text_thickness = 2
    #line_thickness = max(1, int(image.shape[1] / 500.))
    text_scale = 2
    text_thickness = 3
    line_thickness = 5

    cv2.putText(im, 'frame: %d num: %d' % (frame_id, len(tlwhs)),
                (0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=2)

    for i, tlwh in enumerate(tlwhs):
        x1, y1, w, h = tlwh
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        obj_id = int(obj_ids[i])
        id_text = '{}'.format(int(obj_id))
        if ids2 is not None:
            id_text = id_text + ', {}'.format(int(ids2[i]))
        # color = get_color(abs(obj_id))
        if int(flags[i]) > 0:
            color = (255,0,0)
        else:
            color = (0,255,0)
        cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
        # import pdb;pdb.set_trace()
        # cv2.putText(im, id_text, (intbox[0], intbox[1]), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 255, 0),
        #             thickness=text_thickness)
    return im

gen_vid = False
gen_vid_folder = True
split = '/cm/shared/kimth1/Tracking/ByteTrack/datasets/GMOT_40/test'
# split = '/cm/shared/kimth1/Tracking/DanceTrack/val'

for video_name in os.listdir(split):
    if video_name != 'stock-3':
        continue
    print(video_name)
    resutl_path = f'/home/kimth1/GLIP/iGLIP_GMOT40/after_high_threshold_0.35/GMOT_40_boxes/{video_name}.txt'
    original_folder = f'{split}/{video_name}/img1'
    vis_folder = f'/home/kimth1/GLIP/iGLIP_GMOT40/after_high_threshold_0.35/vis/{video_name}'
    contain_videos = f'/home/kimth1/GLIP/iGLIP_GMOT40/after_high_threshold_0.35/vis/'
    if gen_vid == True:
        os.makedirs(contain_videos, exist_ok=True)
    try:
        results = np.loadtxt(resutl_path, delimiter=',', dtype=int)
    except:
        print(f'{resutl_path} not existed')
        continue

    frames = {}
    for i in range(len(results)):
        frame_id = results[i][0]
        if frame_id in frames:
            continue
        frames[frame_id] = {}
        frames[frame_id]["obj_ids"] = []
        frames[frame_id]["tlwhs"] = []
        frames[frame_id]["flag"] = []
        for j in range(len(results)):
            if results[j][0] == frame_id:
                frames[frame_id]["obj_ids"].append(results[j][1])
                frames[frame_id]["tlwhs"].append(results[j][2:6])
                frames[frame_id]["flag"].append(results[j][-1])
        
    print(video_name, len(frames.keys()))

    if gen_vid_folder == True:
        os.makedirs(vis_folder, exist_ok=True)
    img_array = []
    frame_list = os.listdir(original_folder)
    frame_list.sort()
    # print(frame_list)
    for frame_file in frame_list:
        frame_id = int(frame_file[:-4])
        if frame_id not in frames:
            continue
        img = cv2.imread(osp.join(original_folder, frame_file))
        tlwhs = frames[frame_id]["tlwhs"]
        obj_ids = frames[frame_id]["obj_ids"]
        flags = frames[frame_id]["flag"]
        # import pdb;pdb.set_trace()
        img = plot_tracking(img, tlwhs, obj_ids, flags, frame_id)
        if gen_vid_folder == True:
            cv2.imwrite(osp.join(vis_folder, frame_file), img)
        img_array.append(img)

    if gen_vid == True:
        height, width, layers = img_array[0].shape
        size = (width, height)        
        out = cv2.VideoWriter(osp.join(contain_videos, f'{video_name}.avi'), cv2.VideoWriter_fourcc(*'DIVX'), 10, size)
        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()    


