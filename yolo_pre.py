import warnings
warnings.filterwarnings("ignore")
import tqdm
import numpy as np
import glob
import os
from PIL import Image

from yolo_module.yolo_utils import YOLO


def yolo_feature(img_path, yolo):
    try:
        image = Image.open(img_path)
    except:
        return
    else:
        outs = yolo.extract_feature(image)
    return outs


def main(video_frame_path, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    yolo = YOLO()
    frames_list = os.listdir(video_frame_path)
    for name in tqdm.tqdm(frames_list, total=len(frames_list)):
        save_path = os.path.join(save_dir, name+".npy")
        if os.path.exists(save_path):
            continue
        total_num_frames = len(glob.glob(os.path.join(video_frame_path, name, '*.jpg')))
        idx = int(total_num_frames // 2)
        img_path = os.path.join(video_frame_path, name, str("{:06d}".format(idx)) + '.jpg')
        feature = yolo_feature(img_path, yolo)[1]
        feature = feature.squeeze().detach().cpu().numpy()
        np.save(os.path.join(save_dir, name+".npy"), feature)



if __name__ == '__main__':
    video_frame_path = "./datasets/frames_1"
    save_dir = "./datasets/yolo"
    main(video_frame_path, save_dir)





