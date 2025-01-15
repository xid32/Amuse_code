import warnings
warnings.filterwarnings('ignore')
import wandb
import sys
from rt_module import configs as config
import os
from rt_module.ts_model import Timestamp_Feature_Encoder
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from PIL import Image
from rt_module.dataset import float32_to_int16, int16_to_float32
from rt_module.dataset import Timestamp_dataset
from torch.utils.data import Dataset, DataLoader
import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import glob
import librosa
import tqdm
import json


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def inference(model):
    with torch.no_grad():
        my_normalize = Compose([
            Resize([256, 256], interpolation=Image.BICUBIC),
            Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ])
        video_list = []
        for json_list in config.json_lists:
            samples = json.load(open(json_list, 'r'))
            for sample in samples:
                video_name = sample['video_id']
                if video_name not in video_list:
                    video_list.append(video_name)
        total_batch = int(len(video_list) // config.batch_size + 1)
        total_num = len(video_list)
        for b_i in tqdm.tqdm(range(total_batch), total=total_batch):
            visuals = []
            audios = []
            video_names = []
            if int((b_i+1)*config.batch_size) > total_num:
                video_list_min = video_list[int((b_i)*config.batch_size):]
            else:
                video_list_min = video_list[int(b_i*config.batch_size):int((b_i+1)*config.batch_size)]
            for video_name in video_list_min:
                audio, sr = librosa.load(os.path.join(config.audio_dir, video_name + '.wav'), sr=config.sr)
                total_num_frames = len(glob.glob(os.path.join(config.video_frame_dir, video_name, '*.jpg')))
                if audio.shape[0] < config.sr * config.max_sec or total_num_frames < config.max_sec:
                    continue
                audio_segs = []
                for s_i in range(6):
                    tmp = audio[s_i * 320000:(s_i + 1) * 320000]
                    tmp = float32_to_int16(tmp)
                    tmp = int16_to_float32(tmp)
                    audio_segs.append(tmp)
                audio_segs = np.array(audio_segs, dtype=np.float32)
                # audio = float32_to_int16(audio[:config.sr * config.max_sec])
                # audio_segs = int16_to_float32(audio)
                audios.append(audio_segs)
                sample_indx = np.linspace(1, total_num_frames, num=5, dtype=int)
                total_img = []
                for tmp_idx in sample_indx:
                    tmp_img = torchvision.io.read_image(os.path.join(config.video_frame_dir, video_name,
                                                                     str("{:06d}".format(tmp_idx)) + '.jpg')) / 255
                    tmp_img = my_normalize(tmp_img)
                    total_img.append(tmp_img)
                total_img = torch.stack(total_img)
                visuals.append(total_img)
                video_names.append(video_name)
            audios = np.array(audios)
            audios = torch.from_numpy(audios)
            audios = audios.to("cuda")
            visuals = torch.stack(visuals, dim=0)
            visuals = visuals.to("cuda")
            v_outs, a_outs = model(audios, visuals)
            for idx in range(v_outs.size(0)):
                v_out = v_outs[idx].detach().cpu().numpy()
                a_out = a_outs[idx].detach().cpu().numpy()
                np.save(os.path.join(config.save_dir, "f_v", video_names[idx]+".npy"), v_out)
                np.save(os.path.join(config.save_dir, "f_a", video_names[idx]+".npy"), a_out)
    return

def main():
    # Training settings
    torch.manual_seed(config.seed)
    if not os.path.exists(os.path.join(config.save_dir, "f_v")):
        os.makedirs(os.path.join(config.save_dir, "f_v"))
    if not os.path.exists(os.path.join(config.save_dir, "f_a")):
        os.makedirs(os.path.join(config.save_dir, "f_a"))
    model = Timestamp_Feature_Encoder(config=config, for_avqa=True)
    ckpt = torch.load(config.model_dir, map_location="cpu")
    model.load_state_dict(ckpt)
    model = model.cuda()
    model.eval()
    inference(model)


if __name__ == '__main__':
    main()