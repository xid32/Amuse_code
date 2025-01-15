import timm
import torch
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
import torchlibrosa as tl
import torchaudio as ta
import librosa
import numpy as np
import torch
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pandas as pd
import ast
import json
from PIL import Image
import time
import random
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import torchvision
import torchaudio
import glob
# from counting_rhythm_timestamp_modeling import config
import warnings
warnings.filterwarnings('ignore')




def float32_to_int16(x):
    x = np.clip(x, a_min = -1., a_max = 1.)
    return (x * 32767.).astype(np.int16)

def int16_to_float32(x):
    return (x / 32767.).astype(np.float32)



class Timestamp_dataset(Dataset):

    def __init__(self, config):

        self.samples = json.load(open(config.json_list, 'r'))
        self.labels_csv = pd.read_csv(config.label_dir)

        self.audio_dir = config.audio_dir
        self.video_frame_dir = config.video_frame_dir
        self.transform = config.transform

        self.video_list = []
        self.labels = []
        for sample in self.samples:
            video_name = sample['video_id']
            if video_name not in self.video_list:
                indices = self.labels_csv.loc[self.labels_csv['Audio'] == video_name + '.wav'].index
                self.video_list.append(video_name)
                self.labels.append(np.array(self.labels_csv.iloc[int(indices[0]), 1:].tolist(), dtype=np.float32))

        self.video_len = 60 * len(self.video_list)

        self.my_normalize = Compose([
            # Resize([384,384], interpolation=Image.BICUBIC),
            # Resize([192, 192], interpolation=Image.BICUBIC),
            Resize([256,256], interpolation=Image.BICUBIC),
            # CenterCrop(224),
            Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ])

        ### ---> yb calculate stats for AVQA
        self.norm_mean = -5.385333061218262
        self.norm_std = 3.5928637981414795

        self.sr = config.sr
        self.seg_during = config.seg_during
        self.max_sec = config.max_sec

    ### <----

    def __len__(self):
        return len(self.video_list)

    def _wav2fbank(self, filename, filename2=None, idx=None):
        # mixup
        if filename2 == None:
            waveform, sr = torchaudio.load(filename)
            waveform = waveform - waveform.mean()
        # mixup
        else:
            waveform1, sr = torchaudio.load(filename)
            waveform2, _ = torchaudio.load(filename2)

            waveform1 = waveform1 - waveform1.mean()
            waveform2 = waveform2 - waveform2.mean()

            if waveform1.shape[1] != waveform2.shape[1]:
                if waveform1.shape[1] > waveform2.shape[1]:
                    # padding
                    temp_wav = torch.zeros(1, waveform1.shape[1])
                    temp_wav[0, 0:waveform2.shape[1]] = waveform2
                    waveform2 = temp_wav
                else:
                    # cutting
                    waveform2 = waveform2[0, 0:waveform1.shape[1]]

            # sample lambda from uniform distribution
            # mix_lambda = random.random()
            # sample lambda from beta distribtion
            mix_lambda = np.random.beta(10, 10)

            mix_waveform = mix_lambda * waveform1 + (1 - mix_lambda) * waveform2
            waveform = mix_waveform - mix_waveform.mean()

        ## yb: align ##
        if waveform.shape[1] > 16000 * (1.95 + 0.1):
            sample_indx = np.linspace(0, waveform.shape[1] - 16000 * (1.95 + 0.1), num=10, dtype=int)
            waveform = waveform[:, sample_indx[idx]:sample_indx[idx] + int(16000 * 1.95)]

        ## align end ##

        # fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False, window_type='hanning', num_mel_bins=128, dither=0.0, frame_shift=10) ## original
        fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
                                                  window_type='hanning', num_mel_bins=192, dither=0.0, frame_shift=10)

        # target_length = int(1024 * (self.opt.audio_length/10)) ## for audioset: 10s
        target_length = 192  ## yb: overwrite for swin

        ########### ------> very important: audio normalized
        fbank = (fbank - self.norm_mean) / (self.norm_std * 2)
        ### <--------

        # target_length = 512 ## 5s
        # target_length = 256 ## 2.5s
        n_frames = fbank.shape[0]

        p = target_length - n_frames

        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:target_length, :]

        if filename2 == None:
            return fbank, 0
        else:
            return fbank, mix_lambda

    def __getitem__(self, idx):
        name = self.video_list[idx]
        audio, sr = librosa.load(os.path.join(self.audio_dir, name + '.wav'), sr=self.sr)
        total_num_frames = len(glob.glob(os.path.join(self.video_frame_dir, name, '*.jpg')))
        while audio.shape[0] < self.sr*self.max_sec or total_num_frames < self.max_sec:
            idx = random.randint(0, len(self.video_list))
            name = self.video_list[idx]
            audio, sr = librosa.load(os.path.join(self.audio_dir, name + '.wav'), sr=self.sr)
            total_num_frames = len(glob.glob(os.path.join(self.video_frame_dir, name, '*.jpg')))
        label = self.labels[idx]
        label = torch.from_numpy(label)

        audio_segs = []
        # for s_i in range(int(self.max_sec//self.seg_during)):
        #     tmp = audio[s_i*self.sr*self.seg_during:(s_i+1)*self.sr*self.seg_during]
        for s_i in range(6):
            tmp = audio[s_i*320000:(s_i+1)*320000]
            tmp = float32_to_int16(tmp)
            tmp = int16_to_float32(tmp)
            audio_segs.append(tmp)
        audio_segs = np.array(audio_segs, dtype=np.float32)

        # audio = float32_to_int16(audio[:self.sr*self.max_sec])
        # audio_segs = int16_to_float32(audio)
        audio_segs = torch.from_numpy(audio_segs)

        ### ---> video frame process
        sample_indx = np.linspace(1, total_num_frames, num=5, dtype=int)
        total_img = []
        for tmp_idx in sample_indx:
            tmp_img = torchvision.io.read_image(os.path.join(self.video_frame_dir, name,
                                                             str("{:06d}".format(tmp_idx)) + '.jpg')) / 255
            tmp_img = self.my_normalize(tmp_img)
            total_img.append(tmp_img)
        total_img = torch.stack(total_img)

        sample = {'audio': audio_segs, 'visual': total_img, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample


# class Timestamp_dataset_inf(Dataset):
#     def __init__(self, config):
#         self.audio_dir = config.audio_dir
#         self.video_frame_dir = config.video_frame_dir
#         self.transform = config.transform
#         self.video_list = []
#         for json_list in config.json_lists:
#             self.samples = json.load(open(json_list, 'r'))
#             for sample in self.samples:
#                 video_name = sample['video_id']
#                 if video_name not in self.video_list:
#                     self.video_list.append(video_name)
#
#         self.video_len = 60 * len(self.video_list)
#
#         self.my_normalize = Compose([
#             # Resize([384,384], interpolation=Image.BICUBIC),
#             # Resize([192, 192], interpolation=Image.BICUBIC),
#             Resize([256,256], interpolation=Image.BICUBIC),
#             # CenterCrop(224),
#             Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
#         ])
#
#         ### ---> yb calculate stats for AVQA
#         self.norm_mean = -5.385333061218262
#         self.norm_std = 3.5928637981414795
#
#         self.sr = config.sr
#         self.seg_during = config.seg_during
#         self.max_sec = config.max_sec
#
#     ### <----
#
#     def __len__(self):
#         return len(self.video_list)
#
#     def _wav2fbank(self, filename, filename2=None, idx=None):
#         # mixup
#         if filename2 == None:
#             waveform, sr = torchaudio.load(filename)
#             waveform = waveform - waveform.mean()
#         # mixup
#         else:
#             waveform1, sr = torchaudio.load(filename)
#             waveform2, _ = torchaudio.load(filename2)
#
#             waveform1 = waveform1 - waveform1.mean()
#             waveform2 = waveform2 - waveform2.mean()
#
#             if waveform1.shape[1] != waveform2.shape[1]:
#                 if waveform1.shape[1] > waveform2.shape[1]:
#                     # padding
#                     temp_wav = torch.zeros(1, waveform1.shape[1])
#                     temp_wav[0, 0:waveform2.shape[1]] = waveform2
#                     waveform2 = temp_wav
#                 else:
#                     # cutting
#                     waveform2 = waveform2[0, 0:waveform1.shape[1]]
#
#             # sample lambda from uniform distribution
#             # mix_lambda = random.random()
#             # sample lambda from beta distribtion
#             mix_lambda = np.random.beta(10, 10)
#
#             mix_waveform = mix_lambda * waveform1 + (1 - mix_lambda) * waveform2
#             waveform = mix_waveform - mix_waveform.mean()
#
#         ## yb: align ##
#         if waveform.shape[1] > 16000 * (1.95 + 0.1):
#             sample_indx = np.linspace(0, waveform.shape[1] - 16000 * (1.95 + 0.1), num=10, dtype=int)
#             waveform = waveform[:, sample_indx[idx]:sample_indx[idx] + int(16000 * 1.95)]
#
#         ## align end ##
#
#         # fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False, window_type='hanning', num_mel_bins=128, dither=0.0, frame_shift=10) ## original
#         fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
#                                                   window_type='hanning', num_mel_bins=192, dither=0.0, frame_shift=10)
#
#         # target_length = int(1024 * (self.opt.audio_length/10)) ## for audioset: 10s
#         target_length = 192  ## yb: overwrite for swin
#
#         ########### ------> very important: audio normalized
#         fbank = (fbank - self.norm_mean) / (self.norm_std * 2)
#         ### <--------
#
#         # target_length = 512 ## 5s
#         # target_length = 256 ## 2.5s
#         n_frames = fbank.shape[0]
#
#         p = target_length - n_frames
#
#         # cut and pad
#         if p > 0:
#             m = torch.nn.ZeroPad2d((0, 0, 0, p))
#             fbank = m(fbank)
#         elif p < 0:
#             fbank = fbank[0:target_length, :]
#
#         if filename2 == None:
#             return fbank, 0
#         else:
#             return fbank, mix_lambda
#
#     def __getitem__(self, idx):
#         name = self.video_list[idx]
#         audio, sr = librosa.load(os.path.join(self.audio_dir, name + '.wav'), sr=self.sr)
#         total_num_frames = len(glob.glob(os.path.join(self.video_frame_dir, name, '*.jpg')))
#         while audio.shape[0] < self.sr*self.max_sec or total_num_frames < self.max_sec:
#             idx = random.randint(0, len(self.video_list))
#             name = self.video_list[idx]
#             audio, sr = librosa.load(os.path.join(self.audio_dir, name + '.wav'), sr=self.sr)
#             total_num_frames = len(glob.glob(os.path.join(self.video_frame_dir, name, '*.jpg')))
#         label = self.labels[idx]
#         label = torch.from_numpy(label)
#
#         audio_segs = []
#         for s_i in range(int(self.max_sec//self.seg_during)):
#             tmp = audio[s_i*self.sr*self.seg_during:(s_i+1)*self.sr*self.seg_during]
#             tmp = float32_to_int16(tmp)
#             tmp = int16_to_float32(tmp).reshape([1, self.sr*self.seg_during])
#             audio_segs.append(tmp)
#         audio_segs = np.array(audio_segs, dtype=np.float32)
#
#         # audio = float32_to_int16(audio[:self.sr*self.max_sec])
#         # audio_segs = int16_to_float32(audio)
#         audio_segs = torch.from_numpy(audio_segs)
#
#         ### ---> video frame process
#         sample_indx = np.linspace(1, total_num_frames, num=5, dtype=int)
#         total_img = []
#         for tmp_idx in sample_indx:
#             tmp_img = torchvision.io.read_image(os.path.join(self.video_frame_dir, name,
#                                                              str("{:06d}".format(tmp_idx)) + '.jpg')) / 255
#             tmp_img = self.my_normalize(tmp_img)
#             total_img.append(tmp_img)
#         total_img = torch.stack(total_img)
#
#         sample = {'audio': audio_segs, 'visual': total_img, 'label': label}
#
#         if self.transform:
#             sample = self.transform(sample)
#
#         return sample



# if __name__ == '__main__':
#     datasets = Timestamp_dataset(config=config)
#     train_loader = DataLoader(datasets, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
#     for sample in train_loader:
#         print(sample["audio"].shape)
#         print(sample["visual"].shape)
#         print(sample["label"].shape)
#         break


