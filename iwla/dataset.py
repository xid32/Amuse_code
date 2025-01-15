import warnings
warnings.filterwarnings("ignore")
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
import warnings
from iwla import configs as config
import torch.nn.functional as F


def ids_to_multinomial(categories):
    """ label encoding
    Returns:
      1d array, multimonial representation, e.g. [1,0,1,0,0,...]
    """
    id_to_idx = {id: F.one_hot(torch.tensor(index), num_classes=42) for index, id in enumerate(categories)}

    return id_to_idx


def vocab_init(vocab_path):
    samples = json.load(open(vocab_path, 'r'))
    ques_vocab = ['<pad>']
    ans_vocab = []
    i = 0
    for sample in samples:
        i += 1
        question = sample['question_content'].rstrip().split(' ')
        question[-1] = question[-1][:-1]

        p = 0
        for pos in range(len(question)):
            if '<' in question[pos]:
                question[pos] = ast.literal_eval(sample['templ_values'])[p]
                p += 1

        for wd in question:
            if wd not in ques_vocab:
                ques_vocab.append(wd)
        if sample['anser'] not in ans_vocab:
            ans_vocab.append(sample['anser'])
    return ques_vocab, ans_vocab




class IWLA_dataset(Dataset):
    def __init__(self, config, dataset_json_path):
        ques_vocab, ans_vocab = vocab_init(config.vocab_path)
        self.ques_vocab = ques_vocab
        self.ans_vocab = ans_vocab
        self.id_to_idx = ids_to_multinomial(self.ans_vocab)
        self.word_to_ix = {word: i for i, word in enumerate(self.ques_vocab)}

        self.samples = json.load(open(dataset_json_path, 'r'))
        self.max_len = 14  # question length

        self.audio_dir = config.audio_dir
        self.video_frame_dir = config.video_frame_dir
        self.transform = config.transform
        self.yolo_path = config.yolo_path
        self.source_sep_path = config.source_sep_path
        self.diff_bpm_path = config.diff_bpm_path
        self.yolo_list = [i.replace(".npy", "") for i in os.listdir(self.yolo_path)]
        self.source_sep_list = [i.replace(".npy", "") for i in os.listdir(os.path.join(self.source_sep_path, "f_a"))]
        self.diff_bpm_list = [i.replace(".npy", "") for i in os.listdir(os.path.join(self.diff_bpm_path, "f_a"))]

        self.video_list = []
        self.sample_list = []
        for sample in self.samples:
            video_name = sample['video_id']
            if video_name not in self.video_list and video_name in self.yolo_list and video_name in self.source_sep_list and video_name in self.diff_bpm_list:
                self.video_list.append(video_name)
                self.sample_list.append(sample)

        self.video_len = 60 * len(self.video_list)

        self.my_normalize = Compose([
            # Resize([384,384], interpolation=Image.BICUBIC),
            Resize([256, 256], interpolation=Image.BICUBIC),
            # Resize([224,224], interpolation=Image.BICUBIC),
            # CenterCrop(224),
            Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ])

        ### ---> yb calculate stats for AVQA
        self.norm_mean = -5.385333061218262
        self.norm_std = 3.5928637981414795

    ### <----

    def __len__(self):
        return len(self.sample_list)

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
            mix_lambda = np.random.beta(5, 5)

            mix_waveform = mix_lambda * waveform1 + (1 - mix_lambda) * waveform2
            waveform = mix_waveform - mix_waveform.mean()

        ## yb: align ##
        if waveform.shape[1] > 16000 * (1.95 + 0.1):
            sample_indx = np.linspace(0, waveform.shape[1] - 16000 * (1.95 + 0.1), num=5, dtype=int)
            waveform = waveform[:, sample_indx[idx]:sample_indx[idx] + int(16000 * 1.95)]

        ## align end ##

        fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
                                                  window_type='hanning', num_mel_bins=256, dither=0.0, frame_shift=5)

        # target_length = int(1024 * (self.opt.audio_length/10)) ## for audioset: 10s
        target_length = 256  ## yb: overwrite for swin

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
        sample = self.sample_list[idx]
        name = sample['video_id']
        ### ---> video frame process

        total_num_frames = len(glob.glob(os.path.join(self.video_frame_dir, name, '*.jpg')))
        sample_indx = np.linspace(1, total_num_frames, num=5, dtype=int)
        total_img = []
        for tmp_idx in sample_indx:
            tmp_img = torchvision.io.read_image(os.path.join(self.video_frame_dir, name, str("{:06d}".format(tmp_idx)) + '.jpg')) / 255
            tmp_img = self.my_normalize(tmp_img)
            total_img.append(tmp_img)
        total_img = torch.stack(total_img)

        # question
        question_id = sample['question_id']
        question = sample['question_content'].rstrip().split(' ')
        question[-1] = question[-1][:-1]
        p = 0
        for pos in range(len(question)):
            if '<' in question[pos]:
                question[pos] = ast.literal_eval(sample['templ_values'])[p]
                p += 1
        if len(question) < self.max_len:
            n = self.max_len - len(question)
            for i in range(n):
                question.append('<pad>')
        idxs = [self.word_to_ix[w] for w in question]
        ques = torch.tensor(idxs, dtype=torch.long)

        # answer
        answer = sample['anser']
        label = self.id_to_idx[answer]
        label = torch.from_numpy(np.array(label)).long()

        ### ---> loading all audio frames
        total_audio = []
        for audio_sec in range(5):
            fbank, mix_lambda = self._wav2fbank(os.path.join(self.audio_dir, name + '.wav'), idx=audio_sec)
            total_audio.append(fbank)
        total_audio = torch.stack(total_audio)
        ### <----

        ### load yolo source sep diff bpm
        yolo_f = torch.from_numpy(np.load(os.path.join(self.yolo_path, name+".npy")))
        sep_f_a = torch.from_numpy(np.load(os.path.join(self.source_sep_path, "f_a", name + ".npy")))
        sep_f_v = torch.from_numpy(np.load(os.path.join(self.source_sep_path, "f_v", name + ".npy")))
        diff_f_a = torch.from_numpy(np.load(os.path.join(self.diff_bpm_path, "f_a", name + ".npy")))
        diff_f_v = torch.from_numpy(np.load(os.path.join(self.diff_bpm_path, "f_v", name + ".npy")))

        sample = {
            'audio': total_audio,
            'visual': total_img,
            'question': ques,
            'yolo': yolo_f,
            'sep_f_a': sep_f_a,
            'sep_f_v': sep_f_v,
            'diff_f_a': diff_f_a,
            'diff_f_v': diff_f_v,
            'label': label
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


if __name__ == '__main__':
    datasets = IWLA_dataset(config, "../datasets/json_update/avqa-train.json")
    train_loader = DataLoader(datasets, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)
    for sample in train_loader:
        print(sample["audio"].shape)
        print(sample["visual"].shape)
        print(sample["question"].shape)
        print(sample["yolo"].shape)
        print(sample["sep_f_a"].shape)
        print(sample["sep_f_v"].shape)
        print(sample["diff_f_a"].shape)
        print(sample["diff_f_v"].shape)
        print(sample["label"].shape)
        break

# torch.Size([4, 10, 256, 256])
# torch.Size([4, 10, 3, 256, 256])
# torch.Size([4, 14])
# torch.Size([4, 640, 40, 40])
# torch.Size([4, 6, 768])
# torch.Size([4, 5, 768])
# torch.Size([4, 6, 768])
# torch.Size([4, 5, 768])
# torch.Size([4, 42])

