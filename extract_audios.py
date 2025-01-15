import os
import moviepy
from moviepy.audio.AudioClip import AudioArrayClip
from moviepy.editor import VideoFileClip
import tqdm

'''
Function: Extract audio files(.wav) from videos.
'''


def get_audio_wav(name, save_pth, audio_name):
    video = VideoFileClip(name)
    audio = video.audio
    audio.write_audiofile(os.path.join(save_pth, audio_name), fps=32000)


if __name__ == "__main__":

    video_pth = "/home/src/DDDD/wk/iwla_acc/datasets/videos/"
    save_pth = "/home/src/DDDD/wk/iwla_acc/datasets/audios/"

    sound_list = os.listdir(video_pth)
    for audio_id in tqdm.tqdm(sound_list, total=len(sound_list)):
        name = os.path.join(video_pth, audio_id)
        audio_name = audio_id[:-4] + '.wav'
        exist_lis = os.listdir(save_pth)
        if audio_name in exist_lis:
            print("already exist!")
            continue
        try:
            get_audio_wav(name, save_pth, audio_name)
        except:
            print("cannot load ", name)

    print("\n------------------------------ end -------------------------------\n")
