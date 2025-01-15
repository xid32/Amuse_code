# for Rhythmic Timestamp Visual-Audio Feature Encoder modeling



# sample rate
sr = 32000
seg_during = 3 # 3 for diff bpm, 2 for source sep
max_sec = 60
window_size = 1024
hop_size = 320
mel_bins = 64
fmin = 50
fmax = 14000
enable_tscam = True
enable_repeat_mode=False
htsat_attn_heatmap=False
loss_type = None
label_dir = "./datasets/csv/diff_bpm_intervals.csv"
audio_dir = "./datasets/audios"
video_frame_dir = "./datasets/frames_1"
json_list = "./datasets/json_update/avqa-train.json"
json_lists = ["./datasets/json_update/avqa-train.json", "./datasets/json_update/avqa-test.json", "./datasets/json_update/avqa-val.json"]
model_dir = "./ckpts/diff_bpm/diff_bpm.ckpt"
swin_a_ckpt = "./ckpts/htsat/HTSAT_AudioSet_Saved_1.ckpt"
model_save_dir = "./ckpts/diff_bpm/diff_bpm.ckpt"
save_dir = "./datasets/diff_bpm"
transform = None
wandb = False
model_name = "iwla-rt"
seed = 42
batch_size = 16
num_workers = 8
lr = 0.001
epochs=5







