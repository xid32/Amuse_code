seed = 42


### CAL
is_bn = True
is_before_layernorm = True
is_post_layernorm = False
forget_gate = None
q_dims = [64, 64, 32, 32, 8]
prumerge = [True, False, False, False, False]


###
question_layer = 5
qst_vocab_size = [93] * question_layer
word_embed_size = [512, 96, 96, 192, 192, 512]
embed_size = [96, 96, 192, 192, 512, 768]
num_layers = [2] * question_layer
hidden_size = [512] * question_layer + [768]
num_head = [4] * question_layer
vecize = [True] + [False] * (question_layer-1)
tanh_flag = [True] * question_layer

###
w_qst_vocab_size = 93
w_word_embed_size = 512
w_embed_size = 768
w_num_layers = 2
w_hidden_size = 768
w_num_head = 4
w_vecize = True
w_tanh_flag = True

###
random_mask = False
random_type = "both"  # "patch" "time"  "both"
mask_target_type = "both"  # “visual” "audio" "both"
random_mask_rate = 0.5




vocab_path = '/home/src/DDDD/wk/iwla_acc/datasets/json_update/avqa-train.json'
audio_dir = "/home/src/DDDD/wk/iwla_acc/datasets/audios"
video_frame_dir = "/home/src/DDDD/wk/iwla_acc/datasets/frames_1"
yolo_path = "/home/src/DDDD/wk/iwla_acc/datasets/yolo"
source_sep_path = "/home/src/DDDD/wk/iwla_acc/datasets/source_sep"
diff_bpm_path = "/home/src/DDDD/wk/iwla_acc/datasets/diff_bpm"
label_train = "/home/src/DDDD/wk/iwla_acc/datasets/json_update/avqa-train.json"
label_test = "/home/src/DDDD/wk/iwla_acc/datasets/json_update/avqa-test.json"
label_val = "/home/src/DDDD/wk/iwla_acc/datasets/json_update/avqa-val.json"
transform = None


llf_flag = True
att_neck = False

wandb = False
model_name = "iwla"

mode = "train"

#### train

num_workers = 4
batch_size = 8
is_vit_ln = True
lr_caf = 1e-3
lr = 1e-3
epochs = 30
log_interval = 5
model_save_dir = "/home/src/DDDD/wk/iwla_acc/ckpts/iwla"



#### test

model_load_dir = "/home/src/DDDD/wk/iwla_acc/ckpts/iwla/best_model_81.30264819.pt"




