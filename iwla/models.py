import torch
# import torchvision
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import timm
from einops import rearrange, repeat
from iwla import configs as config


class CAL(nn.Module):

    def __init__(self,
                 output_dim,
                 use_bn=True,
                 is_before_layernorm=True,
                 is_post_layernorm=True,
                 forget_gate=True,
                 type="av",
                 q_emb_dim=48,
                 prumerge=False
                 ):
        super().__init__()
        self.is_before_layernorm = is_before_layernorm
        self.is_post_layernorm = is_post_layernorm
        self.gate_av = nn.Parameter(torch.zeros(1))
        self.use_bn = use_bn
        self.forget_gate = forget_gate
        self.bn2 = nn.BatchNorm2d(output_dim)

        """

        """

        self.type = type
        self.prumerge = prumerge
        if self.type == "q":
            self.ques_emb = nn.Conv2d(14, q_emb_dim*q_emb_dim, 1, bias=False)



        self.activation = nn.ReLU(inplace=True)

        self.ln_before = nn.LayerNorm(output_dim)
        self.ln_post = nn.LayerNorm(output_dim)
        """

        """
        self.mlp = nn.Conv2d(output_dim, output_dim, 1, bias=False)

        self.gate = nn.Parameter(torch.zeros(1))

    def adaptive_token_reduction(self, x, attention_mask, k=None):
        B, L, M = x.shape
        attention_mask = torch.sum(attention_mask, dim=2)  # Flatten M*N to a single dimension per batch
        Q1 = attention_mask.quantile(0.25, dim=1, keepdim=True)
        Q3 = attention_mask.quantile(0.75, dim=1, keepdim=True)
        IQR = Q3 - Q1
        upper_bound = Q3 + 1.5 * IQR
        is_outlier = attention_mask > upper_bound

        outlier_indices = torch.nonzero(is_outlier, as_tuple=True)

        # Ensure we have at most k outliers per batch
        if k is not None and len(outlier_indices[1]) > k:
            outlier_indices = outlier_indices[:, :k]

        # Token Clustering
        token_dis = torch.bmm(x.permute(0, 2, 1), x)
        batch_outliers, token_outliers = outlier_indices

        # Assign non-outliers to closest outlier
        new_tokens = torch.zeros(B, L, M, device=x.device)
        for b in range(B):
            outlier_members_dict = {}
            outlier_indices_b = token_outliers[batch_outliers == b]
            if outlier_indices_b.size(0) == 0:
                continue
            non_outlier_indices_b = torch.tensor([i for i in range(M) if i not in outlier_indices_b], device=x.device)

            for outlier_idx in outlier_indices_b:
                outlier_members_dict[outlier_idx.item()] = 1
                new_tokens[b, :, outlier_idx] += x[b, :, outlier_idx]

            for non_outlier_idx in non_outlier_indices_b:
                distances = token_dis[b, non_outlier_idx, outlier_indices_b]
                closest_outlier_idx = outlier_indices_b[distances.argmin()]

                # Assign non-outlier to closest outlier
                new_tokens[b, :, closest_outlier_idx] += x[b, :, non_outlier_idx]
                outlier_members_dict[closest_outlier_idx.item()] += 1

            # Weighted Cluster Center Update for outliers
            for outlier_idx in outlier_indices_b:
                new_tokens[b, :, outlier_idx] = new_tokens[b, :, outlier_idx] / outlier_members_dict[outlier_idx.item()]

        return new_tokens

    def forward(self, x, y=None):
        # res ca

        if self.type == "q":
            y = self.ques_emb(y.unsqueeze(-1)).permute(0, 2, 1, 3)
            y = repeat(y, 'b s dim k -> b t s dim k', t=5)
            y = rearrange(y, 'b t s dim k -> (b t) s dim k')


        cross_att = torch.bmm(x.squeeze(-1), y.squeeze(-1).permute(0, 2, 1))

        cross_att = F.softmax(cross_att, dim=-1)

        if self.prumerge and self.type == "q":
            x = self.adaptive_token_reduction(x.squeeze(-1).permute(0, 2, 1), cross_att).permute(0, 2, 1).unsqueeze(-1)

        x_res = torch.bmm(cross_att, y.squeeze(-1)).unsqueeze(-1)

        x = x + self.gate_av * x_res.contiguous()

        if self.is_before_layernorm:
            x = self.ln_before(x.squeeze(-1).permute(0, 2, 1)).permute(0, 2, 1).unsqueeze(-1)

        output = self.activation(x)
        if self.use_bn:
            output = self.bn2(output)

        output = self.mlp(output)

        if self.is_post_layernorm:
            output = self.ln_post(output.squeeze(-1).permute(0, 2, 1)).permute(0, 2, 1).unsqueeze(-1)

        if self.forget_gate is not None:
            output = self.gate * output

        return output


class CAL_yolo(nn.Module):

    def __init__(self,
                 output_dim,
                 use_bn=True,
                 is_before_layernorm=True,
                 is_post_layernorm=True,
                 forget_gate=True,
                 ):
        super().__init__()
        self.is_before_layernorm = is_before_layernorm
        self.is_post_layernorm = is_post_layernorm
        self.gate_av = nn.Parameter(torch.zeros(1))
        self.use_bn = use_bn
        self.forget_gate = forget_gate
        self.bn2 = nn.BatchNorm2d(output_dim)

        """

        """
        self.ques_emb = nn.Conv2d(14, 768, 1, bias=False)



        self.activation = nn.ReLU(inplace=True)

        self.ln_before = nn.LayerNorm(output_dim)
        self.ln_post = nn.LayerNorm(output_dim)
        """

        """
        self.mlp = nn.Conv2d(output_dim, output_dim, 1, bias=False)

        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x, q):
        # res ca
        y = self.ques_emb(q.unsqueeze(-1)).permute(0, 2, 1, 3)
        y = torch.mean(y, dim=1).unsqueeze(1)

        cross_att = torch.bmm(x.squeeze(-1), y.squeeze(-1).permute(0, 2, 1))

        cross_att = F.softmax(cross_att, dim=-1)
        x_res = torch.bmm(cross_att, y.squeeze(-1)).unsqueeze(-1)

        x = x + self.gate_av * x_res.contiguous()

        if self.is_before_layernorm:
            x = self.ln_before(x.squeeze(-1)).permute(0, 2, 1).unsqueeze(-1)

        output = self.activation(x)
        if self.use_bn:
            output = self.bn2(output)

        output = self.mlp(output)

        if self.is_post_layernorm:
            output = self.ln_post(output.squeeze(-1).permute(0, 2, 1)).permute(0, 2, 1).unsqueeze(-1)

        if self.forget_gate is not None:
            output = self.gate * output

        return output


class CAL_weak_label(nn.Module):

    def __init__(self,
                 output_dim,
                 use_bn=True,
                 is_before_layernorm=True,
                 is_post_layernorm=True,
                 forget_gate=True,
                 dim=6
                 ):
        super().__init__()
        self.is_before_layernorm = is_before_layernorm
        self.is_post_layernorm = is_post_layernorm
        self.gate_av = nn.Parameter(torch.zeros(1))
        self.use_bn = use_bn
        self.forget_gate = forget_gate
        self.bn2 = nn.BatchNorm2d(output_dim)

        """

        """
        self.ques_emb = nn.Conv2d(14, dim, 1, bias=False)



        self.activation = nn.ReLU(inplace=True)

        self.ln_before = nn.LayerNorm(output_dim)
        self.ln_post = nn.LayerNorm(output_dim)
        """

        """
        self.mlp = nn.Conv2d(output_dim, output_dim, 1, bias=False)

        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x, q):
        # res ca
        y = self.ques_emb(q.unsqueeze(-1))

        cross_att = torch.bmm(x.squeeze(-1), y.squeeze(-1).permute(0, 2, 1))

        cross_att = F.softmax(cross_att, dim=-1)
        x_res = torch.bmm(cross_att, y.squeeze(-1)).unsqueeze(-1)

        x = x + self.gate_av * x_res.contiguous()

        if self.is_before_layernorm:
            x = self.ln_before(x.squeeze(-1)).permute(0, 2, 1).unsqueeze(-1)

        output = self.activation(x)
        if self.use_bn:
            output = self.bn2(output)

        output = self.mlp(output)

        if self.is_post_layernorm:
            output = self.ln_post(output.squeeze(-1).permute(0, 2, 1)).permute(0, 2, 1).unsqueeze(-1)

        if self.forget_gate is not None:
            output = self.gate * output

        return output


class Question_Encoder(nn.Module):
    def __init__(self, qst_vocab_size, word_embed_size, embed_size, num_layers, hidden_size, num_head, vecize=True,
                 tanh_flag=True):
        super(Question_Encoder, self).__init__()
        self.vecize = vecize
        self.tanh_flag = tanh_flag
        if self.vecize:
            self.word2vec = nn.Embedding(qst_vocab_size, word_embed_size)
        if self.tanh_flag:
            self.tanh = nn.Tanh()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(word_embed_size, num_head, hidden_size, 0.1, batch_first=True),
            num_layers=num_layers,
        )
        self.fc = nn.Linear(word_embed_size, embed_size)

    def forward(self, qst_vec):
        if self.vecize:
            qst_vec = self.word2vec(qst_vec)
        if self.tanh_flag:
            qst_vec = self.tanh(qst_vec)
        qst_feature = self.transformer(qst_vec)
        if self.tanh_flag:
            qst_feature = self.tanh(qst_feature)
        qst_feature = self.fc(qst_feature)
        return qst_feature


class Learnable_Feature_Fusion(nn.Module):
    def __init__(self, input_size, embed_size, sigmoid=True, att_neck=False):
        super(Learnable_Feature_Fusion, self).__init__()
        self.att_neck = att_neck
        if self.att_neck:
            self.squeeze_att_q = nn.MultiheadAttention(input_size, 2, batch_first=True, dropout=0.1)
            self.squeeze_att_x = nn.MultiheadAttention(input_size, 2, batch_first=True, dropout=0.1)
        self.sigmoid_flag = sigmoid
        self.fc_att = nn.Sequential(
            nn.Linear(input_size, embed_size),
            nn.Linear(embed_size, input_size)
        )
        if self.sigmoid_flag:
            self.sig = nn.Sigmoid()

    def forward(self, x, qst_vec):
        if self.att_neck:
            qst_vec = torch.mean(self.squeeze_att_q(qst_vec, qst_vec, qst_vec)[0], dim=1)
            x = torch.mean(self.squeeze_att_x(x, x, x)[0], dim=1)
        else:
            qst_vec = torch.mean(qst_vec, dim=1)
            x = torch.mean(x, dim=1)
        if self.sigmoid_flag:
            qst_vec = self.sig(qst_vec)

        outs = x * qst_vec
        return outs



class IWLA_Model(nn.Module):

    def __init__(self, opt):
        super(IWLA_Model, self).__init__()

        self.opt = opt


        # question
        self.question_encoder = Question_Encoder(
                qst_vocab_size=self.opt.qst_vocab_size[0],
                word_embed_size=self.opt.word_embed_size[0],
                embed_size=self.opt.embed_size[0],
                num_layers=self.opt.num_layers[0],
                hidden_size=self.opt.hidden_size[0],
                num_head=self.opt.num_head[0],
                vecize=self.opt.vecize[0],
                tanh_flag=self.opt.tanh_flag[0],
            )

        self.question_encoders = nn.ModuleList([
            Question_Encoder(
                qst_vocab_size=self.opt.qst_vocab_size[i],
                word_embed_size=self.opt.word_embed_size[i],
                embed_size=self.opt.embed_size[i],
                num_layers=self.opt.num_layers[i],
                hidden_size=self.opt.hidden_size[i],
                num_head=self.opt.num_head[i],
                vecize=self.opt.vecize[i],
                tanh_flag=self.opt.tanh_flag[i],
            )
            for i in range(1, self.opt.question_layer)])

        self.swin = timm.create_model('swinv2_tiny_window8_256', pretrained=True)


        self.yolo_emb = nn.Conv2d(640, 768, 1, bias=False)

        ### ------------> for swin
        hidden_list = []
        for idx_layer, my_blk in enumerate(self.swin.layers):
            for blk in my_blk.blocks:
                hidden_d_size = blk.norm1.normalized_shape[0]
                hidden_list.append(hidden_d_size)
        ### <--------------
        self.audio_ques_cal_blocks = nn.ModuleList([
            CAL(
                output_dim=hidden_list[i],
                use_bn=self.opt.is_bn,
                is_before_layernorm=self.opt.is_before_layernorm,
                is_post_layernorm=self.opt.is_post_layernorm,
                forget_gate=self.opt.forget_gate,
                type="q",
                q_emb_dim=self.opt.q_dims[i],
                prumerge=self.opt.prumerge[i]
            )
            for i in range(self.opt.question_layer-1)])

        self.vis_ques_cal_blocks = nn.ModuleList([
            CAL(
                output_dim=hidden_list[i],
                use_bn=self.opt.is_bn,
                is_before_layernorm=self.opt.is_before_layernorm,
                is_post_layernorm=self.opt.is_post_layernorm,
                forget_gate=self.opt.forget_gate,
                type="q",
                q_emb_dim=self.opt.q_dims[i],
                prumerge=self.opt.prumerge[i]
            )
            for i in range(self.opt.question_layer-1)])

        self.audio_cal_blocks_p0 = nn.ModuleList([
            CAL(
                output_dim=hidden_list[i],
                use_bn=self.opt.is_bn,
                is_before_layernorm=self.opt.is_before_layernorm,
                is_post_layernorm=self.opt.is_post_layernorm,
                forget_gate=self.opt.forget_gate,
            )
            for i in range(self.opt.question_layer-1)])

        self.vis_cal_blocks_p0 = nn.ModuleList([
            CAL(
                output_dim=hidden_list[i],
                use_bn=self.opt.is_bn,
                is_before_layernorm=self.opt.is_before_layernorm,
                is_post_layernorm=self.opt.is_post_layernorm,
                forget_gate=self.opt.forget_gate,
            )
            for i in range(self.opt.question_layer-1)])

        self.audio_cal_blocks_p1 = nn.ModuleList([
            CAL(
                output_dim=hidden_list[i],
                use_bn=self.opt.is_bn,
                is_before_layernorm=self.opt.is_before_layernorm,
                is_post_layernorm=self.opt.is_post_layernorm,
                forget_gate=self.opt.forget_gate
            )
            for i in range(self.opt.question_layer-1, len(hidden_list))])

        self.vis_cal_blocks_p1 = nn.ModuleList([
            CAL(
                output_dim=hidden_list[i],
                use_bn=self.opt.is_bn,
                is_before_layernorm=self.opt.is_before_layernorm,
                is_post_layernorm=self.opt.is_post_layernorm,
                forget_gate=self.opt.forget_gate
            )
            for i in range(self.opt.question_layer-1, len(hidden_list))])

        self.audio_cal_blocks_p2 = nn.ModuleList([
            CAL(
                output_dim=hidden_list[i],
                use_bn=self.opt.is_bn,
                is_before_layernorm=self.opt.is_before_layernorm,
                is_post_layernorm=self.opt.is_post_layernorm,
                forget_gate=self.opt.forget_gate
            )
            for i in range(self.opt.question_layer-1, len(hidden_list))])

        self.vis_cal_blocks_p2 = nn.ModuleList([
            CAL(
                output_dim=hidden_list[i],
                use_bn=self.opt.is_bn,
                is_before_layernorm=self.opt.is_before_layernorm,
                is_post_layernorm=self.opt.is_post_layernorm,
                forget_gate=self.opt.forget_gate
            )
            for i in range(self.opt.question_layer-1, len(hidden_list))])

        self.outs_ques = Question_Encoder(
                qst_vocab_size=self.opt.qst_vocab_size[-1],
                word_embed_size=self.opt.word_embed_size[-1],
                embed_size=self.opt.embed_size[-1],
                num_layers=self.opt.num_layers[-1],
                hidden_size=self.opt.hidden_size[-1],
                num_head=self.opt.num_head[-1],
                vecize=self.opt.vecize[-1],
                tanh_flag=self.opt.tanh_flag[-1]
            )

        self.outs_aud_cal = CAL(
                output_dim=768,
                use_bn=self.opt.is_bn,
                is_before_layernorm=self.opt.is_before_layernorm,
                is_post_layernorm=self.opt.is_post_layernorm,
                forget_gate=self.opt.forget_gate,
                type="q",
                q_emb_dim=self.opt.q_dims[-1]
            )

        self.outs_vis_cal = CAL(
                output_dim=768,
                use_bn=self.opt.is_bn,
                is_before_layernorm=self.opt.is_before_layernorm,
                is_post_layernorm=self.opt.is_post_layernorm,
                forget_gate=self.opt.forget_gate,
                type="q",
                q_emb_dim=self.opt.q_dims[-1]
            )

        # self.feature_map_seq_a = nn.Sequential(
        #     nn.Conv2d(768, 768, 7, 1, 0),
        #     nn.Conv2d(768, 768, 2, 1, 0),
        # )
        #
        # self.feature_map_seq_v = nn.Sequential(
        #     nn.Conv2d(768, 768, 7, 1, 0),
        #     nn.Conv2d(768, 768, 2, 1, 0),
        # )

        self.question_encoder_yolo = Question_Encoder(
                qst_vocab_size=self.opt.w_qst_vocab_size,
                word_embed_size=self.opt.w_word_embed_size,
                embed_size=self.opt.w_embed_size,
                num_layers=self.opt.w_num_layers,
                hidden_size=self.opt.w_hidden_size,
                num_head=self.opt.w_num_head,
                vecize=self.opt.w_vecize,
                tanh_flag=self.opt.w_tanh_flag,
            )

        self.question_encoder_diff_a = Question_Encoder(
                qst_vocab_size=self.opt.w_qst_vocab_size,
                word_embed_size=self.opt.w_word_embed_size,
                embed_size=self.opt.w_embed_size,
                num_layers=self.opt.w_num_layers,
                hidden_size=self.opt.w_hidden_size,
                num_head=self.opt.w_num_head,
                vecize=self.opt.w_vecize,
                tanh_flag=self.opt.w_tanh_flag,
            )

        self.question_encoder_diff_v = Question_Encoder(
                qst_vocab_size=self.opt.w_qst_vocab_size,
                word_embed_size=self.opt.w_word_embed_size,
                embed_size=self.opt.w_embed_size,
                num_layers=self.opt.w_num_layers,
                hidden_size=self.opt.w_hidden_size,
                num_head=self.opt.w_num_head,
                vecize=self.opt.w_vecize,
                tanh_flag=self.opt.w_tanh_flag,
            )

        self.question_encoder_sep_a = Question_Encoder(
                qst_vocab_size=self.opt.w_qst_vocab_size,
                word_embed_size=self.opt.w_word_embed_size,
                embed_size=self.opt.w_embed_size,
                num_layers=self.opt.w_num_layers,
                hidden_size=self.opt.w_hidden_size,
                num_head=self.opt.w_num_head,
                vecize=self.opt.w_vecize,
                tanh_flag=self.opt.w_tanh_flag,
            )

        self.question_encoder_sep_v = Question_Encoder(
                qst_vocab_size=self.opt.w_qst_vocab_size,
                word_embed_size=self.opt.w_word_embed_size,
                embed_size=self.opt.w_embed_size,
                num_layers=self.opt.w_num_layers,
                hidden_size=self.opt.w_hidden_size,
                num_head=self.opt.w_num_head,
                vecize=self.opt.w_vecize,
                tanh_flag=self.opt.w_tanh_flag,
            )

        self.outs_yolo_cal = CAL_yolo(
                output_dim=768,
                use_bn=self.opt.is_bn,
                is_before_layernorm=self.opt.is_before_layernorm,
                is_post_layernorm=self.opt.is_post_layernorm,
                forget_gate=self.opt.forget_gate,
            )

        self.outs_diff_a_cal = CAL_weak_label(
                output_dim=768,
                use_bn=self.opt.is_bn,
                is_before_layernorm=self.opt.is_before_layernorm,
                is_post_layernorm=self.opt.is_post_layernorm,
                forget_gate=self.opt.forget_gate,
                dim=6
            )

        self.outs_diff_v_cal = CAL_weak_label(
                output_dim=768,
                use_bn=self.opt.is_bn,
                is_before_layernorm=self.opt.is_before_layernorm,
                is_post_layernorm=self.opt.is_post_layernorm,
                forget_gate=self.opt.forget_gate,
                dim=5
            )

        self.outs_sep_a_cal = CAL_weak_label(
                output_dim=768,
                use_bn=self.opt.is_bn,
                is_before_layernorm=self.opt.is_before_layernorm,
                is_post_layernorm=self.opt.is_post_layernorm,
                forget_gate=self.opt.forget_gate,
                dim=6
            )

        self.outs_sep_v_cal = CAL_weak_label(
                output_dim=768,
                use_bn=self.opt.is_bn,
                is_before_layernorm=self.opt.is_before_layernorm,
                is_post_layernorm=self.opt.is_post_layernorm,
                forget_gate=self.opt.forget_gate,
                dim=5
            )

        if self.opt.llf_flag:
            self.llf_g_a = Learnable_Feature_Fusion(768, 1024, att_neck=self.opt.att_neck)
            self.llf_g_v = Learnable_Feature_Fusion(768, 1024, att_neck=self.opt.att_neck)
            self.llf_yolo = Learnable_Feature_Fusion(768, 1024, att_neck=False)
            self.llf_diff_a = Learnable_Feature_Fusion(768, 1024, att_neck=self.opt.att_neck)
            self.llf_diff_v = Learnable_Feature_Fusion(768, 1024, att_neck=self.opt.att_neck)
            self.llf_sep_a = Learnable_Feature_Fusion(768, 1024, att_neck=self.opt.att_neck)
            self.llf_sep_v = Learnable_Feature_Fusion(768, 1024, att_neck=self.opt.att_neck)

            self.outs_fc = nn.Sequential(
                nn.Linear(768, 512),
                nn.GELU(),
                nn.Linear(512, 42)
            )
        else:
            self.outs_fc = nn.Sequential(
                nn.Linear(768 * 7, 512),
                nn.GELU(),
                nn.Linear(512, 42)
            )

    def random_mask_patch(self, patch):
        B, L, P = patch.shape
        num_masks = int(P * self.opt.random_mask_rate)
        indices = torch.randperm(P)[:num_masks]
        patch[:, :, indices] = 0
        return patch

    def random_mask_time(self, seq):
        T = seq.shape[1]
        num_masks = int(T * self.opt.random_mask_rate)
        indices = torch.randperm(T)[num_masks:]
        r_seq = seq[:, indices]
        return r_seq



    def forward(self, audio, visual, question, yolo, diff_a, diff_v, sep_a, sep_v, stage='eval'):
        '''
            input question shape:    [B, T]
            input audio shape:       [B, T, L, D]
            input visual shape: [B, T, C, H, W]
            input yolo shape: [B, Cy, Hy, Wy]
            input diff_a shape: [B, T, Cd]
            input diff_v shape: [B, T, Cd]
            input sep_a shape: [B, T, Cd]
            input sep_v shape: [B, T, Cd]
        '''

        if not self.opt.random_mask:
            bs, t, c, h, w = visual.shape

            audio = repeat(audio, 'b t len dim -> b t c len dim', c=3)
            audio = rearrange(audio, 'b t c w h -> (b t) c w h')

            ###### ---------------->
            f_a = self.swin.patch_embed(audio)

            visual = rearrange(visual, 'b t c w h -> (b t) c w h')
            f_v = self.swin.patch_embed(visual)
        else:
            if self.opt.mask_target_type != "visual" and self.opt.random_type != "patch":
                audio = self.random_mask_time(audio)
            if self.opt.mask_target_type != "audio" and self.opt.random_type != "patch":
                visual = self.random_mask_time(visual)

            bs, t, c, h, w = visual.shape

            audio = repeat(audio, 'b t len dim -> b t c len dim', c=3)
            audio = rearrange(audio, 'b t c w h -> (b t) c w h')

            ###### ---------------->
            f_a = self.swin.patch_embed(audio)
            if self.opt.mask_target_type != "visual" and self.opt.random_type != "time":
                f_a = self.random_mask_patch(f_a)

            visual = rearrange(visual, 'b t c w h -> (b t) c w h')
            f_v = self.swin.patch_embed(visual)
            if self.opt.mask_target_type != "audio" and self.opt.random_type != "time":
                f_v = self.random_mask_patch(f_v)
        idx_layer = 0

        ## question features
        qst_feature = self.question_encoder(question)
        qst_yolo = self.question_encoder_yolo(question)
        qst_diff_a = self.question_encoder_diff_a(question)
        qst_diff_v = self.question_encoder_diff_v(question)
        qst_sep_a = self.question_encoder_sep_a(question)
        qst_sep_v = self.question_encoder_sep_v(question)


        for _, my_blk in enumerate(self.swin.layers):

            for blk in my_blk.blocks:
                if idx_layer < self.opt.question_layer-1:
                    f_a_res = self.audio_ques_cal_blocks[idx_layer](f_a.permute(0, 2, 1).unsqueeze(-1), qst_feature)
                    f_v_res = self.vis_ques_cal_blocks[idx_layer](f_v.permute(0, 2, 1).unsqueeze(-1), qst_feature)
                    qst_feature = self.question_encoders[idx_layer](qst_feature)
                else:
                    f_a_res = self.audio_cal_blocks_p1[idx_layer-self.opt.question_layer+1](f_a.permute(0, 2, 1).unsqueeze(-1),
                                                                      f_v.permute(0, 2, 1).unsqueeze(-1))
                    f_v_res = self.vis_cal_blocks_p1[idx_layer-self.opt.question_layer+1](f_v.permute(0, 2, 1).unsqueeze(-1),
                                                                    f_a.permute(0, 2, 1).unsqueeze(-1))
                f_v = f_v + blk.drop_path1(blk.norm1(blk._attn(f_v)))
                f_v = f_v + f_v_res.squeeze(-1).permute(0, 2, 1)

                f_a = f_a + blk.drop_path1(blk.norm1(blk._attn(f_a)))
                f_a = f_a + f_a_res.squeeze(-1).permute(0, 2, 1)

                if idx_layer < self.opt.question_layer-1:
                    f_a_res = self.audio_cal_blocks_p0[idx_layer](f_a.permute(0, 2, 1).unsqueeze(-1),
                                                                  f_v.permute(0, 2, 1).unsqueeze(-1))
                    f_v_res = self.vis_cal_blocks_p0[idx_layer](f_v.permute(0, 2, 1).unsqueeze(-1),
                                                                f_a.permute(0, 2, 1).unsqueeze(-1))
                else:
                    f_a_res = self.audio_cal_blocks_p2[idx_layer-self.opt.question_layer+1](f_a.permute(0, 2, 1).unsqueeze(-1),
                                                                      f_v.permute(0, 2, 1).unsqueeze(-1))
                    f_v_res = self.vis_cal_blocks_p2[idx_layer-self.opt.question_layer+1](f_v.permute(0, 2, 1).unsqueeze(-1),
                                                                    f_a.permute(0, 2, 1).unsqueeze(-1))

                f_v = f_v + blk.drop_path2(blk.norm2(blk.mlp(f_v)))
                f_v = f_v + f_v_res.squeeze(-1).permute(0, 2, 1)

                f_a = f_a + blk.drop_path2(blk.norm2(blk.mlp(f_a)))
                f_a = f_a + f_a_res.squeeze(-1).permute(0, 2, 1)

                idx_layer = idx_layer + 1
            #####
            f_v = my_blk.downsample(f_v)
            f_a = my_blk.downsample(f_a)

        f_q = self.outs_ques(qst_feature)
        f_v = self.swin.norm(f_v)
        f_a = self.swin.norm(f_a)

        outs_v = f_v
        outs_a = f_a
        # outs_v = self.outs_vis_cal(f_v.permute(0, 2, 1).unsqueeze(-1), f_q).squeeze(-1).permute(0, 2, 1)
        # outs_a = self.outs_vis_cal(f_a.permute(0, 2, 1).unsqueeze(-1), f_q).squeeze(-1).permute(0, 2, 1)

        outs_v = rearrange(outs_v, 'bt (h w) c -> bt c h w', h=8, w=8)
        outs_a = rearrange(outs_a, 'bt (h w) c -> bt c h w', h=8, w=8)
        # outs_v = self.feature_map_seq_v(outs_v).squeeze(-1).squeeze(-1)
        # outs_a = self.feature_map_seq_a(outs_a).squeeze(-1).squeeze(-1)
        outs_v = torch.mean(outs_v, dim=[2, 3])
        outs_a = torch.mean(outs_a, dim=[2, 3])

        outs_v = rearrange(outs_v, '(b t) c -> b t c', b=bs, t=t)
        outs_a = rearrange(outs_a, '(b t) c -> b t c', b=bs, t=t)

        yolo = self.yolo_emb(yolo)
        outs_yolo = self.outs_yolo_cal(torch.mean(yolo, dim=[2, 3]).unsqueeze(-1).permute(0, 2, 1).unsqueeze(-1), qst_yolo).squeeze(-1).permute(0, 2, 1)
        outs_diff_a = self.outs_diff_a_cal(diff_a.unsqueeze(-1), qst_diff_a).squeeze().permute(0, 2, 1)
        outs_diff_v = self.outs_diff_v_cal(diff_v.unsqueeze(-1), qst_diff_v).squeeze().permute(0, 2, 1)
        outs_sep_a = self.outs_sep_a_cal(sep_a.unsqueeze(-1), qst_sep_a).squeeze().permute(0, 2, 1)
        outs_sep_v = self.outs_sep_v_cal(sep_v.unsqueeze(-1), qst_sep_v).squeeze().permute(0, 2, 1)

        if self.opt.llf_flag:
            outs = (self.llf_g_a(outs_a, f_q) + self.llf_g_v(outs_v, f_q)
                    + self.llf_yolo(outs_yolo, qst_yolo)
                    + self.llf_diff_a(outs_diff_a, qst_diff_a)
                    + self.llf_diff_v(outs_diff_v, qst_diff_v)
                    + self.llf_sep_a(outs_sep_a, qst_sep_a)
                    + self.llf_sep_v(outs_sep_v, qst_sep_v))

        else:
            outs = torch.cat([torch.mean(outs_v, dim=1), torch.mean(outs_a, dim=1)], dim=1)
        outs = self.outs_fc(outs)

        return outs

