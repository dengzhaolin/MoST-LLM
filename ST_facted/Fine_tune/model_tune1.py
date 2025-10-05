# This project uses Llama 3.2, a foundational large language model developed by Meta.
# Llama 3.2 is licensed under the Llama 3.2 Community License,
# Copyright © Meta Platforms, Inc. All Rights Reserved.


import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM,LlamaConfig
from torch.utils.data import Dataset

from math import sqrt


class TrafficInstructionDataset(torch.utils.data.Dataset):
    def __init__(self, embeddings, raw_data):
        """
        Args:
            embeddings: Tensor of shape [dataset_size, input_length, num_regions, embed_dim].
            raw_data: Tensor of shape [dataset_size, input_length, num_regions, grid_dim, grid_dim].
        """
        self.embeddings = embeddings
        self.raw_data = raw_data

    def __len__(self):
        return self.embeddings.shape[0]

    def __getitem__(self, idx):
        embedding = self.embeddings[idx].clone().detach().float()  # [input_length, num_regions, embed_dim]
        raw_data = self.raw_data[idx].clone().detach().float()  # [input_length, num_regions, grid_dim, grid_dim]
        return {"embedding": embedding, "raw_data": raw_data}


class LLMFineTuner(nn.Module):
    def __init__(self,model_name, tokenizer_name, device='cuda'):
        super(LLMFineTuner, self).__init__()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name)

        self.embedding_to_hidden =nn.Sequential(nn.Linear(192, 1536)) # args.hidden_channels*modality_channel


        #self.fustion=GatedFusionBlock(1536, 192,1536)

        self.fustion=Encoder_PCA(192,1536)

        # self.word_embeddings = self.model.get_input_embeddings().weight
        # self.vocab_size = self.word_embeddings.shape[0]  # 150000
        # self.num_tokens=98
        # self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)
        #
        #self.reprogramming_layer=ReprogrammingLayer(192, 8,d_llm=1536)
        #
        # self.device = device
        #
        # self.channels=48
        #
        # self.sag=SAG(1536, 50, 1536,0.1)


        self.device = device

        for param in self.model.parameters():
            param.requires_grad = False

        #self._unfreeze_last_layers(1)
    def _unfreeze_last_layers(self, num_layers_to_unfreeze):
        total_layers = len(self.model.model.layers)
        for i in range(total_layers):
            for param in self.model.model.layers[i].parameters():
                param.requires_grad = i >= total_layers - num_layers_to_unfreeze

    # def _unfreeze_last_layers(self, num_layers_to_unfreeze):
    #     for i in range(num_layers_to_unfreeze):
    #         for param in self.model.model.layers[i].parameters():
    #             param.requires_grad = True

    def forward(self, inputs_embeds,prompt_embedding):

        #source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)

        #inputs_embeds = self.reprogramming_layer(inputs_embeds,  prompt_embedding,  prompt_embedding)


        #enc_out , _ =self.sag.encode(enc_out)

        ##inputs_embeds=self.fustion(inputs_embeds,prompt_embedding)

        inputs_embeds=self.embedding_to_hidden(inputs_embeds)

        inputs_embeds1=torch.cat([inputs_embeds, prompt_embedding], dim=1)


        outputs = self.model(inputs_embeds=inputs_embeds1,output_hidden_states=True).hidden_states[-1]

        outputs=outputs[:, :inputs_embeds.shape[1], :]




        #outputs=self.sag.decode(outputs,source_embeddings.repeat(inputs_embeds.shape[0], 1, 1))

        return outputs




class ModalExpert(nn.Module):
    def __init__(self, in_channels, out_channels=16):
        """
        模态专家模块
        Args:
            in_channels: 输入通道数 (512)
            out_channels: 输出通道数 (默认3)
        """
        super(ModalExpert, self).__init__()
        self.predictor = nn.Sequential(
            nn.Conv3d(in_channels, 128, kernel_size=(1, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(128, 48, kernel_size=(1, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(48, out_channels, kernel_size=(1, 1, 1)))


    def forward(self, x):
        """输入: (b, c, m, n, 1)"""
        return self.predictor(x)


class EfficientMoE(nn.Module):
    def __init__(self, in_channels, num_modals=4, num_experts=4):
        """
        高效的模态混合专家模型
        Args:
            in_channels: 输入通道数 (512)
            num_modals: 模态数量 (4)
            num_experts: 专家数量 (4)
        """
        super(EfficientMoE, self).__init__()
        self.num_experts = num_experts
        self.num_modals = num_modals
        self.in_channels = in_channels

        # 共享专家池
        self.experts = nn.ModuleList([
            ModalExpert(in_channels) for _ in range(num_experts)
        ])

        # 高效门控网络
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool3d((num_modals, 1, 1)),  # 只聚合空间和时间维度
            nn.Conv3d(in_channels, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv3d(128, num_experts, kernel_size=1),
            nn.Flatten(start_dim=1)  # 展平以便后续处理
        )

    def forward(self, x):
        """输入: (b, c, m, n, 1) = (batch, 512, 4, 98, 1)"""
        b, _, m, n, t = x.size()

        # 计算门控权重
        gate_scores = self.gate(x)  # (b, num_experts * num_modals)

        gate_scores = gate_scores.view(b, self.num_modals, self.num_experts)  # (b, m, experts)
        weights = F.softmax(gate_scores, dim=2)  # (b, m, experts)

        # 获取专家输出
        expert_outputs = []
        for expert in self.experts:
            expert_out = expert(x)  # (b, 3, m, n, 1)
            expert_outputs.append(expert_out)

        # 加权组合专家输出 (每个模态独立选择)
        output = torch.zeros_like(expert_outputs[0])  # (b, 3, m, n, 1)

        # 对每个模态单独处理
        for modal_idx in range(self.num_modals):
            # 当前模态的权重: (b, experts)
            modal_weights = weights[:, modal_idx, :]

            # 对每个专家进行加权求和
            weighted_sum = 0
            for exp_idx in range(self.num_experts):
                weight_factor = modal_weights[:, exp_idx].view(b, 1, 1, 1, 1)  # (b, 1, 1, 1, 1)
                weighted_sum += expert_outputs[exp_idx][:, :, modal_idx:modal_idx + 1] * weight_factor

            output[:, :, modal_idx] = weighted_sum.squeeze(2)

        return output


# 简化封装
class ModalMoE(nn.Module):
    def __init__(self, in_channels=384, num_modals=4, num_experts=6):
        super(ModalMoE, self).__init__()
        self.moe = EfficientMoE(in_channels, num_modals, num_experts)

    def forward(self, x):
        return self.moe(x)


class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        super(ReprogrammingLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection =nn.Sequential(nn.Linear(d_keys * n_heads, d_llm),
                                           nn.ReLU())
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape
        B,S, _ = source_embedding.shape
        H = self.n_heads

        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(B,S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(B,S, H, -1)

        out = self.reprogramming(target_embedding, source_embedding, value_embedding)


        out = out.reshape(B, L, -1)

        return self.out_projection(out)
    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape

        scale = 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", target_embedding, source_embedding)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        reprogramming_embedding = torch.einsum("bhls,bshe->blhe", A, value_embedding)

        return reprogramming_embedding


class GatedFusionBlock(nn.Module):
    def __init__(self, sem_input_dim, x_input_dim, gate_output_dim):
        super(GatedFusionBlock, self).__init__()
        self.gate_sem = nn.Linear(sem_input_dim + x_input_dim, gate_output_dim)
        self.gate_x = nn.Linear(x_input_dim, gate_output_dim)
        self.out1 = nn.Linear(gate_output_dim, gate_output_dim)
        self.out2 = nn.Linear(gate_output_dim, gate_output_dim)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        self.gate_output_dim = gate_output_dim

    def forward(self, x, sem):
        gate_sem = self.gate_sem(torch.cat((sem, x), dim=-1))
        gate_sem = self.sigmoid(gate_sem)
        output = self.gate_x(x)
        output = gate_sem * output
        output = self.out1(output)
        output = self.relu(output)
        output = self.out2(output)

        return output






class Prediction(nn.Module):
    def __init__(self, in_channels, out_channels=3):
        """
        模态专家模块
        Args:
            in_channels: 输入通道数 (512)
            out_channels: 输出通道数 (默认3)
        """
        super(Prediction, self).__init__()
        self.predictor = nn.Sequential(
            nn.Conv3d(in_channels, 128, kernel_size=(1, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(128, 48, kernel_size=(1, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(48, out_channels, kernel_size=(1, 1, 1)))


    def forward(self, x):
        """输入: (b, c, m, n, 1)"""
        return self.predictor(x)


# model=Prediction(384)
#
# x=torch.rand(64,384,4,98,1)
#
# print(model(x).shape)
class Encoder_PCA(nn.Module):
    def __init__(self, input_dim, hidden_dim=768, num_heads=12, num_encoder_layers=1):
        super(Encoder_PCA, self).__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads)



    def forward(self, x,word_embedding):
        B = x.shape[0]

        if word_embedding.ndim == 2:
            word_embedding = word_embedding.repeat(B, 1, 1)
        elif word_embedding.shape[0] != B:
            word_embedding = word_embedding[0].repeat(B, 1, 1)

        x = self.linear(x)

        x = self.transformer_encoder(x.transpose(0, 1)).transpose(0, 1)


        q = x.transpose(0, 1)

        k = v = word_embedding.transpose(0, 1)
        x, _ = self.cross_attention(q, k, v)

        x = x.transpose(0, 1)

        return x


# model=Encoder_PCA(192,torch.rand(16,98,1536),1536)
#
# x=torch.rand(16,98,192)
#
# y=model(x)
# print(y.shape)