# This project uses Llama 3.2, a foundational large language model developed by Meta.
# Llama 3.2 is licensed under the Llama 3.2 Community License,
# Copyright © Meta Platforms, Inc. All Rights Reserved.


import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM,LlamaConfig
from torch.utils.data import Dataset


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

        self.embedding_to_hidden = nn.Linear(192, self.model.config.hidden_size)# args.hidden_channels*modality_channel

        self.device = device

        self.channels=48


        self.device = device

        for param in self.model.parameters():
            param.requires_grad = False

        self._unfreeze_last_layers(10)
    def _unfreeze_last_layers(self, num_layers_to_unfreeze):
        total_layers = len(self.model.model.layers)
        for i in range(total_layers):
            for param in self.model.model.layers[i].parameters():
                param.requires_grad = i >= total_layers - num_layers_to_unfreeze

    # def _unfreeze_last_layers(self, num_layers_to_unfreeze):
    #     for i in range(num_layers_to_unfreeze):
    #         for param in self.model.model.layers[i].parameters():
    #             param.requires_grad = True

    def forward(self, inputs_embeds):


        outputs = self.model(inputs_embeds=inputs_embeds,output_hidden_states=True).hidden_states[-1]

        return outputs

class GLU(nn.Module):
    def __init__(self, features, dropout=0.1):
        super(GLU, self).__init__()
        self.conv1 = nn.Conv3d(features, features, (1, 1,1))
        self.conv2 = nn.Conv3d(features, features, (1,1, 1))
        self.conv3 = nn.Conv3d(features, features, (1,1, 1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        out = x1 * torch.sigmoid(x2)
        out = self.dropout(out)
        out = self.conv3(out)
        return out


class MA(nn.Module):
    def __init__(self, channels):
        super(MA, self).__init__()
        self.channels = channels
        self.predictor = nn.Sequential(
            nn.Conv3d(in_channels=512, out_channels=256, kernel_size=(1, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(in_channels=256, out_channels=128, kernel_size=(1, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(in_channels=128, out_channels=channels, kernel_size=(1, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(in_channels=channels, out_channels=3, kernel_size=(1, 1, 1))
        )



    def forward(self, rep):   #  rep  b c m n l   16, 48, 4, 98, 16

        # query = self.Wq(rep).permute(0,3,4,2,1)
        # key = self.Wk(rep).permute(0,3,4,2,1)
        # value = self.Wv(rep).permute(0,3,4,2,1)
        # attention = torch.matmul(query, key.transpose(3, 4))
        # attention /= (self.channels ** 0.5)
        # attention = F.softmax(attention, dim=-1)
        # rep = torch.matmul(attention, value)
        # rep=self.norm(rep)
        # rep = self.FC(rep.permute(0,4,3,1,2)) #16, 1, 4, 98, 3

        rep=self.predictor(rep)

        return rep


class MA1(nn.Module):
    def __init__(self,channels1, channels):
        super(MA1, self).__init__()
        self.channels = channels






        self.Wq = nn.Sequential(
            nn.Linear(channels1, self.channels),
            nn.ReLU())
        self.Wk = nn.Sequential(
            nn.Linear(channels, self.channels),
            nn.ReLU())
        self.Wv = nn.Sequential(
            nn.Linear(self.channels, self.channels),
            nn.ReLU())

        self.FC = nn.Sequential(
            nn.Linear(self.channels, self.channels),
            nn.ReLU())






    def forward(self, rep,rep1):   #  rep  b c m n l   16, 48, 4, 98, 16

        query = self.Wq(rep)
        key = self.Wk(rep1)
        value = self.Wv(rep1)
        attention = torch.matmul(query, key.transpose(-2, -1))
        attention /= (self.channels ** 0.5)
        attention = F.softmax(attention, dim=-1)
        rep = torch.matmul(attention, value)
        rep = self.FC(rep) #16, 1, 4, 98, 3

        return rep

"""

ma=MA1(64)
x=torch.randn(16, 64, 4, 98, 16)
y=ma(x)
print(y.shape)

"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ModalExpert(nn.Module):
    def __init__(self, in_channels, out_channels=3):
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
    def __init__(self, in_channels=384, num_modals=4, num_experts=4):
        super(ModalMoE, self).__init__()
        self.moe = EfficientMoE(in_channels, num_modals, num_experts)

    def forward(self, x):
        return self.moe(x)



