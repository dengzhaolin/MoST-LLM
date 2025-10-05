# This project uses Llama 3.2, a foundational large language model developed by Meta.
# Llama 3.2 is licensed under the Llama 3.2 Community License,
# Copyright © Meta Platforms, Inc. All Rights Reserved.


import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM,LlamaConfig


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

class LLMFineTuner_4(nn.Module):
    def __init__(self,model_name, tokenizer_name, device='cuda'):
        super(LLMFineTuner_4, self).__init__()
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

class LLMFineTuner_6(nn.Module):
    def __init__(self,model_name, tokenizer_name, device='cuda'):
        super(LLMFineTuner_6, self).__init__()
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

class LLMFineTuner_8(nn.Module):
    def __init__(self,model_name, tokenizer_name, device='cuda'):
        super(LLMFineTuner_8, self).__init__()
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
    def __init__(self, in_channels=384, num_modals=4, num_experts=8):
        super(ModalMoE, self).__init__()
        self.moe = EfficientMoE(in_channels, num_modals, num_experts)

    def forward(self, x):
        return self.moe(x)



class Prediction(nn.Module):
    def __init__(self, in_channels, out_channels=16, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.predictor = nn.Sequential(
            nn.Conv3d(in_channels, 128, kernel_size=(1, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(128, 48, kernel_size=(1, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(48, out_channels, kernel_size=(1, 1, 1)))

    def forward(self, x):
        return self.predictor(x)






