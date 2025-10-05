import torch
import torch.nn as nn
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers import AutoTokenizer
import json
from typing import Dict, Any
import numpy as np
class TemporalEmbedding(nn.Module):
    def __init__(self, time, features):
        super(TemporalEmbedding, self).__init__()

        self.time = time
        # temporal embeddings
        self.time_day = nn.Parameter(torch.empty(time, features))
        nn.init.xavier_uniform_(self.time_day)

        self.time_week = nn.Parameter(torch.empty(7, features))
        nn.init.xavier_uniform_(self.time_week)

    def forward(self, x):
        day_emb = x[..., 1]
        time_day = self.time_day[
            (day_emb[:, -1, :] * self.time).type(torch.LongTensor)
        ]
        time_day = time_day.transpose(1, 2).unsqueeze(-1)

        week_emb = x[..., 2]
        time_week = self.time_week[
            (week_emb[:, -1, :]).type(torch.LongTensor)
        ]
        time_week = time_week.transpose(1, 2).unsqueeze(-1)

        # temporal embeddings
        tem_emb = time_day + time_week
        return tem_emb

class PFA(nn.Module):
    def __init__(self, data_name, gpt_layers=6, U=1):
        super(PFA, self).__init__()
        self.gpt2 = GPT2Model.from_pretrained(
            "/data/dzl2023/LLM-Factory/GPT2", output_attentions=True, output_hidden_states=True
        )

        if(data_name=="BJ"):
            self.description = 'The traffic dataset of Beijing City is a commonly used dataset in traffic prediction problems, mainly including taxi traffic data and bicycle traffic data, there are four modalities.'
        else:
            self.description = 'The traffic dataset of New York City is a commonly used dataset in traffic prediction problems, mainly including taxi traffic data and bicycle traffic data, there are four modalities.'

        self.tokenizer = AutoTokenizer.from_pretrained("/data/dzl2023/LLM-Factory/GPT2")
        self.gpt2.h = self.gpt2.h[:gpt_layers]
        self.U = U

        for layer_index, layer in enumerate(self.gpt2.h):
            for name, param in layer.named_parameters():
                if layer_index < gpt_layers - self.U:
                    if "ln" in name or "wpe" in name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
                else:
                    if "mlp" in name:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True

    def forward(self, x):

        prompts=[]

        # 计算动态图

        for i in range(x.shape[0]):
            sim=torch.matmul(x[i, :,:],x[i, :,:].permute(1,0))

            sim_info = self.tensor_summary(sim, name="sim",  max_nodes=6, precision=4)
            sim_info_json = json.dumps(sim_info, ensure_ascii=False)

            # 若你确实需要插入完整矩阵（可能非常长，不推荐）：
            # sim_full = sim.detach().cpu().to(torch.float32).numpy().tolist()
            # sim_full_json = json.dumps(sim_full)

            # 构造提示词：把动态图描述与 sim 的摘要嵌入
            prompt_ = (
                "<|start_prompt|>"
                f"Dataset description:\n{self.description}\n\n"
                "Dynamic graph description:\n"
                "- Data are organized as a temporal graph sequence. Node set is fixed per batch; edges/weights vary over time.\n"
                "- Node similarity per batch is computed as sim = X @ X^T "
                "(implemented by torch.matmul(x, x.permute(0,2,1))).\n\n"
                f"Similarity summary (for insertion):\n{sim_info_json}\n\n"
                "Task description:\n"
                f"- Forecast the next {16} steps given the previous {16} steps.\n"
                "- Use temporal patterns in features and structural dynamics implied by the similarity matrices.\n"
                "- Do not use any future information beyond time t.\n\n"
                "Output format:\n"
                "- Provide predictions for all target nodes and features for the next 16 steps.\n"
                "<|<end_prompt>|>"
            )

            prompts.append(prompt_)



        tokenized_prompt = self.tokenizer(prompts, return_tensors="pt", max_length=40).to(x.device).input_ids


        prompt_embedding = self.gpt2.get_input_embeddings()(tokenized_prompt)


        inputs_embeds=torch.cat([prompt_embedding, x], dim=1)



        return self.gpt2(inputs_embeds=inputs_embeds).last_hidden_state

    def tensor_summary(self,t: torch.Tensor, name: str = "sim",
                       max_nodes: int = 5, precision: int = 4) -> Dict[str, Any]:
        """
        仅支持二维相似度矩阵 t: [N, N]。
        返回统计信息与左上角子矩阵预览。
        """
        if t is None:
            return {"name": name, "error": "tensor is None"}

        if not isinstance(t, torch.Tensor):
            return {"name": name, "error": f"expect torch.Tensor, got {type(t)}"}

        if t.ndim != 2:
            return {"name": name, "shape": list(t.shape), "error": "expect 2D tensor [N, N]"}

        # 基本信息与统计（detach 以免跟踪梯度）
        t_detached = t.detach()
        N = t_detached.shape[0]

        out: Dict[str, Any] = {
            "name": name,
            "shape": [int(N), int(N)],
            "dtype": str(t_detached.dtype),
            "stats": {
                "min": float(t_detached.min().item()),
                "max": float(t_detached.max().item()),
                "mean": float(t_detached.mean().item()),
                "std": float(t_detached.std().item()),
            },
            "preview": []
        }

        # 子矩阵预览：左上角 n_show × n_show
        n_show = int(min(N, max_nodes))
        sub = t_detached[:n_show, :n_show].cpu().to(torch.float32).numpy()
        sub = np.round(sub, decimals=int(precision)).tolist()
        out["preview"].append({"submatrix": sub})

        return out

class STG_LLM(nn.Module):
    def __init__(
        self,
            data_name,
        input_dim=4,
        channels=16,
        num_nodes=98,
        input_len=16,
        output_len=16,
        llm_layer=6,
        U=1,
        device= "cuda:7"
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.node_dim = channels
        self.input_len = input_len
        self.input_dim = input_dim
        self.output_len = output_len
        self.llm_layer = llm_layer
        self.U = U
        self.device = device


        if num_nodes == 170 or num_nodes == 307:
            time = 288
        elif num_nodes == 98 or num_nodes == 175:
            time = 48

        gpt_channel = 256
        to_gpt_channel = 768

        self.gpt_channel=gpt_channel

        self.Temb = TemporalEmbedding(time, gpt_channel)

        self.node_emb = nn.Parameter(torch.empty(self.num_nodes, gpt_channel))
        nn.init.xavier_uniform_(self.node_emb)

        self.start_conv = nn.Conv2d(
            self.input_dim * self.input_len, gpt_channel, kernel_size=(1, 1)
        )

        # embedding layer
        self.gpt = PFA(data_name, gpt_layers=self.llm_layer, U=self.U)

        self.feature_fusion = nn.Conv2d(
            gpt_channel * 3, to_gpt_channel, kernel_size=(1, 1)
        )

        # regression
        self.regression_layer = nn.Conv2d(
            gpt_channel * 3, self.output_len*self.input_dim, kernel_size=(1, 1)
        )

    # return the total parameters of model
    def param_num(self):
        return sum([param.nelement() for param in self.parameters()])

    def forward(self, history_data):




        input_data = history_data.squeeze(-1).permute(0,3,2,1)
        batch_size, _, num_nodes, _ = input_data.shape


        tem_emb = torch.randn(batch_size,self.gpt_channel,num_nodes,1).to(input_data.device)
        node_emb = []
        node_emb.append(
            self.node_emb.unsqueeze(0)
            .expand(batch_size, -1, -1)
            .transpose(1, 2)
            .unsqueeze(-1)
        )

        input_data = input_data.transpose(1, 2).contiguous()
        input_data = (
            input_data.view(batch_size, num_nodes, -1).transpose(1, 2).unsqueeze(-1)
        )
        input_data = self.start_conv(input_data)

        data_st = torch.cat(
            [input_data] + [tem_emb] + node_emb, dim=1
        )
        data_st = self.feature_fusion(data_st)
        # data_st = F.leaky_relu(data_st)

        data_st = data_st.permute(0, 2, 1, 3).squeeze(-1)




        data_st = self.gpt(data_st)[:,:num_nodes,:]

        data_st = data_st.permute(0, 2, 1).unsqueeze(-1)


        prediction = self.regression_layer(data_st)

        prediction = prediction.reshape(batch_size,self.output_len,self.input_dim,num_nodes,1).permute(0,1,3,2,4)


        return prediction
