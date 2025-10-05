import torch
import torch.nn as nn
import torch.nn.functional as F


class NTN(nn.Module):
    def __init__(self, input_dim, feature_map_dim):
        super(NTN, self).__init__()
        self.interaction_dim = feature_map_dim
        self.V = nn.Parameter(torch.randn(feature_map_dim, input_dim * 2, 1))
        nn.init.xavier_normal_(self.V)
        self.W1 = nn.Parameter(torch.randn(feature_map_dim, input_dim, input_dim))
        nn.init.xavier_normal_(self.W1)
        self.W2 = nn.Parameter(torch.randn(feature_map_dim, input_dim, input_dim))
        nn.init.xavier_normal_(self.W2)
        self.b = nn.Parameter(torch.zeros(feature_map_dim))

    def forward(self, x_1, x_2):
        feature_map = []
        for i in range(self.interaction_dim):
            x_1_t = torch.matmul(x_1, self.W1[i])
            x_2_t = torch.matmul(x_2, self.W2[i])
            part1 = torch.cosine_similarity(x_1_t, x_2_t, dim=-1).unsqueeze(dim=-1)
            part2 = torch.matmul(torch.cat([x_1, x_2], dim=-1), self.V[i])
            fea = part1 + part2 + self.b[i]
            feature_map.append(fea)
        return torch.relu(torch.cat(feature_map, dim=-1))


class Decoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(Decoder, self).__init__()
        self.end_conv_1 = nn.Conv2d(in_dim, hidden_dim, kernel_size=(1, 1))
        self.end_conv_2 = nn.Conv2d(hidden_dim, out_dim, kernel_size=(1, 1))

    def forward(self, x):
        x = F.relu(self.end_conv_1(x))
        return self.end_conv_2(x)


class Embed_Trans(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(Embed_Trans, self).__init__()
        self.conv_1 = nn.Conv2d(in_dim, hidden_dim, kernel_size=(1, 1))
        self.conv_2 = nn.Conv2d(hidden_dim, out_dim, kernel_size=(1, 1))

    def forward(self, x):
        x = F.relu(self.conv_1(x))
        return self.conv_2(x)


class memda_net(nn.Module):
    def __init__(self, encoder='gwn', num_nodes=98, num_modes=4, out=1):
        super(memda_net, self).__init__()
        self.num_nodes = num_nodes
        self.num_modes = num_modes
        encoder_dim = 256
        mem_num = 20
        mem_dim = 32
        ntn_dim = 32
        ntn_k = 5

        # Encoder
        if encoder == 'gwn':
            self.encoder = nn.Linear(256, encoder_dim)
        else:
            raise NameError("Encoder Undefined")

        # Memory
        self.pattern_memory = nn.Parameter(torch.randn(mem_num, mem_dim))
        nn.init.xavier_normal_(self.pattern_memory)
        self.mem_proj1 = nn.Linear(encoder_dim, mem_dim)
        self.mem_proj2 = nn.Linear(mem_dim, encoder_dim)

        # Drift modules (cross-mode)
        self.ntn_dim = ntn_dim
        self.ntn_k = ntn_k
        # all pairwise mode interactions
        self.drift_pairs = nn.ModuleDict()
        for i in range(num_modes):
            for j in range(i + 1, num_modes):
                self.drift_pairs[f"{i}_{j}"] = NTN(ntn_dim, ntn_k)

        self.drift_proj1 = nn.Linear(encoder_dim, ntn_dim)
        self.drift_proj2 = nn.Linear(len(self.drift_pairs) * ntn_k, encoder_dim)

        # Meta weights
        self.meta_W = nn.Linear(encoder_dim, num_modes)
        self.meta_b = nn.Linear(encoder_dim, 1)

        # Decoder
        self.decoder = Decoder(in_dim=encoder_dim, hidden_dim=512, out_dim=16)

    def forward(self, x):
        """
        x: (B, F, N, M) = (64, 256, 98, 4)
        """
        B, F, N, M = x.shape

        # (B, N, M, F)
        hidden_merge = x.permute(0, 2, 3, 1)

        # Memory module
        query = self.mem_proj1(hidden_merge)  # (B, N, M, mem_dim)
        att = torch.softmax(torch.matmul(query, self.pattern_memory.t()), dim=-1)
        res_mem = torch.matmul(att, self.pattern_memory)  # (B, N, M, mem_dim)
        res_mem = self.mem_proj2(res_mem)  # (B, N, M, F)

        # Drift module (pairwise across modes)
        hidden_cts = self.drift_proj1(hidden_merge)  # (B, N, M, ntn_dim)
        drift_results = []
        for key, ntn in self.drift_pairs.items():
            i, j = map(int, key.split("_"))
            drift_results.append(ntn(hidden_cts[:, :, i, :], hidden_cts[:, :, j, :]))
        drift_mtx = torch.cat(drift_results, dim=-1)  # (B, N, total_ntn_k)
        drift_mtx = self.drift_proj2(drift_mtx)  # (B, N, F)

        # Meta-weights across modes
        W = self.meta_W(drift_mtx)  # (B, N, M)
        b = self.meta_b(drift_mtx)  # (B, N, 1)
        W = torch.softmax(W, dim=-1).unsqueeze(1)  # (B,1,N,M)

        # Combine hidden + memory
        hidden = hidden_merge + res_mem  # (B, N, M, F)
        hidden = hidden.permute(0, 3, 1, 2)  # (B, F, N, M)


        # Apply weights across modes
        hidden = torch.einsum("bfni,bnmi->bfni", hidden, W) + b.unsqueeze(1)

        # Decode
        output = self.decoder(hidden).unsqueeze(-1) # (B, out, N, M)
        return output

x=torch.randn(64,48,16,98,4)
model=memda_net()
output=model(x)
print(output.shape)