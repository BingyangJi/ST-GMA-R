import torch.nn as nn
from Layers.GAT import STGAT
from Layers.Mamba2 import Mamba2EncoderLayer
from Layers.ResFusionNet import ResFusionNet
from Layers.Router import Router
from Layers.Fusion_Head import FusionHead

class ST_GMA(nn.Module):
    def __init__(self, in_features, out_features, hidden_features, T, num_heads, num_layers, dropout, edge_dim=1):
        super(ST_GMA, self).__init__()
        self.stgat = STGAT(in_channel=in_features, hidden_dim=hidden_features,
                           num_layers=num_layers, num_heads=num_heads, dropout=dropout, edge_dim=edge_dim)

        self.mamba2encoder = Mamba2EncoderLayer(d_model=num_heads*hidden_features,
        d_ff=hidden_features,
        d_state=hidden_features,
        d_conv=4,
        expand=2,
        headdim=64)

        hidden_fla_dim = hidden_features * num_heads * T

        self.ffn = nn.Sequential(
            nn.Linear(hidden_fla_dim, out_features),
            nn.ReLU()
        )
        self.flatten = nn.Flatten()

    def forward(self, x, edge_index, edge_attr, batch, T, N):
        x = self.stgat(x, edge_index, edge_attr, batch, T, N)

        x = self.mamba2encoder(x)

        x = self.flatten(x)
        x = self.ffn(x)

        return x


class ResFusionNet(nn.Module):
    def __init__(self, input_dim, num_classes=2, include_top=True, out_features=1024):
        super(ResFusionNet, self).__init__()
        self.h2dnet = ResFusionNet(input_dim=input_dim, num_classes=num_classes, include_top=include_top)
        self.activation = nn.ReLU()
        self.ffn = nn.Linear(2048, out_features)
    def forward(self, x):
        x = self.h2dnet(x)
        x = self.ffn(x)

        return x

class ST_GMA_R(nn.Module):
    def __init__(self, in_features, out_features, hidden_features, T, num_heads, N,
                 num_layers, dropout, edge_dim=1, num_classes=2, include_top=True):
        super(ST_GMA_R, self).__init__()
        self.branch_1 = ST_GMA(in_features, out_features, hidden_features, T, num_heads, num_layers, dropout, edge_dim=edge_dim)
        self.branch_2 = ResFusionNet(input_dim=N*2, num_classes=num_classes, include_top=include_top, out_features=out_features)
        self.router_model = Router(in_feature=N*T*in_features, num_hidden=out_features, bias=False)
        self.fusionhead =FusionHead(in_dim=out_features, target_dim=num_classes)
        self.activation = nn.LogSoftmax(dim=-1)


    def forward(self, x_1, x_2, edge_index, edge_attr, batch, T, N):
        out_1 = self.branch_1(x_1, edge_index, edge_attr, batch, T, N)
        out_2 = self.branch_2(x_2)
        w_1, w_2 = self.router_model(x_2[:,:N,:,:])
        out = self.fusionhead(out_1, out_2, w_1, w_2)
        out = self.activation(out)

        return out
