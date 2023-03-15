import torch
from torch import nn
import torch.nn.functional as F

class CNNText(nn.Module):
    
    def __init__(self, embed_num, embed_dim, class_num, kernel_num, kernel_sizes, dropout):
        super(CNNText, self).__init__()
        V = embed_num
        D = embed_dim
        C = class_num
        Ci = 1
        Co = kernel_num
        Ks = kernel_sizes

        self.convs = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(len(Ks) * Co, C)


    def forward(self, x):
        x = x.unsqueeze(1)  # (N, Ci, W, D)
        x = [torch.tanh(conv(x)).squeeze(3) for conv in self.convs]  # [(N, Co, W), ...]*len(Ks)
        x = [self.dropout(i) for i in x]  # (N, len(Ks)*Co)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
        x = torch.cat(x, 1)
        logits = self.fc1(x)  # (N, C)
        return logits