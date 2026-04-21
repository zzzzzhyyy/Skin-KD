import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadWeightedLoss(nn.Module):
    def __init__(self, weight, T = 4, entropy_weight=0.05):
        super().__init__()
        weights = torch.Tensor([29.041, 7.682, 9.675, 104.217, 5.596, 1.966, 39.996, 98.584]).cuda()
        self.entropy_weight = entropy_weight
        self.weight = weight
        self.T = T
        self.ce = nn.CrossEntropyLoss(weight=weights)

    def forward(self, logits_s, logits_t, target):

        s_logits = logits_s['logits']
        t_logits0 = logits_t['logits0']
        t_logits1 = logits_t['logits1']

        loss_ce = self.ce(s_logits, target)

        loss_k1 = F.kl_div(
            F.log_softmax(s_logits / self.T, dim=1),
            F.softmax(t_logits0.detach() / self.T, dim=1),
            reduction='batchmean'
        ) * (self.T * self.T)

        loss_k2 = F.kl_div(
            F.log_softmax(s_logits / self.T, dim=1),
            F.softmax(t_logits1.detach() / self.T, dim=1),
            reduction='batchmean'
        ) * (self.T * self.T)

        loss = loss_ce + (1- self.weight) * loss_k1 + self.weight * loss_k2
        return loss
