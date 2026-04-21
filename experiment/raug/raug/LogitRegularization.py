import torch
import torch.nn.functional as F

class LogitRegularization(torch.nn.Module):
    def __init__(self, tau=1.0, weight=None, reduction='mean'):
        super().__init__()
        self.tau = tau
        self.weight = weight
        self.reduction = reduction

    def forward(self, logits, labels):
        ce_loss = F.cross_entropy(logits, labels, weight=self.weight, reduction=self.reduction)
        norm = torch.norm(logits, p=2, dim=1)
        norm_loss = ((norm - self.tau) ** 2).mean()
        lambda_reg = 0.05
        total_loss = ce_loss + lambda_reg * norm_loss
        return total_loss