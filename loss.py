import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):

    def __init__(self, lbd=0, alpha=1, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.lbd = lbd  # focusing parameter
        self.alpha = alpha  # balancing weight
        self.reduction = reduction

    def forward(self, logits, targets):
        logp = F.log_softmax(logits, dim=1)
        alpha = self.alpha
        if isinstance(self.alpha, int):
            alpha = self.alpha * torch.ones(logits.size(1)).to(targets)
        loss = -(1 - logp.exp()).pow(self.lbd) * logp
        loss = loss.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1)
        loss.mul_(alpha[targets])
        if self.reduction != "none":
            loss = loss.sum()
        if self.reduction == "mean":
            loss.div_(logits.numel() // logits.size(1))
        return loss


def focal_loss(logits, targets, lbd=0, alpha=1, reduction="mean"):
    logp = F.log_softmax(logits, dim=1)
    if isinstance(alpha, int):
        alpha = alpha * torch.ones(logits.size(1)).to(targets)
    loss = -(1 - logp.exp()).pow(lbd) * logp
    loss = loss.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1)
    loss.mul_(alpha[targets])
    if reduction != "none":
        loss = loss.sum()
    if reduction == "mean":
        loss.div_(logits.numel() // logits.size(1))
    return loss


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fl_loss = FocalLoss()
    ce_loss = nn.CrossEntropyLoss()
    logits = 3 * torch.randn(128, 10, 3, 3).to(device)
    targets = torch.randint(0, 10, (128, 3, 3)).to(device)
    print(fl_loss(logits, targets).item(), ce_loss(logits, targets).item())
