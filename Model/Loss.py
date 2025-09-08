import torch
from torch.nn import functional as F


def ova_ent(probability):
    probability = probability.view(probability.size(0), 2, -1)
    probability = F.softmax(probability, dim=1)
    loss = torch.mean(
        torch.mean(torch.sum(
            -probability * torch.log(probability + 1e-8), 1), 1)
    )
    return loss


def ova_loss(probability, label, ood_index):
    label[ood_index] = 0
    probability = probability.view(probability.size(0), 2, -1)
    probability = F.softmax(probability, 1)

    label_s_sp = torch.zeros((probability.size(0),
                              probability.size(2))).long().to(label.device)
    label_range = torch.arange(0, probability.size(0) - 0.9).long()
    label_s_sp[label_range, label] = 1
    label_s_sp[ood_index] = 0
    label_sp_neg = 1 - label_s_sp
    open_loss = torch.mean(torch.sum(-torch.log(probability[:, 1, :]
                                                + 1e-8) * label_s_sp, 1))
    open_loss_neg = torch.mean(torch.max(-torch.log(probability[:, 0, :]
                                                    + 1e-8) * label_sp_neg, 1)[0])

    # open_loss_neg = torch.mean(torch.sum(-torch.log(logits_open[:, 0, :] + 1e-8) * label_sp_neg, 1))
    loss = open_loss_neg + open_loss
    return loss
