import torch
import torch.nn as nn
import torch.nn.functional as F

device = "cuda"

class AntiContrastiveLoss(nn.Module):
    def __init__(self, margin):
        super(AntiContrastiveLoss, self).__init__()
        self.margin = margin
        self.mse_loss = nn.MSELoss()

    def forward(self, v1_embeddings, v2_embeddings):
        distances = torch.sqrt(torch.sum((v1_embeddings - v2_embeddings) ** 2, dim=1))  # +ve
        loss = torch.mean(torch.clamp(distances - self.margin, min=0))  # torch.max()
        return loss
    
def dispersion_loss(cluster, min_variance=0.1):
    n = cluster.size(0)
    if n <= 1:
        return 0
    mean = torch.mean(cluster, dim=0)
    variances = torch.mean((cluster - mean) ** 2, dim=0)
    # Penalize if the variance in any dimension is below a threshold
    penalty = torch.mean(F.relu(min_variance - variances))
    return penalty

def mean_loss(cluster1, cluster2):
    mean1 = torch.mean(cluster1, dim=0)
    mean2 = torch.mean(cluster2, dim=0)
    return F.mse_loss(mean1, mean2)

def distance_loss(cluster1, cluster2):
    # Assuming cluster2 is a fixed set of points (frozen)
    distances = []
    for point in cluster1:
        dist = torch.min(torch.sum((cluster2 - point) ** 2, dim=1))
        distances.append(dist)
    return torch.mean(torch.stack(distances))

class CustomLoss(nn.Module):
    def __init__(self, weight_mean_loss, weight_distance_loss, weight_dispersion_loss):
        super(CustomLoss, self).__init__()
        self.weight_mean_loss = weight_mean_loss
        self.weight_distance_loss = weight_distance_loss
        self.weight_dispersion_loss = weight_dispersion_loss

    def forward(self, cluster1, cluster2):
        mean_loss_val = mean_loss(cluster1, cluster2)
        distance_loss_val = distance_loss(cluster1, cluster2)
        # distance_loss_val = AntiContrastiveLoss(margin=0)(cluster1, cluster2)
        dispersion_loss_val = dispersion_loss(cluster1)
        # print(f"Mean Loss: {mean_loss_val.item()}, Distance Loss: {distance_loss_val.item()}, Dispersion Loss: {dispersion_loss_val.item()}")
        
        loss = (self.weight_mean_loss * mean_loss_val +
                self.weight_distance_loss * distance_loss_val +
                self.weight_dispersion_loss * dispersion_loss_val)
        return loss

def MMD(x, y, kernel):
    """
    Emprical maximum mean discrepancy. The lower the result, the more evidence that distributions are the same.

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"
    """
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    dxx = rx.t() + rx - 2. * xx # Used for A in (1)
    dyy = ry.t() + ry - 2. * yy # Used for B in (1)
    dxy = rx.t() + ry - 2. * zz # Used for C in (1)

    XX, YY, XY = (torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device))

    if kernel == "multiscale":
        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a**2 * (a**2 + dxx)**-1
            YY += a**2 * (a**2 + dyy)**-1
            XY += a**2 * (a**2 + dxy)**-1

    if kernel == "rbf":
        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5*dxx/a)
            YY += torch.exp(-0.5*dyy/a)
            XY += torch.exp(-0.5*dxy/a)

    return torch.mean(XX + YY - 2. * XY)
