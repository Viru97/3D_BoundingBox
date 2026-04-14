import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

def hungarian_corner_loss(pred, target):
    """
    Computes smoothed L1 loss between predicted and target 3D bounding box corners
    using bipartite matching (Hungarian sequence) to avoid corner ordering ambiguity.
    """
    pred = pred.float()
    target = target.float()
    B = pred.shape[0]
    loss = 0.0
    for b in range(B):
        # Compute cost matrix between predicted and target corners
        cost = torch.cdist(pred[b], target[b]).detach().cpu().numpy()
        row, col = linear_sum_assignment(cost)
        
        # Apply smooth L1 loss to the matched corners
        loss += F.smooth_l1_loss(pred[b][row], target[b][col], beta=0.05)
        
    return loss / B

def compute_loss(pred, tgt, center):
    """
    Compute the total loss composed of:
    1. Chamfer distance between predicted and target bounding box corners.
    2. Hungarian matched smooth L1 corner loss.
    3. Centre Smooth-L1 regression loss (in anchor-relative space).
    """
    pred_f = pred.float()
    tgt_f = tgt.float()
    center_f = center.float()
    
    # Centre loss — both in anchor-relative space (not world space)
    # target centre ≈ [0,0,0] since targets are anchor-subtracted
    tgt_centre = tgt_f.mean(dim=1)
    loss_centre = F.smooth_l1_loss(center_f, tgt_centre, beta=0.01)

    dist = torch.cdist(pred_f, tgt_f)
    chamfer = dist.min(2)[0].mean() + dist.min(1)[0].mean()
    hungarian = hungarian_corner_loss(pred_f, tgt_f)

    loss = chamfer + 1.0 * hungarian + 1.0 * loss_centre
    
    return loss, chamfer, hungarian, loss_centre

@torch.no_grad()
def mean_corner_dist(pred, target):
    """
    Computes the Mean Corner Distance (MCD) for evaluation points.
    Matches prediction corners with target corners using the Hungarian algorithm
    and calculates the average point-to-point Euclidean distance.
    """
    pred = pred.float()
    target = target.float()
    B = pred.shape[0]
    mcds = []
    for b in range(B):
        cost = torch.cdist(pred[b], target[b]).cpu().numpy()
        row, col = linear_sum_assignment(cost)
        d = (pred[b][row] - target[b][col]).pow(2).sum(-1).sqrt().mean()
        mcds.append(d.item())
    return mcds
