import torch
from torch.nn import functional as F
import numpy as np
import torch.nn as nn
from torch.autograd import Variable


def dice_loss(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss


def dice_loss1(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target)
    z_sum = torch.sum(score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss


def entropy_loss(p, C=2):
    # p N*C*W*H*D
    y1 = -1 * torch.sum(p * torch.log(p + 1e-6), dim=1) / \
         torch.tensor(np.log(C)).cuda()
    ent = torch.mean(y1)

    return ent


def softmax_dice_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    n = input_logits.shape[1]
    dice = 0
    for i in range(0, n):
        dice += dice_loss1(input_softmax[:, i], target_softmax[:, i])
    mean_dice = dice / n

    return mean_dice


def entropy_loss_map(p, C=2):
    ent = -1 * torch.sum(p * torch.log(p + 1e-6), dim=1,
                         keepdim=True) / torch.tensor(np.log(C)).cuda()
    return ent


def softmax_mse_loss(input_logits, target_logits, sigmoid=False):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    if sigmoid:
        input_softmax = torch.sigmoid(input_logits)
        target_softmax = torch.sigmoid(target_logits)
    else:
        input_softmax = F.softmax(input_logits, dim=1)
        target_softmax = F.softmax(target_logits, dim=1)

    mse_loss = (input_softmax - target_softmax) ** 2
    return mse_loss


def softmax_kl_loss(input_logits, target_logits, sigmoid=False):
    """Takes softmax on both sides and returns KL divergence

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    if sigmoid:
        input_log_softmax = torch.log(torch.sigmoid(input_logits))
        target_softmax = torch.sigmoid(target_logits)
    else:
        input_log_softmax = F.log_softmax(input_logits, dim=1)
        target_softmax = F.softmax(target_logits, dim=1)

    # return F.kl_div(input_log_softmax, target_softmax)
    kl_div = F.kl_div(input_log_softmax, target_softmax, reduction='mean')
    # mean_kl_div = torch.mean(0.2*kl_div[:,0,...]+0.8*kl_div[:,1,...])
    return kl_div


def symmetric_mse_loss(input1, input2):
    """Like F.mse_loss but sends gradients to both directions

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to both input1 and input2.
    """
    assert input1.size() == input2.size()
    return torch.mean((input1 - input2) ** 2)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            # N,C,H,W => N,C,H*W
            input = input.view(input.size(0), input.size(1), -1)
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


class DiceLossWithMask(nn.Module):
    def __init__(self, n_classes):
        super(DiceLossWithMask, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _one_hot_mask_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor * i == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-10
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def _dice_mask_loss(self, score, target, mask):
        target = target.float()
        mask = mask.float()
        smooth = 1e-10
        intersect = torch.sum(score * target * mask)
        y_sum = torch.sum(target * target * mask)
        z_sum = torch.sum(score * score * mask)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, mask=None, weight=None, softmax=False, sigmoid=False, multi=False):
        if sigmoid and softmax:
            assert (0)
        elif sigmoid:
            inputs = inputs.sigmoid()
            target = target.squeeze(1)
        elif softmax:
            inputs = torch.softmax(inputs, dim=1)
        if multi:
            if mask is not None:
                dice = self._dice_mask_loss(inputs, target, mask)
            else:
                dice = self._dice_loss(inputs, target)
            return dice
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        if mask is not None:
            mask = self._one_hot_mask_encoder(mask)
            for i in range(0, self.n_classes):
                dice = self._dice_mask_loss(
                    inputs[:, i], target[:, i], mask[:, i])
                class_wise_dice.append(1.0 - dice.item())
                loss += dice * weight[i]
        else:
            for i in range(0, self.n_classes):
                dice = self._dice_loss(inputs[:, i], target[:, i])
                class_wise_dice.append(1.0 - dice.item())
                loss += dice * weight[i]
        return loss / self.n_classes


class DiceLossWithMaskGen(DiceLossWithMask):
    def _dice_loss(self, score, target, batch_w):
        target = target.float()
        dim_num = len(target.shape)
        dim = (1,2) if dim_num == 3 else (1,2,3)
        smooth = 1e-10
        intersect = torch.sum(score * target, dim=dim)
        y_sum = torch.sum(target * target, dim=dim)
        z_sum = torch.sum(score * score, dim=dim)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        loss *= batch_w
        return loss.mean()

    def _dice_mask_loss(self, score, target, mask, batch_w):
        target = target.float()
        mask = mask.float()
        dim_num = len(target.shape)
        dim = (1, 2) if dim_num == 3 else (1, 2, 3)
        smooth = 1e-10
        intersect = torch.sum(score * target * mask, dim=dim)
        y_sum = torch.sum(target * target * mask, dim=dim)
        z_sum = torch.sum(score * score * mask, dim=dim)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        loss *= batch_w
        return loss.mean()

    def forward(self, inputs, target, mask=None, batch_w=None, weight=None, softmax=False, sigmoid=False, multi=False):
        if batch_w is None:
            batch_w = torch.ones(inputs.shape[0]).to('cuda')
        if sigmoid and softmax:
            assert(0)
        elif sigmoid:
            inputs = inputs.sigmoid()
            target = target.squeeze(1)
        elif softmax:
            inputs = torch.softmax(inputs, dim=1)
        if multi:
            if mask is not None:
                dice = self._dice_mask_loss(inputs, target, mask, batch_w)
            else:
                dice = self._dice_loss(inputs, target, batch_w)
            return dice
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        if mask is not None:
            mask = self._one_hot_mask_encoder(mask)
            for i in range(0, self.n_classes):
                dice = self._dice_mask_loss(
                    inputs[:, i], target[:, i], mask[:, i], batch_w)
                class_wise_dice.append(1.0 - dice.item())
                loss += dice * weight[i]
        else:
            for i in range(0, self.n_classes):
                dice = self._dice_loss(inputs[:, i], target[:, i], batch_w)
                class_wise_dice.append(1.0 - dice.item())
                loss += dice * weight[i]
        return loss / self.n_classes


def entropy_minmization(p):
    y1 = -1 * torch.sum(p * torch.log(p + 1e-6), dim=1)
    ent = torch.mean(y1)

    return ent


def entropy_map(p):
    ent_map = -1 * torch.sum(p * torch.log(p + 1e-6), dim=1,
                             keepdim=True)
    return ent_map


def compute_kl_loss(p, q):
    p_loss = F.kl_div(F.log_softmax(p, dim=-1),
                      F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1),
                      F.softmax(p, dim=-1), reduction='none')

    # Using function "sum" and "mean" are depending on your task
    p_loss = p_loss.mean()
    q_loss = q_loss.mean()

    loss = (p_loss + q_loss) / 2
    return loss


# 新增损失
def soft_dice_loss(y_true, y_pred, epsilon=1e-6):
    """
    Soft dice loss calculation for arbitrary batch size, number of classes, and number of spatial dimensions.
    # Arguments
        y_true: b x c x X x Y( x Z...)  One hot encoding of ground truth
        y_pred: b x c x X x Y( x Z...)  Network output, must sum to 1 over c channel (such as after softmax)
        epsilon: Used for numerical stability to avoid divide by zero errors

    # References
        V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation
        https://arxiv.org/abs/1606.04797
        More details on Dice loss formulation
        https://mediatum.ub.tum.de/doc/1395260/1395260.pdf (page 72)

        Adapted from https://github.com/Lasagne/Recipes/issues/99#issuecomment-347775022
    """

    # skip the batch and class axis for calculating Dice score
    dim = (len(y_pred.shape) - 2, len(y_pred.shape) - 1)
    numerator = 2. * torch.sum(y_pred * y_true, dim)
    denominator = torch.sum(y_pred ** 2 + y_true ** 2, dim)
    # axes = tuple(range(1, len(y_pred.shape) - 1))
    # numerator = 2. * np.sum(y_pred * y_true, axes)
    # denominator = np.sum(np.square(y_pred) + np.square(y_true), axes)

    return 1 - torch.mean(numerator / (denominator + epsilon))  # average over classes and batch


# class SCELoss(torch.nn.Module):
#     def __init__(self, alpha, beta, num_classes=10):
#         """ 2019 - iccv - Symmetric cross entropy for robust learning with noisy labels """
#         super(SCELoss, self).__init__()
#         self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
#         self.alpha = alpha
#         self.beta = beta
#         self.num_classes = num_classes
#         self.softmax = torch.nn.Softmax(dim=1)
#
#     def forward(self, pred, target, mask):
#         """
#         param:
#         pred: shape of n,c
#         target: shape of n,
#         """
#         if mask.sum() == 0:
#             loss = 0.0 * pred.sum()
#             return loss
#         # pre-process
#         pred = pred.permute(0, 2, 3, 1).reshape(-1, pred.shape[1])  # bhw,c
#         mask = mask.view(-1).bool()  # bhw
#         target = target.view(-1)  # bhw
#         pred = pred[mask]  # n,c
#         target = target[mask]  # n
#
#         pred = self.softmax(pred)
#         label_one_hot = torch.nn.functional.one_hot(target, self.num_classes).float().to(self.device)  # n,c
#
#         pred = torch.clamp(pred, min=1e-7, max=1.0)
#         label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
#
#         ce = (-1 * torch.sum(label_one_hot * torch.log(pred), dim=1)).mean()
#         rce = (-1 * torch.sum(pred * torch.log(label_one_hot), dim=1)).mean()
#         loss = self.alpha * ce + self.beta * rce
#
#         return loss
class SCELoss(torch.nn.Module):
    def __init__(self, num_classes=10):
        """ 2024 - TMI - Imbalanced Medical Image Segmentation with Pixel-dependent Noisy Labels """
        super(SCELoss, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.num_classes = num_classes
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, pred, target, mask, alpha):
        """
        param:
        pred: shape of n,c
        target: shape of n,
        """
        if mask.sum() == 0:
            loss = 0.0 * pred.sum()
            return loss
        # pre-process
        pred = pred.permute(0, 2, 3, 1).reshape(-1, pred.shape[1])  # bhw,c
        mask = mask.view(-1).bool()  # bhw
        target = target.view(-1)  # bhw
        alpha = alpha.view(-1)  # bhw
        pred = pred[mask]  # n,c
        target = target[mask]  # n
        alpha = alpha[mask]  # n

        pred = self.softmax(pred)
        label_one_hot = torch.nn.functional.one_hot(target, self.num_classes).float().to(self.device)  # n,c
        # 高级索引规则：当使用两个形状相同的一维张量作为索引时，PyTorch 会按位置配对索引
        # （即(row_indices[0], B[0])、(row_indices[1], B[1])...），最终返回形状为(N,)的一维张量
        # alpha = pred[torch.arange(pred.shape[0]), target]  # n

        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)

        # ce = (-1 * torch.sum(label_one_hot * torch.log(pred), dim=1)).mean()
        # rce = (-1 * torch.sum(pred * torch.log(label_one_hot), dim=1)).mean()
        ce = (-1 * torch.sum(label_one_hot * torch.log(pred), dim=1)) * alpha
        rce = (-1 * torch.sum(pred * torch.log(label_one_hot), dim=1)) * (1-alpha)
        loss = (ce + rce).mean()

        return loss


class label_smooth(torch.nn.Module):
    def __init__(self, num_classes):
        super(label_smooth, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.num_classes = num_classes
        self.softmax = torch.nn.Softmax(dim=1)
        self.alpha = 0.1

    def forward(self, output, target):
        """
        param:
        pred: shape of b,c
        target: shape of b,
        """
        probs = self.softmax(output)
        label_one_hot = torch.nn.functional.one_hot(target, self.num_classes).float().to(self.device)
        label_one_hot = label_one_hot * (1. - self.alpha) + self.alpha / float(self.num_classes)
        loss = torch.sum(-label_one_hot * torch.log(probs), dim=1).mean()
        return loss


class CosineSimilarityLoss(nn.Module):
    def __init__(self, dim=1, reduction='mean'):
        """
        Args:
            dim (int): 计算全心相似度的维度，默认 1 (通道维度)
            reduction (str): 损失聚合方式，'mean', 'sum' 或 'none'
        """
        super(CosineSimilarityLoss, self).__init__()
        self.dim = dim
        self.reduction = reduction

    def forward(self, f1, f2):
        # f1, f2 的形状均为 (B, C, H, W)

        # F.cosine_similarity 返回值的范围在 [-1, 1] 之间
        # 1 表示方向完全相同，-1 表示方向完全相反
        cos_sim = F.cosine_similarity(f1, f2, dim=self.dim)

        # 将其转化为 Loss：我们希望 cos_sim 趋近于 1，所以 Loss = 1 - cos_sim
        # 转化后 Loss 范围在 [0, 2] 之间，越接近 0 越相似
        loss = 1.0 - cos_sim

        # 聚合损失
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
