# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations
import os
import pickle
import numpy as np
import re
from scipy.ndimage import distance_transform_edt as distance
# from skimage import segmentation as skimage_seg
import torch
from skimage.segmentation import find_boundaries
from torch.utils.data.sampler import Sampler
import torch.distributed as dist
import cv2
from numpy import ndarray
from scipy.ndimage.morphology import distance_transform_edt
import os.path as osp
import torch.nn.functional as F
from collections import defaultdict


def save_model(model, epoch, directory):
    model_dict = model.state_dict()
    style_feats = {'mean': [], 'cov': []}

    for conststyle in model.conststyle:
        style_feats['mean'].append(conststyle.const_mean)
        style_feats['cov'].append(conststyle.const_cov)
    state = {
        'state_dict': model_dict,
        'epoch': epoch + 1,
        'style_feats': style_feats
    }
    torch.save(state, directory)


def load_model(model, directory):
    if not osp.exists(directory):
        raise FileNotFoundError(
            'Model not found at "{}"'.format(directory)
        )

    checkpoint = torch.load(directory, map_location='cpu')  # 'cpu'
    state_dict = checkpoint['state_dict']
    epoch = checkpoint['epoch']

    print(
        'Loading weights from "{}" (epoch = {})'.format(directory, epoch)
    )
    model.load_state_dict(state_dict)

    for idx, conststyle in enumerate(model.conststyle):
        conststyle.const_mean = checkpoint['style_feats']['mean'][idx]
        conststyle.const_cov = checkpoint['style_feats']['cov'][idx]
        conststyle.generator = torch.distributions.MultivariateNormal(loc=conststyle.const_mean,
                                                                      covariance_matrix=conststyle.const_cov)

    return epoch


def save_model_icnn(model, epoch, directory):
    model_dict = model.state_dict()
    style_feats = {'convex_g': [], 'convex_f': []}

    for conststyle in model.conststyle:
        style_feats['convex_g'].append(conststyle.ot_icnn.convex_g.state_dict())
        style_feats['convex_f'].append(conststyle.ot_icnn.convex_f.state_dict())
    state = {
        'state_dict': model_dict,
        'epoch': epoch + 1,
        'style_feats': style_feats
    }
    torch.save(state, directory)


def save_dpmodel(model, epoch, directory):
    model_dict = model.state_dict()
    style_feats = {'mean': [], 'cov': [], 'mean_mcs': [], 'cov_mcs': [], 'mean_cls': [], 'cov_cls': []}

    for conststyle in model.conststyle:
        style_feats['mean'].append(conststyle.const_mean)
        style_feats['cov'].append(conststyle.const_cov)
        style_feats['mean_mcs'].append(conststyle.const_mean_mcs)
        style_feats['cov_mcs'].append(conststyle.const_cov_mcs)
        style_feats['mean_cls'].append(conststyle.const_mean_cls)
        style_feats['cov_cls'].append(conststyle.const_cov_cls)

    state = {
        'state_dict': model_dict,
        'epoch': epoch + 1,
        'style_feats': style_feats
    }
    torch.save(state, directory)


def load_dpmodel(model, directory):
    if not osp.exists(directory):
        raise FileNotFoundError(
            'Model not found at "{}"'.format(directory)
        )

    # 1. 核心修改：将整个 checkpoint 先加载到系统内存 (CPU) 中
    checkpoint = torch.load(directory, map_location='cpu')

    state_dict = checkpoint['state_dict']
    epoch = checkpoint['epoch']

    print(
        'Loading weights to {} (epoch = {})'.format(directory, epoch)
    )

    # 2. 加载权重到模型。
    # 前提：只要你在此函数外部已经将 model 放在了 GPU 上（例如 model.to('cuda:0')），
    # load_state_dict 就会自动将 CPU 上的 state_dict 数据复制到 GPU 的模型参数中。
    model.load_state_dict(state_dict)

    # 3. 此时 checkpoint 中的其他键值对（如 style_feats）都处于 CPU 上，
    # 赋值给 conststyle 时，它们也不会占用宝贵的显存。
    style_feats = checkpoint['style_feats']
    for idx, conststyle in enumerate(model.conststyle):
        conststyle.const_mean = style_feats['mean'][idx]
        conststyle.const_cov = style_feats['cov'][idx]
        conststyle.const_mean_mcs = style_feats['mean_mcs'][idx]
        conststyle.const_cov_mcs = style_feats['cov_mcs'][idx]
        conststyle.const_mean_cls = style_feats['mean_cls'][idx]
        conststyle.const_cov_cls = style_feats['cov_cls'][idx]
        # TODO:在此处增加generator
        if conststyle.const_mean is not None:
            conststyle.generator = torch.distributions.MultivariateNormal(loc=conststyle.const_mean,
                                                                          covariance_matrix=conststyle.const_cov)
        if conststyle.const_mean_mcs is not None:
            conststyle.generator_mcs = torch.distributions.MultivariateNormal(loc=conststyle.const_mean_mcs,
                                                                              covariance_matrix=conststyle.const_cov_mcs)
        for i in range(len(conststyle.const_mean_cls)):
            if conststyle.const_mean_cls[i] is not None:
                conststyle.generator_cls[i] = torch.distributions.MultivariateNormal(loc=conststyle.const_mean_cls[i],
                                                                                     covariance_matrix=conststyle.const_cov_cls[i])

    return epoch


def save_dpotmodel(model, epoch, directory):
    model_dict = model.state_dict()
    convex_g = []
    for idx, conststyle in enumerate(model.conststyle):
        ot_icnn = {}
        ot_icnn['cs'] = conststyle.ot_icnn.convex_g.state_dict()
        ot_icnn['mcs'] = conststyle.icnn_mcs.convex_g.state_dict()
        ot_icnn['cls'] = []
        for cls in range(model.n_classes):
            ot_icnn['cls'].append(conststyle.icnn_cls[cls].convex_g.state_dict())

        convex_g.append(ot_icnn)

    state = {
        'state_dict': model_dict,
        'epoch': epoch + 1,
        'convex_g': convex_g
    }
    torch.save(state, directory)


def load_dpotmodel(model, directory):
    if not osp.exists(directory):
        raise FileNotFoundError(
            'Model not found at "{}"'.format(directory)
        )

    checkpoint = torch.load(directory, map_location='cuda:0')  # 'cpu'
    state_dict = checkpoint['state_dict']
    epoch = checkpoint['epoch']
    convex_g = checkpoint['convex_g']

    print('Loading weights from "{}" (epoch = {})'.format(directory, epoch))
    model.load_state_dict(state_dict)
    for idx, conststyle in enumerate(model.conststyle):
        ot_icnn = convex_g[idx]
        conststyle.ot_icnn.convex_g.load_state_dict(ot_icnn['cs'])
        conststyle.icnn_mcs.convex_g.load_state_dict(ot_icnn['mcs'])
        for cls in range(model.n_classes):
            conststyle.icnn_cls[cls].convex_g.load_state_dict(ot_icnn['cls'][cls])

    return epoch


def save_dprusotmodel(model, epoch, directory):
    model_dict = model.state_dict()
    convex_g = []
    style_feats = {'mean': [], 'cov': [], 'mean_mcs': [], 'cov_mcs': [], 'mean_cls': [], 'cov_cls': []}
    for idx, conststyle in enumerate(model.conststyle):
        if idx < model.ot_num:
            convex_g.append(conststyle.ot_icnn.convex_g.state_dict())
        else:
            style_feats['mean'].append(conststyle.const_mean)
            style_feats['cov'].append(conststyle.const_cov)
            style_feats['mean_mcs'].append(conststyle.const_mean_mcs)
            style_feats['cov_mcs'].append(conststyle.const_cov_mcs)
            style_feats['mean_cls'].append(conststyle.const_mean_cls)
            style_feats['cov_cls'].append(conststyle.const_cov_cls)


    state = {
        'state_dict': model_dict,
        'epoch': epoch + 1,
        'convex_g': convex_g,
        'style_feats': style_feats
    }
    torch.save(state, directory)


# import networks

# many issues with this function
# def load_model(path):
#     """Loads model and return it without DataParallel table."""
#     if os.path.isfile(path):
#         print("=> loading checkpoint '{}'".format(path))
#         checkpoint = torch.load(path)
#
#         for key in checkpoint["state_dict"]:
#             print(key)
#
#         # size of the top layer
#         N = checkpoint["state_dict"]["decoder.out_conv.bias"].size()
#
#         # build skeleton of the model
#         sob = "sobel.0.weight" in checkpoint["state_dict"].keys()
#         model = models.__dict__[checkpoint["arch"]](sobel=sob, out=int(N[0]))
#
#         # deal with a dataparallel table
#         def rename_key(key):
#             if not "module" in key:
#                 return key
#             return "".join(key.split(".module"))
#
#         checkpoint["state_dict"] = {
#             rename_key(key): val for key, val in checkpoint["state_dict"].items()
#         }
#
#         # load weights
#         model.load_state_dict(checkpoint["state_dict"])
#         print("Loaded")
#     else:
#         model = None
#         print("=> no checkpoint found at '{}'".format(path))
#     return model


def load_checkpoint(path, model, optimizer, from_ddp=False):
    """loads previous checkpoint

    Args:
        path (str): path to checkpoint
        model (model): model to restore checkpoint to
        optimizer (optimizer): torch optimizer to load optimizer state_dict to
        from_ddp (bool, optional): load DistributedDataParallel checkpoint to regular model. Defaults to False.

    Returns:
        model, optimizer, epoch_num, loss
    """
    # load checkpoint
    checkpoint = torch.load(path)
    # transfer state_dict from checkpoint to model
    model.load_state_dict(checkpoint["state_dict"])
    # transfer optimizer state_dict from checkpoint to model
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    # track loss
    loss = checkpoint["loss"]
    return model, optimizer, checkpoint["epoch"], loss.item()


def restore_model(logger, snapshot_path, model_num=None):
    """wrapper function to read log dir and load restore a previous checkpoint

    Args:
        logger (Logger): logger object (for info output to console)
        snapshot_path (str): path to checkpoint directory

    Returns:
        model, optimizer, start_epoch, performance
    """
    try:
        # check if there is previous progress to be restored:
        logger.info(f"Snapshot path: {snapshot_path}")
        iter_num = []
        name = "model_iter"
        if model_num:
            name = model_num
        for filename in os.listdir(snapshot_path):
            if name in filename:
                basename, extension = os.path.splitext(filename)
                iter_num.append(int(basename.split("_")[2]))
        iter_num = max(iter_num)
        for filename in os.listdir(snapshot_path):
            if name in filename and str(iter_num) in filename:
                model_checkpoint = filename
    except Exception as e:
        logger.warning(f"Error finding previous checkpoints: {e}")

    try:
        logger.info(f"Restoring model checkpoint: {model_checkpoint}")
        model, optimizer, start_epoch, performance = load_checkpoint(
            snapshot_path + "/" + model_checkpoint, model, optimizer
        )
        logger.info(f"Models restored from iteration {iter_num}")
        return model, optimizer, start_epoch, performance
    except Exception as e:
        logger.warning(f"Unable to restore model checkpoint: {e}, using new model")


def save_checkpoint(epoch, model, optimizer, loss, path):
    """Saves model as checkpoint"""
    torch.save(
        {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        path,
    )


class UnifLabelSampler(Sampler):
    """Samples elements uniformely accross pseudolabels.
    Args:
        N (int): size of returned iterator.
        images_lists: dict of key (target), value (list of data with this target)
    """

    def __init__(self, N, images_lists):
        self.N = N
        self.images_lists = images_lists
        self.indexes = self.generate_indexes_epoch()

    def generate_indexes_epoch(self):
        size_per_pseudolabel = int(self.N / len(self.images_lists)) + 1
        res = np.zeros(size_per_pseudolabel * len(self.images_lists))

        for i in range(len(self.images_lists)):
            indexes = np.random.choice(
                self.images_lists[i],
                size_per_pseudolabel,
                replace=(len(self.images_lists[i]) <= size_per_pseudolabel),
            )
            res[i * size_per_pseudolabel: (i + 1) * size_per_pseudolabel] = indexes

        np.random.shuffle(res)
        return res[: self.N].astype("int")

    def __iter__(self):
        return iter(self.indexes)

    def __len__(self):
        return self.N


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def learning_rate_decay(optimizer, t, lr_0):
    for param_group in optimizer.param_groups:
        lr = lr_0 / np.sqrt(1 + lr_0 * param_group["weight_decay"] * t)
        param_group["lr"] = lr


class Logger:
    """Class to update every epoch to keep trace of the results
    Methods:
        - log() log and save
    """

    def __init__(self, path):
        self.path = path
        self.data = []

    def log(self, train_point):
        self.data.append(train_point)
        with open(os.path.join(self.path), "wb") as fp:
            pickle.dump(self.data, fp, -1)


def compute_sdf(img_gt, out_shape):
    """
    compute the signed distance map of binary mask
    input: segmentation, shape = (batch_size, x, y, z)
    output: the Signed Distance Map (SDM)
    sdf(x) = 0; x in segmentation boundary
             -inf|x-y|; x in segmentation
             +inf|x-y|; x out of segmentation
    normalize sdf to [-1,1]
    """

    img_gt = img_gt.astype(np.uint8)
    normalized_sdf = np.zeros(out_shape)

    for b in range(out_shape[0]):  # batch size
        posmask = img_gt[b].astype(np.bool)
        if posmask.any():
            negmask = ~posmask
            posdis = distance(posmask)
            negdis = distance(negmask)
            boundary = skimage_seg.find_boundaries(posmask, mode="inner").astype(
                np.uint8
            )
            sdf = (negdis - np.min(negdis)) / (np.max(negdis) - np.min(negdis)) - (
                    posdis - np.min(posdis)
            ) / (np.max(posdis) - np.min(posdis))
            sdf[boundary == 1] = 0
            normalized_sdf[b] = sdf
            # assert np.min(sdf) == -1.0, print(np.min(posdis), np.max(posdis), np.min(negdis), np.max(negdis))
            # assert np.max(sdf) ==  1.0, print(np.min(posdis), np.min(negdis), np.max(posdis), np.max(negdis))

    return normalized_sdf


# set up process group for distributed computing
def distributed_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    print("setting up dist process group now")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def load_ddp_to_nddp(state_dict):
    pattern = re.compile("module")
    for k, v in state_dict.items():
        if re.search("module", k):
            model_dict[re.sub(pattern, "", k)] = v
        else:
            model_dict = state_dict
    return model_dict


def save_osmancheckpoint(epoch, ema_model, model, optimizer, best_dice, best_iter, stu_best_dice, stu_best_iter, path):
    """Saves model as checkpoint"""
    style_feats = {'mean': [], 'cov': []}
    for conststyle in model.conststyle:
        style_feats['mean'].append(conststyle.const_mean)
        style_feats['cov'].append(conststyle.const_cov)

    ema_style_feats = {'mean': [], 'cov': []}
    for conststyle in ema_model.conststyle:
        ema_style_feats['mean'].append(conststyle.const_mean)
        ema_style_feats['cov'].append(conststyle.const_cov)

    torch.save(
        {
            "epoch": epoch,
            "ema_state_dict": ema_model.state_dict(),
            'ema_style_feats': ema_style_feats,
            "state_dict": model.state_dict(),
            'style_feats': style_feats,
            "optimizer_state_dict": optimizer.state_dict(),
            "best_dice": best_dice,
            "best_iter": best_iter,
            "stu_best_dice": stu_best_dice,
            "stu_best_iter": stu_best_iter
        },
        path,
    )


def load_osmancheckpoint(path, ema_model, model, optimizer, from_ddp=False):
    """loads previous checkpoint

    Args:
        path (str): path to checkpoint
        model (model): model to restore checkpoint to
        optimizer (optimizer): torch optimizer to load optimizer state_dict to
        from_ddp (bool, optional): load DistributedDataParallel checkpoint to regular model. Defaults to False.

    Returns:
        model, optimizer, epoch_num, loss
    """
    # TODO: 未将style_feat进行加载
    # load checkpoint
    checkpoint = torch.load(path)
    # transfer state_dict from checkpoint to model
    ema_model.load_state_dict(checkpoint["ema_state_dict"])
    model.load_state_dict(checkpoint["state_dict"])
    # transfer optimizer state_dict from checkpoint to model
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    best_dice, best_iter = checkpoint['best_dice'], checkpoint['best_iter']
    stu_best_dice, stu_best_iter = checkpoint['stu_best_dice'], checkpoint['stu_best_iter']
    return checkpoint["epoch"], ema_model, model, optimizer, best_dice, best_iter, stu_best_dice, stu_best_iter


def draw_contour(
        image: ndarray,
        label: ndarray,
        color: tuple[int, int, int],
) -> ndarray:
    """Draw contour of label on image with color.

    Uses `cv2.dilate` to find contour.

    Args:
        image: ndarray of shape (h, w, 3)
        label: ndarray of shape (h, w) where non-zero values are foreground
        color: color of contour
    """
    binary = (label > 0).astype('uint8') * 255
    dilated = cv2.dilate(binary, np.ones((3, 3), dtype='uint8'), iterations=1)
    contour_mask = (dilated - binary) > 0
    image[contour_mask] = color
    return image


def make_prediction(
        image: ndarray,
        *predictions: tuple[ndarray, tuple[int, int, int]],
) -> ndarray:
    """Draw contour of predictions on image.

    Args:
        image: ndarray of shape (h, w) or (h, w, 3)
            if image is (h, w), it will be repeated to (h, w, 3)
        predictions: each prediction is a tuple of (label, color)
            label: ndarray of shape (c, h, w) or (h, w)
                where c is number of classes or 1
            color: color of contour
    """
    if image.ndim == 2:
        # expand to (h, w, 3)
        image = np.repeat(image[..., np.newaxis], 3, axis=-1)

    for prediction, color in predictions:
        if prediction.ndim == 2:
            # expand to (1, h, w)
            prediction = prediction[np.newaxis, ...]
        for class_map in prediction:
            image = draw_contour(image, class_map, color)
    return image


def draw_contour_and_save(img, pred, mask, save_path='./img/1/example.png'):
    color_gt = (255, 0, 0)
    color_pred_list = [(0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    if len(pred.shape) == 2:
        pred = pred.unsqueeze(0)
    if len(mask.shape) == 2:
        mask = mask.unsqueeze(0)
    if len(img.shape) == 2:
        img = img.unsqueeze(0)
    if len(img) == 1:
        img = img.repeat(3, 1, 1)
    img = (img - img.min()) / (img.max() - img.min()) * 255
    img = img.cpu().numpy().transpose(1, 2, 0)
    classes = len(pred)
    for i in range(classes):
        predi = pred[i].float().cpu().numpy()
        maski = mask[i].float().cpu().numpy()
        seg_res = make_prediction(img, (predi, color_pred_list[i]), (maski, color_gt))
        seg_res = cv2.cvtColor(seg_res, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, seg_res)


def draw_mask_and_save(img, pred, save_path='./img/1/example.png'):
    color_list = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    if len(pred.shape) == 2:
        pred = pred.unsqueeze(0)
    if len(img.shape) == 2:
        img = img.unsqueeze(0)
    if len(img) == 1:
        img = img.repeat(3, 1, 1)
    d, h, w = img.shape
    rgb = np.zeros((h, w, 3))
    mask = np.ones((h, w, 3))
    if img.min() < 0:
        img = (img + 1) * 127.5
    elif img.max() < 1:
        img = img * 255
    img = img.cpu().numpy().transpose(1, 2, 0)
    classes = len(pred)
    for i in reversed(range(classes)):
        mask[pred[i] == 1] = (0.5, 0.5, 0.5)
        rgb[pred[i] == 1] = color_list[i]
    res = (img + rgb) * mask
    res = res.astype(np.uint8)
    res = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, res)


def get_pixel_sets_distrans(src_sets, radius=2):
    """
        src_sets: shape->[N, 28, 28]
    """
    if isinstance(src_sets, torch.Tensor):
        src_sets = src_sets.numpy()
    if isinstance(src_sets, np.ndarray):
        keeps = []
        for src_set in src_sets:
            # 计算数组中值为 非零 的点到最近 零值点 的距离，并同时返回距离矩阵
            # 计算每个点到边界的距离， Ture为 inn and out
            keep = distance_transform_edt(np.logical_not(src_set))
            keep = keep < radius
            keeps.append(keep.astype(np.float32))
    else:
        raise ValueError(f'only np.ndarray is supported!')
    return torch.tensor(keeps).to(dtype=torch.long)


def bound_region(label_info, radius):
    bound = []
    for mi in range(label_info.shape[0]):
        bound_bool = find_boundaries(label_info[mi, :, :].cpu().numpy().astype(np.int8), mode='thick')
        bound.append(torch.from_numpy(bound_bool))
    bound = torch.stack(bound, dim=0).int().cuda()  # bsz,h,w
    if radius > 0:
        # 扩展边界
        edge_sets_dilate = get_pixel_sets_distrans(bound.cpu(), radius=radius)
        bound = edge_sets_dilate.cuda()  # - bound

    return bound


def mask_error_rate(mask, pred, label):
    mask = torch.squeeze(mask)
    assert mask.shape == pred.shape and mask.shape == label.shape
    pred_m, label_m = pred[mask == 1], label[mask == 1]  # n
    corr_ratio = (pred_m == label_m).sum() / label_m.shape[0]

    return 1 - corr_ratio


def to_2d(input_tensor):
    """将{0, 1, 2}转化为{cup, disc(含cup)}"""
    input_tensor = input_tensor.unsqueeze(1)
    tensor_list = []
    temp_prob = input_tensor == torch.ones_like(input_tensor)
    tensor_list.append(temp_prob)
    temp_prob2 = input_tensor > torch.zeros_like(input_tensor)
    tensor_list.append(temp_prob2)
    output_tensor = torch.cat(tensor_list, dim=1)
    return output_tensor.float()


def to_3d(input_tensor):
    input_tensor = input_tensor.unsqueeze(1)
    tensor_list = []
    for i in range(1, 4):
        temp_prob = input_tensor == i * torch.ones_like(input_tensor)
        tensor_list.append(temp_prob)
    output_tensor = torch.cat(tensor_list, dim=1)
    return output_tensor.float()


def to_onehot(data_name, output, mask):
    if data_name == 'fundus':
        pred_label = torch.max(torch.softmax(output, dim=1), dim=1)[1]
        pred_onehot = to_2d(pred_label)
        mask_onehot = to_2d(mask)
        # pred_label = torch.sigmoid(output).ge(0.5)
        # pred_onehot = pred_label.clone()
        # mask_onehot = mask.clone()
    elif data_name == 'prostate':
        pred_label = torch.max(torch.softmax(output, dim=1), dim=1)[1]
        pred_onehot = pred_label.clone().unsqueeze(1)
        mask_onehot = mask.clone().unsqueeze(1)
    elif data_name == 'MNMS':
        pred_label = torch.max(torch.softmax(output, dim=1), dim=1)[1]
        pred_onehot = to_3d(pred_label)
        mask_onehot = to_3d(mask)
    else:
        raise ValueError

    return pred_onehot, mask_onehot


def draw_mask(img, pred, label):
    """对类别（除bg）在原图上着色"""
    assert pred.shape == label.shape
    color_list = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    if len(pred.shape) == 3:
        pred = pred.unsqueeze(1)
    if len(label.shape) == 3:
        label = label.unsqueeze(1)
    if len(img.shape) == 3:
        img = img.unsqueeze(1)
    if img.shape[1] == 1:
        img = img.repeat(1, 3, 1, 1)
    b, d, h, w = img.shape
    rgb, rgb_l = np.zeros((b, h, w, 3)), np.zeros((b, h, w, 3))
    mask, mask_l = np.ones((b, h, w, 3)), np.ones((b, h, w, 3))

    if img.min() < 0:
        img = (img + 1) * 127.5
    elif img.max() < 1:
        img = img * 255
    img = img.cpu().numpy().transpose(0, 2, 3, 1)
    classes = pred.shape[1]
    for i in reversed(range(classes)):
        mask[pred[:, i, :, :] == 1] = (0.5, 0.5, 0.5)
        rgb[pred[:, i, :, :] == 1] = color_list[i]

        mask_l[label[:, i, :, :] == 1] = (0.5, 0.5, 0.5)
        rgb_l[label[:, i, :, :] == 1] = color_list[i]

    res = (img + rgb) * mask
    res = res.astype(np.uint8).transpose(0, 3, 1, 2)  # b,3,h,w
    res = torch.from_numpy(res).float() / 255.0

    res_l = (img + rgb_l) * mask_l
    res_l = res_l.astype(np.uint8).transpose(0, 3, 1, 2)  # b,3,h,w
    res_l = torch.from_numpy(res_l).float() / 255.0

    return res, res_l


def visualize_error(mask, pseudo_label, label, num_classes):
    """
    可视化预测结果与标签的对比

    参数:
    - mask: 形状为[B,H,W]的tensor，0代表不可靠，1代表可靠
    - pseudo_label: 形状为[B,H,W]的tensor，预测值，取值为{0,1,...,C}
    - label: 形状为[B,H,W]的tensor，真实标签，取值为{0,1,...,C}
    - num_classes: 类别数量C

    return
    1. 可靠且匹配：类别颜色
    2. 可靠但不匹配：深紫色
    3. 不可靠：青色
    """
    # 获取批次大小、高度和宽度
    B, H, W = mask.shape
    color_list = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1)]  # 依次为：红，绿，蓝，黄，亮紫
    # 创建颜色映射，为每个类别分配一种颜色
    colors = np.array(color_list[:num_classes])  # 取前三个通道(RGB)
    blue = np.array([0, 1, 1])  # 不可靠位置的青色
    red = np.array([0.5, 0, 0.5])  # 预测错误的深紫色

    # 将颜色转换为tensor并调整形状以便广播
    colors_tensor = torch.tensor(colors, dtype=torch.float32).view(1, num_classes, 3, 1, 1).cuda()
    blue_tensor = torch.tensor(blue, dtype=torch.float32).view(1, 3, 1, 1).cuda()
    red_tensor = torch.tensor(red, dtype=torch.float32).view(1, 3, 1, 1).cuda()

    # 生成类别颜色图像 [B, 3, H, W]
    # 使用高级索引选择每个像素对应的类别颜色
    class_indices = pseudo_label.unsqueeze(1)  # [B, 1, H, W]
    class_colors = torch.gather(colors_tensor.expand(B, -1, -1, H, W),
                                dim=1,
                                index=class_indices.expand(-1, 3, -1, -1).unsqueeze(1)).squeeze(1)  # b,3,h,w

    # 创建可靠/不可靠位置的掩码
    reliable_mask = mask.unsqueeze(1).float()  # [B, 1, H, W]，可靠位置为1
    unreliable_mask = 1 - reliable_mask  # [B, 1, H, W]，不可靠位置为1

    # 创建预测匹配/不匹配的掩码
    match_mask = (pseudo_label == label).unsqueeze(1).float()  # [B, 1, H, W]，匹配为1
    mismatch_mask = 1 - match_mask  # [B, 1, H, W]，不匹配为1

    # 计算最终图像：组合三种情况
    # 1. 可靠且匹配：类别颜色
    # 2. 可靠但不匹配：深紫色
    # 3. 不可靠：青色
    vis_images = (reliable_mask * match_mask * class_colors +
                  reliable_mask * mismatch_mask * red_tensor +
                  unreliable_mask * blue_tensor)

    return vis_images


class StyleStatManager:
    def __init__(self, num_classes, track_global=True):
        self.num_classes = num_classes
        self.track_global = track_global
        self.reset()  # 初始化存储容器

    def reset(self):
        """清空或初始化字典"""
        # 使用 defaultdict，当访问不存在的 key 时会自动创建空列表或字典，无需手动写 for 循环初始化
        if self.track_global:
            self.mean_set = defaultdict(list)
            self.std_set = defaultdict(list)
            self.domain_set = defaultdict(list)

        # 嵌套 defaultdict 用于 class 级别的统计
        self.mean_set_cls = defaultdict(lambda: defaultdict(list))
        self.std_set_cls = defaultdict(lambda: defaultdict(list))
        self.domain_set_cls = defaultdict(lambda: defaultdict(list))

        """
            for non-aligned feature

            mean_set = {i: [] for i in range(5)} # i:layer
            
            std_set = {i: [] for i in range(5)}
            
            domain_set = {i: [] for i in range(5)}
            
            mean_set_cls = {i: {j: [] for j in range(1, num_class)} for i in range(5)} # i:layer | j:class
            
            std_set_cls = {i: {j: [] for j in range(1, num_class)} for i in range(5)} # i:layer | j:class
            
            domain_set_cls = {i: {j: [] for j in range(1, num_class)} for i in range(5)}
            
            # for aligned feature
            
            mean_cs_set = {i: [] for i in range(model.num_conststyle)}
            
            std_cs_set = {i: [] for i in range(model.num_conststyle)}
            
            domain_cs_set = {i: [] for i in range(model.num_conststyle)}
            
            mean_cs_set_cls = {i: {j: [] for j in range(1, num_class)} for i in range(model.num_conststyle)}
            
            std_cs_set_cls = {i: {j: [] for j in range(1, num_class)} for i in range(model.num_conststyle)}
            
            domain_cs_set_cls = {i: {j: [] for j in range(1, num_class)} for i in range(model.num_conststyle)}
        """

    @torch.no_grad()
    def update(self, features_list, label, dc):
        """核心逻辑：传入特征列表、标签和域信息，自动追加到字典中"""
        for i, xi in enumerate(features_list):
            b, c, h, w = xi.shape
            # 对齐 label 分辨率
            labels = F.interpolate(label, (h, w), mode='nearest').reshape(b, -1)  # b, 1hw
            xi = xi.permute(0, 2, 3, 1).reshape(b, -1, c)  # b, hw, c

            for j in range(b):
                # 1. 记录全局特征 (如果需要)
                if self.track_global:
                    # 强烈建议这里加上 .cpu()，否则大量 tensor 存在列表中会导致 GPU 显存溢出 (OOM)
                    self.mean_set[i].append(xi[j].mean(dim=0).cpu())
                    self.std_set[i].append((xi[j].var(dim=0) + 1e-6).sqrt().cpu())
                    self.domain_set[i].append(dc)

                # 2. 记录类别特征
                for cli in range(1, self.num_classes):
                    mask = (labels[j] == cli)
                    if mask.sum() > 0:  # 避免全 0 切片报错
                        cls_fea = xi[j][mask]  # n, c
                        self.mean_set_cls[i][cli].append(cls_fea.mean(dim=0).cpu())
                        self.std_set_cls[i][cli].append((cls_fea.var(dim=0) + 1e-6).sqrt().cpu())
                        self.domain_set_cls[i][cli].append(dc)
