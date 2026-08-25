import argparse
import logging
import os
import random
import sys
from typing import Iterable

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from cl.css_uni import StyleProtoLoss
from utils.evaluate import eval_dprus
from networks.unet_model import OptionalConstStyleModel
from dataloaders.dataloader import FundusSegmentation, ProstateSegmentation, MNMSSegmentation
import dataloaders.custom_transforms as tr
from utils import losses, metrics, ramps, util
from utils.util import to_2d, to_3d
from torch.cuda.amp import autocast, GradScaler
import contextlib

from torch.optim.lr_scheduler import LambdaLR
import math

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='prostate', choices=['fundus', 'prostate', 'MNMS'])
parser.add_argument("--save_name", type=str, default="debug", help="experiment_name")
parser.add_argument("--overwrite", action='store_true')
parser.add_argument("--model", type=str, default="unet", help="model_name")
parser.add_argument("--max_iterations", type=int, default=60000, help="maximum epoch number to train")
parser.add_argument('--num_eval_iter', type=int, default=500)
parser.add_argument("--deterministic", type=int, default=1, help="whether use deterministic training")
parser.add_argument("--base_lr", type=float, default=0.03, help="segmentation network learning rate")
parser.add_argument("--seed", type=int, default=1337, help="random seed")
parser.add_argument("--gpu", type=str, default='0')
parser.add_argument('--load',action='store_true')
parser.add_argument('--load_path',type=str,default='../model/lb1_ratio0.2/iter_6000.pth')
parser.add_argument("--threshold", type=float, default=0.95, help="confidence threshold for using pseudo-labels",)

parser.add_argument('--amp', type=int, default=1, help='use mixed precision training or not')

parser.add_argument("--label_bs", type=int, default=4, help="labeled_batch_size per gpu")
parser.add_argument("--unlabel_bs", type=int, default=4)
parser.add_argument("--test_bs", type=int, default=1)
parser.add_argument('--domain_num', type=int, default=6)
parser.add_argument('--lb_domain', type=int, default=1)
parser.add_argument('--lb_num', type=int, default=40, help="20 for prostate and fundus, 5 for M&MS, and 64 (1/8) or 129 (1/4) for BUSI")
# costs
parser.add_argument("--ema_decay", type=float, default=0.99, help="ema_decay")
parser.add_argument("--consistency_type", type=str, default="mse", help="consistency_type")
parser.add_argument("--consistency", type=float, default=1.0, help="consistency")
parser.add_argument("--consistency_rampup", type=float, default=200.0, help="consistency_rampup")

parser.add_argument('--depth', type=int, default=28)
parser.add_argument('--widen_factor', type=int, default=2)
parser.add_argument('--leaky_slope', type=float, default=0.1)
parser.add_argument('--bn_momentum', type=float, default=0.1)
parser.add_argument('--dropout', type=float, default=0.0)

parser.add_argument('--num_cs', type=int, default=5)
parser.add_argument('--cs_start', type=int, default=5)
parser.add_argument('--cs_update', type=int, default=25)
parser.add_argument('--gen_cls', type=str, default='gens', choices=['genw', 'gens', 'avg'])
parser.add_argument("--w_mcs", default=0.1, type=float, help="weight for mask cs")
parser.add_argument("--w_con", default=1.0, type=float, help="weight for consistency loss")
# proto_loss
parser.add_argument("--lambda_p", default=1.0, type=float, help="weight for proto_loss")
parser.add_argument("--cl_layer", default=3, type=int, help="which layer for proto_loss")
parser.add_argument('--save_img',action='store_true')
args = parser.parse_args()


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # teacher network: ema_model
    # student network: model
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)

def cycle(iterable: Iterable):
    """Make an iterator returning elements from the iterable.

    .. note::
        **DO NOT** use `itertools.cycle` on `DataLoader(shuffle=True)`.\n
        Because `itertools.cycle` saves a copy of each element, batches are shuffled only at the first epoch. \n
        See https://docs.python.org/3/library/itertools.html#itertools.cycle for more details.
    """
    while True:
        for x in iterable:
            yield x

def get_SGD(net, name='SGD', lr=0.1, momentum=0.9, \
                  weight_decay=5e-4, nesterov=True, bn_wd_skip=True):
    '''
    return optimizer (name) in torch.optim.
    If bn_wd_skip, the optimizer does not apply
    weight decay regularization on parameters in batch normalization.
    '''
    optim = getattr(torch.optim, name)
    
    decay = []
    no_decay = []
    for name, param in net.named_parameters():
        if ('bn' in name) and bn_wd_skip:
            no_decay.append(param)
        else:
            decay.append(param)
    
    per_param_args = [{'params': decay},
                      {'params': no_decay, 'weight_decay': 0.0}]
    
    optimizer = optim(per_param_args, lr=lr,
                    momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)
    return optimizer
        
        
def get_cosine_schedule_with_warmup(optimizer,
                                    num_training_steps,
                                    num_cycles=7./16.,
                                    num_warmup_steps=0,
                                    last_epoch=-1):
    '''
    Get cosine scheduler (LambdaLR).
    if warmup is needed, set num_warmup_steps (int) > 0.
    '''
    
    def _lr_lambda(current_step):
        '''
        _lr_lambda returns a multiplicative factor given an interger parameter epochs.
        Decaying criteria: last_epoch
        '''
        
        if current_step < num_warmup_steps:
            _lr = float(current_step) / float(max(1, num_warmup_steps))
        else:
            num_cos_steps = float(current_step - num_warmup_steps)
            num_cos_steps = num_cos_steps / float(max(1, num_training_steps - num_warmup_steps))
            _lr = max(0.0, math.cos(math.pi * num_cycles * num_cos_steps))
        return _lr
    
    return LambdaLR(optimizer, _lr_lambda, last_epoch)


def obtain_loss(ce_loss, dice_loss, pred, label, mask, spw_ce, spw, softmax, sigmoid, multi):
    logits_lb_x_w, logits_ulb_x_s = pred
    lb_mask, pseudo_label = label
    mask_u = mask

    sup_loss = (ce_loss(logits_lb_x_w, lb_mask) * spw_ce[:args.label_bs]).mean() + \
               dice_loss(logits_lb_x_w, lb_mask.unsqueeze(1), batch_w=spw[:args.label_bs], softmax=softmax, sigmoid=sigmoid, multi=multi)

    unsup_loss_s = (ce_loss(logits_ulb_x_s, pseudo_label) * mask_u.squeeze(1) * spw_ce[args.label_bs * 3:]).mean() + \
                   dice_loss(logits_ulb_x_s, pseudo_label.unsqueeze(1), mask=mask_u, batch_w=spw[args.label_bs * 3:],
                             softmax=softmax, sigmoid=sigmoid, multi=multi)

    return sup_loss, unsup_loss_s


def out_transfer(name, pred_label, mask):
    if name == 'fundus':
        pred_label_t = to_2d(pred_label)
        mask_t = to_2d(mask)
        pred_onehot = pred_label_t.clone().cpu()
        mask_onehot = mask_t.clone().cpu()
    elif name == 'prostate':
        pred_label_t = pred_label
        mask_t = mask
        pred_onehot = pred_label.clone().unsqueeze(1).cpu()
        mask_onehot = mask.clone().unsqueeze(1).cpu()
    elif name == 'MNMS':
        pred_label_t = pred_label
        mask_t = mask
        pred_onehot = to_3d(pred_label).cpu()
        mask_onehot = to_3d(mask).cpu()

    return pred_label_t, mask_t, pred_onehot, mask_onehot


@torch.no_grad()
def test(args, model, ema_model, test_dataloader, epoch, apply_conststyle, writer):
    model.eval()
    ema_model.eval()
    model_name = ['ema1', 'ema2', 'stu1', 'stu2']  # ema_cs, ema_mcs, stu_cs, stu_mcs
    val_loss = [0.0, 0.0, 0.0, 0.0]
    val_dice = [[0.0] * n_part, [0.0] * n_part, [0.0] * n_part, [0.0] * n_part]
    domain_num = len(test_dataloader)
    ce_loss = CrossEntropyLoss(reduction='none')
    softmax, sigmoid, multi = True, False, False
    dice_loss = losses.DiceLossWithMask(num_classes)
    for i in range(domain_num):
        cur_dataloader = test_dataloader[i]
        dc = -1
        domain_val_loss = [0.0, 0.0, 0.0, 0.0]
        domain_val_dice = [[0.0] * n_part, [0.0] * n_part, [0.0] * n_part, [0.0] * n_part]
        for batch_num, sample in enumerate(cur_dataloader):
            dc = sample['dc'][0].item()
            data = sample['image'].cuda()
            mask = sample['label'].cuda()
            if args.dataset == 'fundus':
                lb_mask = (mask <= 128) * 2
                lb_mask[mask == 0] = 1
                mask = lb_mask
            elif args.dataset == 'prostate':
                mask = mask.eq(0).long()
            elif args.dataset == 'MNMS':
                mask_ = mask[:, ..., 0].eq(255).float()
                mask_[mask[:, ..., 1].eq(255)] = 2
                mask_[mask[:, ..., 2].eq(255)] = 3
                mask = mask_.long()

            # 在测试阶段，不管student还是teacher都采用cs操作
            output = ema_model(data, dc, store_feature=False, apply_conststyle=apply_conststyle, is_test=True)[0]
            loss_seg = ce_loss(output, mask).mean() + \
                       dice_loss(output, mask.unsqueeze(1), softmax=softmax, sigmoid=sigmoid, multi=multi)
            pred_label = torch.max(torch.softmax(output, dim=1), dim=1)[1]
            pred_label_t, mask_t, pred_onehot, mask_onehot = out_transfer(args.dataset, pred_label, mask)
            dice = dice_calcu[args.dataset](np.asarray(pred_label_t.cpu()), mask_t.clone().cpu())
            domain_val_loss[0] += loss_seg.item()
            for i in range(len(domain_val_dice[0])):
                domain_val_dice[0][i] += dice[i]

            if apply_conststyle:
                cs_prob = torch.softmax(output, dim=1)
                prob, pseudo_label = torch.max(cs_prob, dim=1)
                mask_un = (prob > args.threshold).unsqueeze(1).float()
                label_u = pseudo_label.clone().unsqueeze(1)
                label_u[mask_un == 0] = 255
                output = ema_model(data, dc, label=label_u, store_feature=False, apply_conststyle=apply_conststyle, is_test=True)[0]
            loss_seg = ce_loss(output, mask).mean() + \
                       dice_loss(output, mask.unsqueeze(1), softmax=softmax, sigmoid=sigmoid, multi=multi)
            pred_label = torch.max(torch.softmax(output, dim=1), dim=1)[1]
            pred_label_t, mask_t, pred_onehot, mask_onehot = out_transfer(args.dataset, pred_label, mask)
            dice = dice_calcu[args.dataset](np.asarray(pred_label_t.cpu()), mask_t.clone().cpu())
            domain_val_loss[1] += loss_seg.item()
            for i in range(len(domain_val_dice[1])):
                domain_val_dice[1][i] += dice[i]

            if apply_conststyle:
                # teacher的mcs的输出
                mcs_prob = torch.softmax(output, dim=1)
                cs_entropy = -torch.sum(cs_prob * torch.log(cs_prob + 1e-8), dim=1).mean()
                mcs_entropy = -torch.sum(mcs_prob * torch.log(mcs_prob + 1e-8), dim=1).mean()
                scores = torch.tensor([-cs_entropy.item(), -mcs_entropy.item()]).cuda()
                weights = torch.softmax(scores / 1.0, dim=0)
                prob = weights[0] * cs_prob + weights[1] * mcs_prob
                prob, pseudo_label = torch.max(prob, dim=1)
                mask_un = (prob > args.threshold).unsqueeze(1).float()
                label_u = pseudo_label.clone().unsqueeze(1)
                label_u[mask_un == 0] = 255

            output = model(data, dc, store_feature=False, apply_conststyle=apply_conststyle, is_test=True)[0]
            loss_seg = ce_loss(output, mask).mean() + \
                       dice_loss(output, mask.unsqueeze(1), softmax=softmax, sigmoid=sigmoid, multi=multi)
            pred_label = torch.max(torch.softmax(output, dim=1), dim=1)[1]
            pred_label_t, mask_t, pred_onehot, mask_onehot = out_transfer(args.dataset, pred_label, mask)
            dice = dice_calcu[args.dataset](np.asarray(pred_label_t.cpu()), mask_t.clone().cpu())
            domain_val_loss[2] += loss_seg.item()
            for i in range(len(domain_val_dice[2])):
                domain_val_dice[2][i] += dice[i]

            if apply_conststyle:
                output = model(data, dc, label=label_u, store_feature=False, apply_conststyle=apply_conststyle, is_test=True)[0]
            loss_seg = ce_loss(output, mask).mean() + \
                       dice_loss(output, mask.unsqueeze(1), softmax=softmax, sigmoid=sigmoid, multi=multi)
            pred_label = torch.max(torch.softmax(output, dim=1), dim=1)[1]
            pred_label_t, mask_t, pred_onehot, mask_onehot = out_transfer(args.dataset, pred_label, mask)
            dice = dice_calcu[args.dataset](np.asarray(pred_label_t.cpu()), mask_t.clone().cpu())
            domain_val_loss[3] += loss_seg.item()
            for i in range(len(domain_val_dice[3])):
                domain_val_dice[3][i] += dice[i]

        logging.info('-------------------------------')
        for j in range(4):
            domain_val_loss[j] /= len(cur_dataloader)
            val_loss[j] += domain_val_loss[j]
            writer.add_scalar('{}_val/domain{}/loss'.format(model_name[j], dc), domain_val_loss[j], epoch)
            for i in range(len(domain_val_dice[j])):
                domain_val_dice[j][i] /= len(cur_dataloader)
                val_dice[j][i] += domain_val_dice[j][i]
            for n, p in enumerate(part):
                writer.add_scalar('{}_val/domain{}/val_{}_dice'.format(model_name[j], dc, p), domain_val_dice[j][n], epoch)
            text = 'domain%d epoch %d model %s : loss : %f' % (dc, epoch, model_name[j], domain_val_loss[j])
            for n, p in enumerate(part):
                text += ' val_%s_dice: %f' % (p, domain_val_dice[j][n])
                if n != n_part - 1:
                    text += ','
            logging.info(text)

    model.train()
    ema_model.train()
    logging.info('-------------------------------')
    for j in range(4):
        val_loss[j] /= domain_num
        writer.add_scalar('{}_val/loss'.format(model_name[j]), val_loss[j], epoch)
        for i in range(len(val_dice[j])):
            val_dice[j][i] /= domain_num
        for n, p in enumerate(part):
            writer.add_scalar('{}_val/val_{}_dice'.format(model_name[j], p), val_dice[j][n], epoch)
        text = 'epoch %d model %s : loss : %f' % (epoch, model_name[j], val_loss[j])
        for n, p in enumerate(part):
            text += ' val_%s_dice: %f' % (p, val_dice[j][n])
            if n != n_part - 1:
                text += ','
        logging.info(text)
    return val_dice


def train(args, snapshot_path):
    writer = SummaryWriter(snapshot_path + '/log')
    base_lr = args.base_lr
    max_iterations = args.max_iterations
    max_epoch = max_iterations // args.num_eval_iter

    # data loader
    weak = transforms.Compose([tr.RandomScaleCrop(patch_size),
            tr.RandomScaleRotate(fillcolor=fillcolor),
            tr.RandomHorizontalFlip(),
            tr.elastic_transform(),
            ])
    
    strong = transforms.Compose([
            tr.Brightness(min_v, max_v),
            tr.Contrast(min_v, max_v),
            tr.GaussianBlur(kernel_size=int(0.1 * patch_size), num_channels=num_channels),
    ])

    normal_toTensor = transforms.Compose([
        tr.Normalize_tf(),
        tr.ToTensor()
    ])

    domain_num = args.domain_num
    domain = list(range(1,domain_num+1))
    if args.dataset == 'fundus':
        domain_len = [50, 99, 320, 320, 320, 320][:domain_num]
    elif args.dataset == 'prostate':
        domain_len = [225, 305, 136, 373, 338, 133][:domain_num]
    elif args.dataset == 'MNMS':
        domain_len = [1030, 1342, 525, 550, 1200][:domain_num]
    lb_domain = args.lb_domain
    data_num = domain_len[lb_domain-1]
    lb_num = args.lb_num
    lb_idxs = list(range(lb_num))
    unlabeled_idxs = list(range(lb_num, data_num))
    test_dataset = []
    test_dataloader = []
    lb_dataset = dataset(base_dir=train_data_path, phase='train', splitid=lb_domain, domain=[lb_domain],
                         selected_idxs = lb_idxs, weak_transform=weak,normal_toTensor=normal_toTensor)
    ulb_dataset = dataset(base_dir=train_data_path, phase='train', splitid=lb_domain, domain=domain,
                          selected_idxs=unlabeled_idxs, weak_transform=weak, strong_tranform=strong,normal_toTensor=normal_toTensor)
    for i in range(1, domain_num+1):
        cur_dataset = dataset(base_dir=train_data_path, phase='test', splitid=-1, domain=[i], normal_toTensor=normal_toTensor)
        test_dataset.append(cur_dataset)
    lb_dataloader = cycle(DataLoader(lb_dataset, batch_size = args.label_bs, shuffle=True, num_workers=2, pin_memory=True, drop_last=True))
    ulb_dataloader = cycle(DataLoader(ulb_dataset, batch_size = args.unlabel_bs, shuffle=True, num_workers=2, pin_memory=True, drop_last=True))
    for i in range(0,domain_num):
        cur_dataloader = DataLoader(test_dataset[i], batch_size = args.test_bs, shuffle=False, num_workers=0, pin_memory=True)
        test_dataloader.append(cur_dataloader)

    def create_model(ema=False):
        # Network definition
        if args.model == 'unet':
            model = OptionalConstStyleModel(n_channels=num_channels, n_classes=num_classes, num_conststyle=args.num_cs,
                                            n_cluster=domain_num, cl_layer=args.cl_layer, sp=True, proj=False,
                                            max_epoch=max_epoch, args=args)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model.cuda()

    model = create_model()
    ema_model = create_model(ema=True)

    iter_num = 0
    start_epoch = 0

    # instantiate optimizers
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    # set to train
    ce_loss = CrossEntropyLoss(reduction='none')
    softmax, sigmoid, multi = True, False, False
    dice_loss = losses.DiceLossWithMaskGen(num_classes)  # DiceLossWithMask
    proto_loss = StyleProtoLoss(num_classes)
    logging.info("{} iterations per epoch".format(args.num_eval_iter))

    best_dice = [0.0] * n_part
    best_dice_iter = [-1] * n_part
    best_avg_dice = 0.0
    best_avg_dice_iter = -1
    dice_of_best_avg = [0.0] * n_part
    best2_dice = [0.0] * n_part
    best2_dice_iter = [-1] * n_part
    best2_avg_dice = 0.0
    best2_avg_dice_iter = -1
    dice_of_best2_avg = [0.0] * n_part

    stu_best_dice = [0.0] * n_part
    stu_best_dice_iter = [-1] *n_part
    stu_best_avg_dice = 0.0
    stu_best_avg_dice_iter = -1
    stu_dice_of_best_avg = [0.0] * n_part
    stu2_best_dice = [0.0] * n_part
    stu2_best_dice_iter = [-1] * n_part
    stu2_best_avg_dice = 0.0
    stu2_best_avg_dice_iter = -1
    stu2_dice_of_best_avg = [0.0] * n_part

    iter_num = int(iter_num)
    if args.load:
        path_str = '../model/prostate/dm{}/checkpoint.pth'.format(args.lb_domain)
        start_epoch, ema_model, model, optimizer, best_avg_dice, best_avg_dice_iter, stu_best_avg_dice, stu_best_avg_dice_iter = util.load_osmancheckpoint(
            path_str, ema_model, model, optimizer
        )
        iter_num = start_epoch*args.num_eval_iter
        logging.info('Models restored from epoch {}'.format(start_epoch))

    threshold = args.threshold
    cs_start = args.cs_start
    cs_update = args.cs_update
    w_mcs = args.w_mcs

    scaler = GradScaler()
    amp_cm = autocast if args.amp else contextlib.nullcontext
    for epoch_num in range(start_epoch, max_epoch):
        # 每个epoch前情况CS的风格特征
        for conststyle in model.conststyle:
            conststyle.clear_memory()
        for conststyle in ema_model.conststyle:
            conststyle.clear_memory()

        store_feature = (epoch_num + 1 == cs_start) or \
                        (epoch_num + 1 > cs_start and (epoch_num + 1 - cs_start) % cs_update == 0)
        apply_conststyle = epoch_num >= cs_start
        # w_mcs = args.w_mcs * ramps.cosine_rampdown(epoch_num, max_epoch)

        model.train()
        ema_model.train()
        p_bar = tqdm(range(args.num_eval_iter))
        p_bar.set_description(f'No. {epoch_num+1}')
        for i_batch in range(1, args.num_eval_iter+1):
            lb_sample = next(lb_dataloader)
            ulb_sample = next(ulb_dataloader)
            lb_x_w, lb_y = lb_sample['image'], lb_sample['label']
            ulb_x_w, ulb_x_s, ulb_y = ulb_sample['image'], ulb_sample['strong_aug'], ulb_sample['label']
            lb_dc, ulb_dc = lb_sample['dc'].cuda(), ulb_sample['dc'].cuda()
            move_transx = lb_x_w.cuda()
            lb_x_w, lb_y, ulb_x_w, ulb_x_s, ulb_y = lb_x_w.cuda(), lb_y.cuda(), ulb_x_w.cuda(), ulb_x_s.cuda(), ulb_y.cuda()

            if args.dataset == 'fundus':
                lb_mask = (lb_y<=128) * 2
                lb_mask[lb_y==0] = 1
                ulb_mask = (ulb_y<=128) * 2
                ulb_mask[ulb_y==0] = 1
            elif args.dataset == 'prostate':
                lb_mask = lb_y.eq(0).long()
                ulb_mask = ulb_y.eq(0).long()
            elif args.dataset == 'MNMS':
                lb_mask = lb_y[:,...,0].eq(255).float()
                lb_mask[lb_y[:,...,1].eq(255)] = 2
                lb_mask[lb_y[:,...,2].eq(255)] = 3
                lb_mask = lb_mask.long()
                ulb_mask = ulb_y[:,...,0].eq(255).float()
                ulb_mask[ulb_y[:,...,1].eq(255)] = 2
                ulb_mask[ulb_y[:,...,2].eq(255)] = 3
                ulb_mask = ulb_mask.long()

            with amp_cm():
                with torch.no_grad():
                    logits_ulb_x_w, rep_tu = ema_model(ulb_x_w, ulb_dc, store_feature=store_feature, apply_conststyle=apply_conststyle)[:2]
                    cs_prob = torch.softmax(logits_ulb_x_w, dim=1)
                    prob, pseudo_label = torch.max(cs_prob, dim=1)
                    mask = (prob > threshold).unsqueeze(1).float()

                    if store_feature and (not apply_conststyle):
                        label_u = pseudo_label.clone().unsqueeze(1)
                        label_u[mask == 0] = 255
                        _ = ema_model(ulb_x_w, ulb_dc, label=label_u, store_feature=store_feature,
                                      apply_conststyle=apply_conststyle)[0]

                    if apply_conststyle:
                        # 在teacher上mcs
                        label_u = pseudo_label.clone().unsqueeze(1)
                        label_u[mask == 0] = 255
                        output_u = ema_model(ulb_x_w, ulb_dc, label=label_u, store_feature=store_feature,
                                             apply_conststyle=apply_conststyle)[0]
                        mcs_prob = torch.softmax(output_u, dim=1)
                        # 基于熵整合
                        cs_entropy = -torch.sum(cs_prob * torch.log(cs_prob + 1e-8), dim=1).mean()
                        mcs_entropy = -torch.sum(mcs_prob * torch.log(mcs_prob + 1e-8), dim=1).mean()
                        scores = torch.tensor([-cs_entropy.item(), -mcs_entropy.item()]).cuda()
                        weights = torch.softmax(scores / 1.0, dim=0)
                        prob_ulb_x_w = weights[0] * cs_prob + weights[1] * mcs_prob
                        prob, pseudo_label = torch.max(prob_ulb_x_w, dim=1)
                        mask = (prob > threshold).unsqueeze(1).float()

                # -----For student model-----
                output_l = model(lb_x_w, lb_dc, store_feature=store_feature, apply_conststyle=apply_conststyle)
                output_l_cs, rep_l, spw_l = output_l
                output_u = model(ulb_x_s, ulb_dc, store_feature=store_feature, apply_conststyle=apply_conststyle)
                output_u_cs, rep_u, spw_u = output_u
                # proto loss for cs
                rep_all = torch.cat((rep_l, rep_u))
                if apply_conststyle:
                    rep_loss1 = proto_loss(embed=rep_all, lb_mask=lb_mask, pseudo_labels=pseudo_label, mask=mask)
                else:
                    rep_loss = 0.0 * rep_all.sum()

                # 考虑直连并存储
                if store_feature and (not apply_conststyle):
                    _ = model(lb_x_w, lb_dc, label=lb_mask.clone().unsqueeze(1), store_feature=store_feature,
                              apply_conststyle=apply_conststyle)[0]
                    label_u = pseudo_label.clone().unsqueeze(1)
                    label_u[mask == 0] = 255
                    _ = model(ulb_x_s, ulb_dc, label=label_u, store_feature=store_feature,
                              apply_conststyle=apply_conststyle)[0]

                con_loss = torch.tensor(0.0).to('cuda')
                spw = torch.ones(len(lb_x_w) * 2).to('cuda')
                spw_ce = spw.view(spw.shape + (1,) * 2)
                if apply_conststyle:
                    output_l_mcs, rep_l2 = model(lb_x_w, lb_dc, label=lb_mask.clone().unsqueeze(1),
                                                 store_feature=store_feature, apply_conststyle=apply_conststyle)[:2]

                    label_u = pseudo_label.clone().unsqueeze(1)
                    label_u[mask == 0] = 255
                    output_u_mcs, rep_u2 = model(ulb_x_s, ulb_dc, label=label_u, store_feature=store_feature,
                                                 apply_conststyle=apply_conststyle)[:2]
                    # proto loss for cs
                    rep_all2 = torch.cat((rep_l2, rep_u2))
                    rep_loss2 = proto_loss(embed=rep_all2, lb_mask=lb_mask, pseudo_labels=pseudo_label, mask=mask)

                    pred_1 = [output_l_cs, output_u_cs]
                    label = [lb_mask, pseudo_label]
                    mask_en = [mask]
                    sup_loss_1, unsup_loss_s_1 = obtain_loss(ce_loss, dice_loss, pred_1, label, mask_en, spw_ce, spw, softmax, sigmoid, multi)

                    pred_2 = [output_l_mcs, output_u_mcs]
                    sup_loss_2, unsup_loss_s_2 = obtain_loss(ce_loss, dice_loss, pred_2, label, mask_en, spw_ce, spw, softmax, sigmoid, multi)

                    sup_loss = sup_loss_1 + w_mcs * sup_loss_2
                    unsup_loss_s = unsup_loss_s_1 + w_mcs * unsup_loss_s_2
                    rep_loss = rep_loss1 + w_mcs * rep_loss2

                    # 一致性loss
                    for out1, out2 in zip(pred_1, pred_2):
                        out1_soft, out2_soft = torch.softmax(out1, dim=1), torch.softmax(out2, dim=1)
                        con_loss += torch.mean((out1_soft - out2_soft) ** 2)
                    con_loss /= 2.0

                else:
                    pred = [output_l_cs, output_u_cs]
                    label = [lb_mask, pseudo_label]
                    mask_en = [mask]
                    sup_loss, unsup_loss_s = obtain_loss(ce_loss, dice_loss, pred, label, mask_en, spw_ce, spw, softmax, sigmoid, multi)

                consistency_weight = get_current_consistency_weight(iter_num // (args.max_iterations / args.consistency_rampup))
                loss = sup_loss + consistency_weight * (unsup_loss_s + args.w_con * con_loss + args.lambda_p * rep_loss)

            optimizer.zero_grad()
            if args.amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            # update ema model
            update_ema_variables(model, ema_model, args.ema_decay, iter_num)

            # update learning rate
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1

            if p_bar is not None:
                p_bar.update()

        if p_bar is not None:
            p_bar.close()

        all_val_dice = test(args, model, ema_model, test_dataloader, epoch_num + 1, apply_conststyle, writer)
        logging.info('test ema1 model')
        text = ''
        val_dice = all_val_dice[0]
        for n, p in enumerate(part):
            if val_dice[n] > best_dice[n]:
                best_dice[n] = val_dice[n]
                best_dice_iter[n] = iter_num
            text += 'val_%s_best_dice: %f at %d iter' % (p, best_dice[n], best_dice_iter[n])
            text += ', '
        if sum(val_dice) / len(val_dice) > best_avg_dice:
            best_avg_dice = sum(val_dice) / len(val_dice)
            best_avg_dice_iter = iter_num
            for n, p in enumerate(part):
                dice_of_best_avg[n] = val_dice[n]
            save_text = "{}_avg_dice_best_ema_model.pth".format(args.model)
            save_best = os.path.join(snapshot_path, save_text)
            logging.info('save cur best avg ema model to {}'.format(save_best))
            util.save_dpmodel(ema_model, epoch_num, save_best)
        text += 'val_best_avg_dice: %f at %d iter' % (best_avg_dice, best_avg_dice_iter)
        if n_part > 1:
            for n, p in enumerate(part):
                text += ', %s_dice: %f' % (p, dice_of_best_avg[n])
        logging.info(text)

        logging.info('test ema2 model')
        text = ''
        val_dice = all_val_dice[1]
        for n, p in enumerate(part):
            if val_dice[n] > best2_dice[n]:
                best2_dice[n] = val_dice[n]
                best2_dice_iter[n] = iter_num
            text += 'val_%s_best_dice: %f at %d iter' % (p, best2_dice[n], best2_dice_iter[n])
            text += ', '
        if sum(val_dice) / len(val_dice) > best2_avg_dice:
            best2_avg_dice = sum(val_dice) / len(val_dice)
            best2_avg_dice_iter = iter_num
            for n, p in enumerate(part):
                dice_of_best2_avg[n] = val_dice[n]
            save_text = "{}_avg_dice_best_ema2_model.pth".format(args.model)
            save_best = os.path.join(snapshot_path, save_text)
            logging.info('save cur best avg ema2 model to {}'.format(save_best))
            util.save_dpmodel(ema_model, epoch_num, save_best)
        text += 'val_best_avg_dice: %f at %d iter' % (best2_avg_dice, best2_avg_dice_iter)
        if n_part > 1:
            for n, p in enumerate(part):
                text += ', %s_dice: %f' % (p, dice_of_best2_avg[n])
        logging.info(text)

        logging.info('test stu1 model')
        stu_val_dice = all_val_dice[2]
        text = ''
        for n, p in enumerate(part):
            if stu_val_dice[n] > stu_best_dice[n]:
                stu_best_dice[n] = stu_val_dice[n]
                stu_best_dice_iter[n] = iter_num
            text += 'stu_val_%s_best_dice: %f at %d iter' % (p, stu_best_dice[n], stu_best_dice_iter[n])
            text += ', '
        if sum(stu_val_dice) / len(stu_val_dice) > stu_best_avg_dice:
            stu_best_avg_dice = sum(stu_val_dice) / len(stu_val_dice)
            stu_best_avg_dice_iter = iter_num
            for n, p in enumerate(part):
                stu_dice_of_best_avg[n] = stu_val_dice[n]
            save_text = "{}_avg_dice_best_model.pth".format(args.model)
            save_best = os.path.join(snapshot_path, save_text)
            logging.info('save cur best avg model to {}'.format(save_best))
            util.save_dpmodel(model, epoch_num, save_best)
        text += 'val_best_avg_dice: %f at %d iter' % (stu_best_avg_dice, stu_best_avg_dice_iter)
        if n_part > 1:
            for n, p in enumerate(part):
                text += ', %s_dice: %f' % (p, stu_dice_of_best_avg[n])
        logging.info(text)

        logging.info('test stu2 model')
        stu2_val_dice = all_val_dice[3]
        text = ''
        for n, p in enumerate(part):
            if stu2_val_dice[n] > stu2_best_dice[n]:
                stu2_best_dice[n] = stu2_val_dice[n]
                stu2_best_dice_iter[n] = iter_num
            text += 'stu2_val_%s_best_dice: %f at %d iter' % (p, stu2_best_dice[n], stu2_best_dice_iter[n])
            text += ', '
        if sum(stu2_val_dice) / len(stu2_val_dice) > stu2_best_avg_dice:
            stu2_best_avg_dice = sum(stu2_val_dice) / len(stu2_val_dice)
            stu2_best_avg_dice_iter = iter_num
            for n, p in enumerate(part):
                stu2_dice_of_best_avg[n] = stu2_val_dice[n]
            save_text = "{}_avg_dice_best_model2.pth".format(args.model)
            save_best = os.path.join(snapshot_path, save_text)
            logging.info('save cur best avg model2 to {}'.format(save_best))
            util.save_dpmodel(model, epoch_num, save_best)
        text += 'val_best_avg_dice: %f at %d iter' % (stu2_best_avg_dice, stu2_best_avg_dice_iter)
        if n_part > 1:
            for n, p in enumerate(part):
                text += ', %s_dice: %f' % (p, stu2_dice_of_best_avg[n])
        logging.info(text)

        if store_feature:
            for idx, conststyle in enumerate(model.conststyle):
                conststyle.cal_mean_std(idx, epoch_num)  # 实际上idx, epoch未使用

            for idx, conststyle in enumerate(ema_model.conststyle):
                conststyle.cal_mean_std(idx, epoch_num)  # 实际上idx, epoch未使用

    writer.close()
    # 对最佳ema模型进行评估
    logging.info('test ema with gud')
    epoch = util.load_dpmodel(ema_model, snapshot_path + '/unet_avg_dice_best_ema_model.pth')
    ema_model.eval()
    eval_dprus(args, ema_model, test_dataloader, epoch, gud=True)

    logging.info('test ema with cgud')
    epoch = util.load_dpmodel(ema_model, snapshot_path + '/unet_avg_dice_best_ema2_model.pth')
    ema_model.eval()
    eval_dprus(args, ema_model, test_dataloader, epoch, gud=False)


if __name__ == "__main__":
    if len(args.save_name) == 0:
        args.save_name = f'sprus_{args.model}CP_db{args.dataset}_dm{args.lb_domain}_lb{args.lb_num}'
    snapshot_path = "../model/" + args.dataset + "/" + args.save_name + "/"
    if args.dataset == 'fundus':
        train_data_path = '../SynFoC-MiDSS/Fundus'
        part = ['cup', 'disc']
        dataset = FundusSegmentation
        num_channels = 3
        patch_size = 256
        num_classes = 3
        min_v, max_v = 0.5, 1.5
        fillcolor = 255
        args.max_iterations = 30000
    elif args.dataset == 'prostate':
        train_data_path = "../SynFoC-MiDSS/Prostate"
        num_channels = 1
        patch_size = 384
        num_classes = 2
        part = ['base']
        dataset = ProstateSegmentation
        min_v, max_v = 0.1, 2
        fillcolor = 255
        args.max_iterations = 60000
    elif args.dataset == 'MNMS':
        train_data_path = "../SynFoC-MiDSS/mnms"
        part = ['lv', 'myo', 'rv']
        dataset = MNMSSegmentation
        num_channels = 1
        patch_size = 288
        num_classes = 4
        min_v, max_v = 0.1, 2
        fillcolor = 0
        args.max_iterations = 60000

    if args.lb_num < 8:
        args.label_bs = 2
        args.unlabel_bs = 2
    else:
        args.label_bs = 4
        args.unlabel_bs = 4

    n_part = len(part)
    dice_calcu = {'fundus': metrics.dice_coeff_2label, 'prostate': metrics.dice_coeff, 'MNMS': metrics.dice_coeff_3label}

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    elif not args.overwrite:
        raise Exception('file {} is exist!'.format(snapshot_path))

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    cmd = " ".join(["python"] + sys.argv)
    logging.info(cmd)
    logging.info(str(args))

    args.load_path = snapshot_path
    train(args, snapshot_path)
    logging.info(snapshot_path)
