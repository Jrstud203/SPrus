"""用于评估DPrus及其衍生品"""
import logging
import numpy as np
import torch
from medpy.metric import binary
from . import metrics
from . import util


def to_2d(input_tensor):
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
def eval_dprus(args, model, test_dataloader, epoch, gud=False):
    # 预先知识
    if args.dataset == 'fundus':
        part = ['cup', 'disc']
    elif args.dataset == 'prostate':
        part = ['base']
    elif args.dataset == 'MNMS':
        part = ['lv', 'myo', 'rv']
    n_part = len(part)
    dice_calcu = {'fundus': metrics.dice_coeff_2label, 'prostate': metrics.dice_coeff, 'MNMS': metrics.dice_coeff_3label}

    model.eval()
    val_dice = [0.0] * n_part
    val_dc, val_jc, val_hd, val_asd = [0.0] * n_part, [0.0] * n_part, [0.0] * n_part, [0.0] * n_part
    domain_num = len(test_dataloader)
    num = 0
    for i in range(domain_num):
        cur_dataloader = test_dataloader[i]
        domain_val_dice = [0.0] * n_part
        domain_val_dc, domain_val_jc, domain_val_hd, domain_val_asd = [0.0] * n_part, [0.0] * n_part, [0.0] * n_part, [0.0] * n_part
        domain_code = i + 1
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

            # gud
            output = model(data, dc, store_feature=False, apply_conststyle=True, is_test=True)[0]
            pred_label = torch.max(torch.softmax(output, dim=1), dim=1)[1]
            pred_label_t, mask_t, pred_onehot, mask_onehot = out_transfer(args.dataset, pred_label, mask)
            dice = dice_calcu[args.dataset](np.asarray(pred_label_t.cpu()), mask_t.clone().cpu())
            if not gud:
                # cgud
                cs_prob = torch.softmax(output, dim=1)
                prob, pseudo_label = torch.max(cs_prob, dim=1)
                mask_un = (prob > args.threshold).unsqueeze(1).float()
                label_u = pseudo_label.clone().unsqueeze(1)
                label_u[mask_un == 0] = 255
                output = model(data, dc, label=label_u, store_feature=False, apply_conststyle=True, is_test=True)[0]
                pred_label = torch.max(torch.softmax(output, dim=1), dim=1)[1]
                pred_label_t, mask_t, pred_onehot, mask_onehot = out_transfer(args.dataset, pred_label, mask)
                dice = dice_calcu[args.dataset](np.asarray(pred_label_t.cpu()), mask_t.clone().cpu())
            avg_dice = sum(dice) / len(dice)

            if args.save_img:
                for j in range(len(data)):
                    num += 1
                    util.draw_mask_and_save(data[j], pred_onehot[j],
                                            '{}/pred_{}/{}_{}_{}.png'.format(args.load_path, 'gud' if gud else 'cgud',
                                                                             domain_code, num, round(avg_dice, 4)))

            dc, jc, hd, asd = [0.0] * n_part, [0.0] * n_part, [0.0] * n_part, [0.0] * n_part
            for j in range(len(data)):
                for i, p in enumerate(part):
                    dc[i] += binary.dc(np.asarray(pred_onehot[j, i], dtype=bool),
                                       np.asarray(mask_onehot[j, i], dtype=bool))
                    jc[i] += binary.jc(np.asarray(pred_onehot[j, i], dtype=bool),
                                       np.asarray(mask_onehot[j, i], dtype=bool))
                    if pred_onehot[j, i].float().sum() < 1e-4:
                        hd[i] += 100
                        asd[i] += 100
                    else:
                        hd[i] += binary.hd95(np.asarray(pred_onehot[j, i], dtype=bool),
                                             np.asarray(mask_onehot[j, i], dtype=bool))
                        asd[i] += binary.asd(np.asarray(pred_onehot[j, i], dtype=bool),
                                             np.asarray(mask_onehot[j, i], dtype=bool))
            for i, p in enumerate(part):
                dc[i] /= len(data)
                jc[i] /= len(data)
                hd[i] /= len(data)
                asd[i] /= len(data)
            for i in range(len(domain_val_dice)):
                domain_val_dice[i] += dice[i]
                domain_val_dc[i] += dc[i]
                domain_val_jc[i] += jc[i]
                domain_val_hd[i] += hd[i]
                domain_val_asd[i] += asd[i]

        for i in range(len(domain_val_dice)):
            domain_val_dice[i] /= len(cur_dataloader)
            val_dice[i] += domain_val_dice[i]
            domain_val_dc[i] /= len(cur_dataloader)
            val_dc[i] += domain_val_dc[i]
            domain_val_jc[i] /= len(cur_dataloader)
            val_jc[i] += domain_val_jc[i]
            domain_val_hd[i] /= len(cur_dataloader)
            val_hd[i] += domain_val_hd[i]
            domain_val_asd[i] /= len(cur_dataloader)
            val_asd[i] += domain_val_asd[i]
        text = 'domain%d epoch %d :' % (domain_code, epoch)
        text += '\n\t'
        for n, p in enumerate(part):
            text += 'val_%s_dice: %f, ' % (p, domain_val_dice[n])
        text += '\n\t'
        for n, p in enumerate(part):
            text += 'val_%s_dc: %f, ' % (p, domain_val_dc[n])
        text += '\t'
        for n, p in enumerate(part):
            text += 'val_%s_jc: %f, ' % (p, domain_val_jc[n])
        text += '\n\t'
        for n, p in enumerate(part):
            text += 'val_%s_hd: %f, ' % (p, domain_val_hd[n])
        text += '\t'
        for n, p in enumerate(part):
            text += 'val_%s_asd: %f, ' % (p, domain_val_asd[n])
        logging.info(text)

    # model.train()
    for i in range(len(val_dice)):
        val_dice[i] /= domain_num
        val_dc[i] /= domain_num
        val_jc[i] /= domain_num
        val_hd[i] /= domain_num
        val_asd[i] /= domain_num
    text = 'epoch %d :' % (epoch)
    text += '\n\t'
    avg = 0.0
    for n, p in enumerate(part):
        text += 'val_%s_dice: %f, ' % (p, val_dice[n])
        avg += val_dice[n]
    avg = avg / len(val_dice)
    text += 'val_avg_dice: %f, ' % (avg)

    text += '\n\t'
    avg = 0.0
    for n, p in enumerate(part):
        text += 'val_%s_dc: %f, ' % (p, val_dc[n])
        avg += val_dc[n]
    avg = avg / len(val_dc)
    text += 'val_avg_dc: %f, ' % (avg)
    text += '\t'

    avg = 0.0
    for n, p in enumerate(part):
        text += 'val_%s_jc: %f, ' % (p, val_jc[n])
        avg += val_jc[n]
    avg = avg / len(val_jc)
    text += 'val_avg_jc: %f, ' % (avg)
    text += '\n\t'

    avg = 0.0
    for n, p in enumerate(part):
        text += 'val_%s_hd: %f, ' % (p, val_hd[n])
        avg += val_hd[n]
    avg = avg / len(val_hd)
    text += 'val_avg_hd: %f, ' % (avg)
    text += '\t'

    avg = 0.0
    for n, p in enumerate(part):
        text += 'val_%s_asd: %f, ' % (p, val_asd[n])
        avg += val_asd[n]
    avg = avg / len(val_asd)
    text += 'val_avg_asd: %f, ' % (avg)
    logging.info(text)
    # return val_dice, val_dc, val_jc, val_hd, val_asd