import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn


def label_onehot(inputs, num_class):
    '''
    inputs is class label
    return one_hot label
    dim will be increasee
    '''
    batch_size, image_h, image_w = inputs.shape
    inputs = torch.relu(inputs)
    outputs = torch.zeros([batch_size, num_class, image_h, image_w]).to(inputs.device)
    return outputs.scatter_(1, inputs.unsqueeze(1), 1.0)


def negative_index_sampler(samp_num, seg_num_list):
    negative_index = []
    for i in range(samp_num.shape[0]):
        for j in range(samp_num.shape[1]):
            negative_index += np.random.randint(low=sum(seg_num_list[: j]),
                                                high=sum(seg_num_list[: j + 1]),
                                                size=int(samp_num[i, j])).tolist()

    return negative_index


class ContrastLoss(nn.Module):
    def __init__(self, num_class, weight, temp):
        super(ContrastLoss, self).__init__()
        self.num_class = num_class
        self.weight = weight
        self.temp = temp
        self.num_queries = 256
        self.num_negatives = 512
        self.strong_threshold = 0.8  # 0.8

    @torch.no_grad()
    def obtain_mix_label_mask(self, logits_x, rep, threshold):
        rep_b, rep_dim, rep_w, rep_h = rep.shape
        rep = rep.permute(0, 2, 3, 1)

        pseudo_logits_cls, pseudo_labels_cls = torch.max(torch.softmax(logits_x, dim=1), dim=1)
        mask = (pseudo_logits_cls > threshold).unsqueeze(1).float()
        label_small = F.interpolate(pseudo_labels_cls.unsqueeze(1).float(), (rep_w, rep_h), mode='nearest')
        mask_small = F.interpolate(mask, (rep_w, rep_h), mode='nearest')

        proto_rep_list = []  # \hat{p}(t)
        for i in range(self.num_class):
            valid_pixel = (label_small == i) * mask_small
            if valid_pixel.sum() == 0:
                return pseudo_logits_cls, pseudo_labels_cls, mask, None, None
            proto_rep_ = torch.mean((rep[valid_pixel.squeeze(1).bool()]), dim=0, keepdim=True)  # p_c(t) 1,d
            proto_rep_list.append(proto_rep_)
        proto_rep = torch.cat(proto_rep_list)

        # 在表征空间中确定伪标签
        norm_rep_u = F.normalize(rep, dim=-1)  # b,h,w,64
        norm_proto = F.normalize(proto_rep, dim=-1).permute(1, 0)  # 64,c
        norm_rep_u = norm_rep_u.reshape(rep_b * rep_w * rep_h, rep_dim)
        sim_mat = torch.mm(norm_rep_u, norm_proto)  # bhw,c
        sim_mat = sim_mat.reshape(rep_b, rep_w, rep_h, self.num_class).permute(0, 3, 1, 2)  # b,c,h,w
        sim_mat = F.interpolate(sim_mat, size=logits_x.shape[2:], mode='bilinear', align_corners=True)
        # indicator for representation space Eq. (5)
        pseudo_logits_rep, pseudo_labels_rep = torch.max(F.softmax(sim_mat / self.temp, dim=1), dim=1)

        eq_mask = pseudo_labels_cls.eq(pseudo_labels_rep).unsqueeze(1).float()  # b,1,hl,wl
        clean_mask = mask * eq_mask
        noise_mask = mask * (1-eq_mask)  # (1-mask) * eq_mask
        diff_mask = 1-eq_mask

        return pseudo_logits_cls, pseudo_labels_cls, clean_mask, noise_mask, diff_mask

    def anchor_sampling(self, rep, valid_pixel_all, prob):
        """在保证所采pixel特征可靠的前提下，从每张图片每一类c（且满足类c的可靠个数大于max_views）分别从容易和难中各选一半pixel特征出来"""
        batch_size = rep.shape[0]  # rep:b,h,w,d
        num_segments = valid_pixel_all.shape[1]  # b,c,h,w
        anchor = []
        label = []
        rep_all_list = []
        num_list = []
        loss = 0.0
        cls_n = 0
        for i in range(num_segments):
            global_valid_pixel = valid_pixel_all[:, i]
            assert global_valid_pixel.sum() > 0
            global_rep_pool = rep[global_valid_pixel.bool()]  # n,d
            rep_all_list.append(global_rep_pool)
            num_list.append(int(global_valid_pixel.sum().item()))
            cls_rep_hard = []
            k = self.num_queries // batch_size  # 每张图要采样的特征数
            proto_style = []
            for img_i in range(batch_size):
                valid_pixel = valid_pixel_all[img_i, i]  # h,w
                if valid_pixel.sum() == 0:
                    continue
                proto_style.append(torch.mean((rep[img_i][valid_pixel.bool()]), dim=0, keepdim=True))  # 1,d
                prob_seg = prob[img_i, i, :, :]
                rep_mask_hard = (prob_seg < self.strong_threshold) * valid_pixel.bool()
                if rep_mask_hard.sum() == 0:
                    continue
                rep_hard = rep[img_i][rep_mask_hard]  # n,d
                if rep_hard.shape[0] > k:
                    sample_idx = torch.randint(rep_hard.shape[0], size=(k,))
                    rep_hard = rep_hard[sample_idx]
                cls_rep_hard.append(rep_hard)

            if len(cls_rep_hard) > 0:
                # 从全局池中补足num_queries个特征
                cls_rep_hard = torch.cat(cls_rep_hard, dim=0)
                fill_k = self.num_queries - cls_rep_hard.shape[0]
                if fill_k > 0:
                    sample_idx = torch.randint(global_rep_pool.shape[0], size=(fill_k,))
                    fill_rep = global_rep_pool[sample_idx]
                    cls_rep_hard = torch.cat([cls_rep_hard, fill_rep], dim=0)
            anchor.append(cls_rep_hard)
            label.append(i)

            # 类风格去除(1 - 平均余弦相似度)
            if len(proto_style) > 1:
                proto_style = torch.cat(proto_style, dim=0)
                proto_mean = torch.mean(proto_style, dim=0, keepdim=True).detach()
                # proto_mean = torch.mean(global_rep_pool, dim=0, keepdim=True).detach()
                logits = torch.cosine_similarity(proto_style, proto_mean, dim=1)  # b
                loss += 1 - logits.mean()
                cls_n += 1

        loss = loss / cls_n if cls_n > 0 else torch.tensor(0.0) + (0 * rep.sum())

        return anchor, label, rep_all_list, num_list, loss

    def contrast(self, rep, label, mask):
        # we gather all representations (mu and sigma) cross mutiple GPUs during this progress
        batch_size, num_feat, rep_w, rep_h = rep.shape
        num_segments = label.shape[1]  # 21
        valid_pixel_all = label * mask
        # Permute representation for indexing" [batch, rep_h, rep_w, feat_num]
        rep = rep.permute(0, 2, 3, 1)

        proto_rep_list = []  # \hat{p}(t)
        for i in range(num_segments):
            valid_pixel = valid_pixel_all[:, i]
            if valid_pixel.sum() == 0:
                return torch.tensor(0.0) + 0 * rep.sum(), torch.tensor(0.0) + 0 * rep.sum()
            with torch.no_grad():
                proto_rep_ = torch.mean((rep[valid_pixel.bool()]), dim=0, keepdim=True)  # p_c(t) 1,d
                proto_rep_list.append(proto_rep_)

        proto_rep = torch.cat(proto_rep_list)  # num_segments,d
        norm_rep_all = F.normalize(rep, dim=-1)
        norm_rep_all = norm_rep_all.reshape(-1, num_feat)
        norm_proto = F.normalize(proto_rep, dim=-1).permute(1, 0)  # d,c
        prob = torch.mm(norm_rep_all, norm_proto)
        prob = prob.reshape(batch_size, rep_w, rep_h, self.num_class).permute(0, 3, 1, 2)
        prob = F.softmax(prob / self.temp, dim=1)

        # rep_all_list = []
        # rep_hard_list = []
        # num_list = []
        # for i in range(num_segments):
        #     valid_pixel = valid_pixel_all[:, i]
        #     if valid_pixel.sum() == 0:
        #         continue
        #     prob_seg = prob[:, i, :, :]
        #     rep_mask_hard = (prob_seg < self.strong_threshold) * valid_pixel.bool()  # Only on single card
        #
        #     rep_all_list.append(rep[valid_pixel.bool()])
        #     rep_hard_list.append(rep[rep_mask_hard])
        #     num_list.append(int(valid_pixel.sum().item()))
        anchor, _, rep_all_list, num_list, proto_loss = self.anchor_sampling(rep, valid_pixel_all, prob)
        # anchor, label, rep_all_list, num_list, proto_rep 要一一对应

        # Compute Probabilistic Representation Contrastive Loss
        if (len(num_list) <= 1):  # in some rare cases, a small mini-batch only contain 1 or no semantic class
            return torch.tensor(0.0) + 0 * rep.sum(), proto_loss
        else:
            contrast_loss = torch.tensor(0.0)
            valid_num = len(num_list)
            seg_len = torch.arange(valid_num)
            for i in range(valid_num):
                if len(anchor[i]) == 0:
                    continue
                anchor_rep = anchor[i]
                with torch.no_grad():
                    # Select negatives
                    # the similarity between the prototype of current class c and other prototypes.
                    id_mask = torch.cat(([seg_len[i: ], seg_len[: i]]))
                    # 这里是计算cosine，所以不要求归一化
                    proto_sim = torch.cosine_similarity(proto_rep[id_mask[0]].unsqueeze(0), proto_rep[id_mask[1:]], dim=1)
                    proto_prob = torch.softmax(proto_sim / self.temp, dim=0)
                    # Creates a categorical distribution parameterized by either probs or logits (but not both).
                    negative_dist = torch.distributions.categorical.Categorical(probs=proto_prob)
                    samp_class = negative_dist.sample(sample_shape=[self.num_queries, self.num_negatives])
                    # samp_num 记录每个anchor在每个负类中采样多少个负样本
                    samp_num = torch.stack([(samp_class == c).sum(1) for c in range(len(proto_prob))], dim=1)
                    negative_num_list = num_list[ i +1: ] + num_list[: i]
                    negative_index = negative_index_sampler(samp_num, negative_num_list)
                    negative_rep_all = torch.cat(rep_all_list[ i +1: ] + rep_all_list[: i])
                    negative_rep = negative_rep_all[negative_index].reshape(self.num_queries, self.num_negatives, num_feat)
                    positive_rep = proto_rep[i].unsqueeze(0).unsqueeze(0).repeat(self.num_queries, 1, 1)
                    all_rep = torch.cat((positive_rep, negative_rep), dim=1)

                logits = torch.cosine_similarity(anchor_rep.unsqueeze(1), all_rep, dim=2)
                contrast_loss = contrast_loss + F.cross_entropy(logits / self.temp, torch.zeros(self.num_queries).long().cuda())

            return contrast_loss / valid_num, proto_loss

    def forward(self, embed, lb_mask, pseudo_labels, mask):
        with torch.no_grad():
            mask_all = torch.cat(((lb_mask.unsqueeze(1) >= 0).float(), mask))  # b,1,h,w
            mask_all = F.interpolate(mask_all, size=embed.shape[2:], mode='nearest')

            label_l = F.interpolate(label_onehot(lb_mask, self.num_class), size=embed.shape[2:], mode='nearest')
            label_u = F.interpolate(label_onehot(pseudo_labels, self.num_class), size=embed.shape[2:], mode='nearest')
            label_all = torch.cat((label_l, label_u))

        contrast_loss, proto_loss = self.contrast(embed, label_all, mask_all)

        return contrast_loss, proto_loss


class StyleProtoLoss(nn.Module):
    def __init__(self, num_class, temp=1.0):
        super(StyleProtoLoss, self).__init__()
        self.num_class = num_class
        self.temp = temp

    def intra_class_loss(self, rep, valid_pixel_all):
        """在保证所采pixel特征可靠的前提下，从每张图片每一类c（且满足类c的可靠个数大于max_views）分别从容易和难中各选一半pixel特征出来"""
        batch_size = rep.shape[0]  # rep:b,h,w,d
        num_segments = valid_pixel_all.shape[1]  # b,c,h,w
        loss = 0.0
        cls_n = 0
        for i in range(num_segments):
            global_valid_pixel = valid_pixel_all[:, i]  # b,h,w
            assert global_valid_pixel.sum() > 0
            proto_style = []
            for img_i in range(batch_size):
                valid_pixel = valid_pixel_all[img_i, i]  # h,w
                if valid_pixel.sum() == 0:
                    continue
                proto_style.append(torch.mean((rep[img_i][valid_pixel.bool()]), dim=0, keepdim=True))  # 1,d

            # 类风格去除(1 - 平均余弦相似度)
            if len(proto_style) > 1:
                proto_style = torch.cat(proto_style, dim=0)
                proto_mean = torch.mean(proto_style, dim=0, keepdim=True).detach()
                # proto_mean = torch.mean(global_rep_pool, dim=0, keepdim=True).detach()
                logits = torch.cosine_similarity(proto_style, proto_mean, dim=1)  # b
                loss += 1 - logits.mean()
                cls_n += 1

        loss = loss / cls_n if cls_n > 0 else torch.tensor(0.0) + (0 * rep.sum())

        return loss

    def forward(self, embed, lb_mask, pseudo_labels, mask):
        with torch.no_grad():
            mask_all = torch.cat(((lb_mask.unsqueeze(1) >= 0).float(), mask))  # b,1,h,w
            mask_all = F.interpolate(mask_all, size=embed.shape[2:], mode='nearest')

            label_l = F.interpolate(label_onehot(lb_mask, self.num_class), size=embed.shape[2:], mode='nearest')
            label_u = F.interpolate(label_onehot(pseudo_labels, self.num_class), size=embed.shape[2:], mode='nearest')
            label_all = torch.cat((label_l, label_u))

        valid_pixel_all = label_all * mask_all
        rep = embed.permute(0, 2, 3, 1)
        proto_loss = self.intra_class_loss(rep, valid_pixel_all)

        return proto_loss


