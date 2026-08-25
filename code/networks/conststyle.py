import torch.nn as nn
import numpy as np
from sklearn.mixture import BayesianGaussianMixture
import torch
import torch.nn.functional as F
from dassl.modeling.backbone.cnn_digitsdg import avg_wass_distance


class ConstStyle(nn.Module):
    def __init__(self, idx, num_clusters=6, eps=1e-6):
        super().__init__()
        self.idx = idx
        self.eps = eps
        self.alpha_test = 0.0  # 0.5
        self.ema_decay = 0.0
        # store instance style
        self.mean = []
        self.std = []

        self.domain = []
        # mean and cov of unified domain
        self.const_mean = None
        self.const_cov = None
        self.generator = None
        # GMM clustering state
        self.bayes_cluster = None
        self.num_clusters = num_clusters
    
    def clear_memory(self):
        self.mean = []
        self.std = []
        self.domain = []
        
    def get_style(self, x):
        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        var = (var + self.eps).sqrt()
        mu, var = mu.detach().squeeze().cpu().numpy(), var.detach().squeeze().cpu().numpy()
        return mu, var
    
    def store_style(self, x, domain):
        mu, var = self.get_style(x)
        self.mean.extend(mu)
        self.std.extend(var)
        self.domain.extend(domain.detach().squeeze().cpu().numpy())
    
    def cal_mean_std(self, idx, epoch):
        """ determine the unified domain"""
        mean_list = np.array(self.mean)
        std_list = np.array(self.std)
        stacked_data = np.stack((mean_list, std_list), axis=1)
        reshaped_data = stacked_data.reshape((len(mean_list), -1))

        print('Clustering using GMM')
        print(f'Number of cluster: {self.num_clusters}')
        self.bayes_cluster = BayesianGaussianMixture(n_components=self.num_clusters, covariance_type='full', init_params='kmeans', max_iter=200)  # k-means++
        self.bayes_cluster.fit(reshaped_data)

        labels = self.bayes_cluster.predict(reshaped_data)
        unique_labels, _ = np.unique(labels, return_counts=True)

        means = [self.bayes_cluster.means_[i] for i in range(len(unique_labels))]
        covs = [self.bayes_cluster.covariances_[i] for i in range(len(unique_labels))]

        gen_cls = 'gens'  # 'ori'
        if gen_cls == 'genw':
            # 根据(- wass_distance)进行unified domain构造
            means_ts = torch.from_numpy(np.array(means))
            covs_ts = torch.from_numpy(np.array(covs))
            avg_dist = avg_wass_distance(means, covs)
            domain_gen = torch.from_numpy(-avg_dist).unsqueeze(1)
            domain_gen = torch.softmax(domain_gen, dim=0)
            const_mean = (domain_gen * means_ts).sum(dim=0)
            const_cov = (domain_gen.unsqueeze(dim=-1) * covs_ts).sum(dim=0)
        elif gen_cls == 'gens':
            # 根据平均域相似度进行unified domain构造
            means_ts = torch.from_numpy(np.array(means))
            covs_ts = torch.from_numpy(np.array(covs))
            means_norm = torch.nn.functional.normalize(means_ts, p=2, dim=1)
            # domain_gen = torch.matmul(means_norm, means_norm.T).mean(dim=1).unsqueeze(1)  # n,1
            domain_gen = torch.matmul(means_norm, means_norm.T)
            mask = ~torch.eye(domain_gen.shape[0], dtype=torch.bool)
            domain_gen = (domain_gen * mask).sum(dim=1, keepdim=True) / mask.sum(dim=1, keepdim=True)  # n,1
            domain_gen = torch.softmax(domain_gen, dim=0)  # n,1
            const_mean = (domain_gen * means_ts).sum(dim=0)
            const_cov = (domain_gen.unsqueeze(dim=-1) * covs_ts).sum(dim=0)
        else:
            # ori method
            cluster_mean = np.mean(means, axis=0)
            cluster_cov = np.mean(covs, axis=0)
            const_mean = torch.from_numpy(cluster_mean)
            const_cov = torch.from_numpy(cluster_cov)

        # EMA更新unified domain统计量
        if self.const_mean is None:
            self.const_mean = const_mean
            self.const_cov = const_cov
        else:
            self.const_mean = self.ema_decay * self.const_mean + (1-self.ema_decay) * const_mean
            self.const_cov = self.ema_decay * self.const_cov + (1 - self.ema_decay) * const_cov
        self.generator = torch.distributions.MultivariateNormal(loc=self.const_mean, covariance_matrix=self.const_cov)

        return reshaped_data

    def zero_grad_g(self):
        """仅作为占位，后续子类重写"""
        pass

    def forward(self, x, domain, store_feature=False, apply_conststyle=False, is_test=False):
        if store_feature:
            self.store_style(x, domain)
        
        if not apply_conststyle:  # (not is_test and np.random.random() > 1.0) or not apply_conststyle
            return x
        
        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        x_normed = (x - mu) / sig
        
        if is_test:
            const_value = torch.reshape(self.const_mean, (2, -1))
            const_mean = const_value[0].float().to('cuda')
            const_std = const_value[1].float().to('cuda')
            
            const_mean = torch.reshape(const_mean, (1, const_mean.shape[0], 1, 1))
            const_std = torch.reshape(const_std, (1, const_std.shape[0], 1, 1))
            
            if self.alpha_test:
                beta = (1 - self.alpha_test) * const_std + self.alpha_test * sig
                gamma = (1 - self.alpha_test) * const_mean + self.alpha_test * mu
            else:
                beta = const_std
                gamma = const_mean
            
        else:
            style_mean = []
            style_std = []
            for i in range(len(x_normed)):
                style = self.generator.sample()
                style = torch.reshape(style, (2, -1))
                style_mean.append(style[0])
                style_std.append(style[1])
            
            const_mean = torch.vstack(style_mean).float()
            const_std = torch.vstack(style_std).float()
            
            const_mean = torch.reshape(const_mean, (const_mean.shape[0], const_mean.shape[1], 1, 1)).to('cuda')
            const_std = torch.reshape(const_std, (const_std.shape[0], const_std.shape[1], 1, 1)).to('cuda')

            beta = const_std
            gamma = const_mean
                
        out = x_normed * beta + gamma
            
        return out


class MultiConstStyle(ConstStyle):
    def __init__(self, idx, num_clusters=6, num_cls=4, eps=1e-6):
        super().__init__(idx, num_clusters, eps)
        self.ema_decay = 0.0
        self.num_cls = num_cls
        self.mean_cls = [[] for _ in range(self.num_cls)]
        self.std_cls = [[] for _ in range(self.num_cls)]
        self.const_mean_cls = [None for _ in range(self.num_cls)]
        self.const_cov_cls = [None for _ in range(self.num_cls)]
        self.generator_cls = [None for _ in range(self.num_cls)]
        # 这里直接不考虑domain信息

    def clear_memory(self):
        # 利用统一域计算
        super().clear_memory()
        # 利于类别域计算
        self.mean_cls = [[] for _ in range(self.num_cls)]
        self.std_cls = [[] for _ in range(self.num_cls)]

    def get_global_cls_style(self, x, label, save=True, cs=False):
        """
        save==True:
            保存域风格：【每张图片】独立计算均值和标准差
            保存类域风格：【每张图片】+【每种类别】独立计算均值和标准差 (student后L层所特有)
        cs==True:
            prob=0.5 (mcs)
            可靠类特征：(F^c - mu^c) / sig^c
            可靠单一类特征或者不可靠特征：(F - mu) / sig
            prob=0.5 (cs)
            所有特征：(F - mu) / sig
        param:
        x: feature of b,c,h,w
        label: gt+pl of b,1,h,w {0,1,2,3,255}
        save: whether save style statistic or not
        cs: whether apply cs or not

        return: x_normed, mcs
        """
        # 用于统一域
        # mu, var = super().get_style(x)
        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        if save:
            self.mean.extend(mu.squeeze().cpu().numpy())
            self.std.extend(sig.squeeze().cpu().numpy())
            if label is not None:
                self.get_x_normed(x, label, mu, sig, save, cs=False)  # 仅保存类域风格

        # 用于类域
        x_normed = None
        mcs = False
        if cs:
            if label is None:
                # 传统CS层：用于student test|teacher|非对比层，整体归一化
                x_normed = (x - mu) / sig
            else:
                # BCS层 or CS层：用于student train的对比层，分块/整体归一化
                mcs = torch.rand(1) < 0.5
                x_normed = self.get_x_normed(x, label, mu, sig, save=False, cs=cs) if mcs else (x - mu) / sig
        return x_normed, mcs

    def get_x_normed(self, x, label, gmu, gsig, save=True, cs=False):
        """ 按【每张图片】+【每种类别】独立计算均值和标准差，对特征进行归一化
        param:
        x: feature of b,c,h,w
        label: gt+pl of b,1,h,w {0,1,2,3,255}
        """
        b, feat_c, h, w = x.shape
        x_normed = []
        for img_idx in range(b):
            x_img = x[img_idx]  # feat_c, h, w
            label_img = label[img_idx]  # 1, h, w
            classes_in_img = torch.unique(label_img)

            x_img = x_img.permute(1, 2, 0).reshape(-1, feat_c)  # h*w, feat_c
            label_img = label_img.reshape(-1)  # h*w
            x_img_normed = x_img.clone()  # 初始化归一化张量为原始值
            for cls in classes_in_img:
                cls_int = cls.item()
                mask = (label_img == cls_int)
                # 计算【该图片内该类别】的特征均值和标准差
                x_img_cls = x_img[mask]  # n_pixels, feat_c
                if x_img_cls.shape[0] > 1 and cls_int != 255:
                    mu = x_img_cls.mean(dim=0, keepdim=True)  # 1,feat_c
                    var = x_img_cls.var(dim=0, keepdim=True)
                    sig = (var + self.eps).sqrt()
                    mu, sig = mu.detach(), sig.detach()
                    if save:
                        self.mean_cls[int(cls)].extend(mu.cpu().numpy())
                        self.std_cls[int(cls)].extend(sig.cpu().numpy())
                else:
                    if cs:
                        # 不可靠或者可靠只有一个样本
                        mu = gmu[img_idx].view(1, -1)
                        sig = gsig[img_idx].view(1, -1)
                if cs:
                    # 3. 对该图片内该类别的特征进行归一化
                    # x_img[mask] = (x_img[mask].clone() - mu) / sig
                    x_img_normed[mask] = (x_img[mask] - mu) / sig
            if cs:
                x_normed.append(x_img_normed.reshape(h, w, feat_c).permute(2, 0, 1))   # feat_c,h,w
        if cs:
            x_normed = torch.stack(x_normed, dim=0)
            return x_normed
        else:
            return None

    def cal_single_mean_std(self, mean, std, const_mean, const_cov):
        """ determine the unified domain"""
        mean_list = np.array(mean)
        std_list = np.array(std)
        stacked_data = np.stack((mean_list, std_list), axis=1)
        reshaped_data = stacked_data.reshape((len(mean_list), -1))

        print('Clustering using GMM')
        print(f'Number of cluster: {self.num_clusters}')
        bayes_cluster = BayesianGaussianMixture(n_components=self.num_clusters, covariance_type='full', init_params='kmeans', max_iter=200)  # k-means++
        bayes_cluster.fit(reshaped_data)

        labels = bayes_cluster.predict(reshaped_data)
        unique_labels, _ = np.unique(labels, return_counts=True)
        means = [bayes_cluster.means_[i] for i in range(len(unique_labels))]
        covs = [bayes_cluster.covariances_[i] for i in range(len(unique_labels))]

        gen_cls = 'gens'  # 'ori'
        if gen_cls == 'genw':
            # 根据(- wass_distance)进行unified domain构造
            means_ts = torch.from_numpy(np.array(means))
            covs_ts = torch.from_numpy(np.array(covs))
            avg_dist = avg_wass_distance(means, covs)
            domain_gen = torch.from_numpy(-avg_dist).unsqueeze(1)
            domain_gen = torch.softmax(domain_gen, dim=0)
            cur_mean = (domain_gen * means_ts).sum(dim=0)
            cur_cov = (domain_gen.unsqueeze(dim=-1) * covs_ts).sum(dim=0)
        elif gen_cls == 'gens':
            # 根据平均域相似度进行unified domain构造
            means_ts = torch.from_numpy(np.array(means))
            covs_ts = torch.from_numpy(np.array(covs))
            means_norm = torch.nn.functional.normalize(means_ts, p=2, dim=1)
            # domain_gen = torch.matmul(means_norm, means_norm.T).mean(dim=1).unsqueeze(1)  # n,1
            domain_gen = torch.matmul(means_norm, means_norm.T)
            print(domain_gen)
            mask = ~torch.eye(domain_gen.shape[0], dtype=torch.bool)
            domain_gen = (domain_gen * mask).sum(dim=1, keepdim=True) / mask.sum(dim=1, keepdim=True)  # n,1
            domain_gen = torch.softmax(domain_gen, dim=0)  # n,1
            cur_mean = (domain_gen * means_ts).sum(dim=0)
            cur_cov = (domain_gen.unsqueeze(dim=-1) * covs_ts).sum(dim=0)
        else:
            # ori method
            cluster_mean = np.mean(means, axis=0)
            cluster_cov = np.mean(covs, axis=0)
            cur_mean = torch.from_numpy(cluster_mean)
            cur_cov = torch.from_numpy(cluster_cov)

        # EMA更新unified domain统计量
        if const_mean is None:
            const_mean = cur_mean
            const_cov = cur_cov
        else:
            const_mean = self.ema_decay * const_mean + (1 - self.ema_decay) * cur_mean
            const_cov = self.ema_decay * const_cov + (1 - self.ema_decay) * cur_cov

        return const_mean, const_cov

    def cal_mean_std(self, idx, epoch):
        if len(self.mean) > 0:
            self.const_mean, self.const_cov = self.cal_single_mean_std(mean=self.mean, std=self.std,
                                                             const_mean=self.const_mean, const_cov=self.const_cov)
            self.generator = torch.distributions.MultivariateNormal(loc=self.const_mean, covariance_matrix=self.const_cov)

        for i in range(self.num_cls):
            if len(self.mean_cls[i]) == 0:  # 针对teacher情况
                continue
            self.const_mean_cls[i], self.const_cov_cls[i] = self.cal_single_mean_std(mean=self.mean_cls[i], std=self.std_cls[i],
                                                                       const_mean=self.const_mean_cls[i],
                                                                       const_cov=self.const_cov_cls[i])
            self.generator_cls[i] = torch.distributions.MultivariateNormal(loc=self.const_mean_cls[i],
                                                                    covariance_matrix=self.const_cov_cls[i])

    def forward(self, x, domain, store_feature=False, apply_conststyle=False, is_test=False, label=None):
        if store_feature or apply_conststyle:
            x_normed, mcs = self.get_global_cls_style(x, label, save=store_feature, cs=apply_conststyle)

        if not apply_conststyle:
            return (x, torch.ones(x.shape[0]).to('cuda'))

        # 根据不相似度计算样本权重
        w_sp = torch.ones(x.shape[0]).to('cuda')

        if is_test:
            const_value = torch.reshape(self.const_mean, (2, -1))
            const_mean = const_value[0].float().to('cuda')
            const_std = const_value[1].float().to('cuda')
            const_mean = torch.reshape(const_mean, (1, const_mean.shape[0], 1, 1))
            const_std = torch.reshape(const_std, (1, const_std.shape[0], 1, 1))

        elif (label is None) or (not mcs):
            # teacher训练 或者 student训练使用CS
            style_mean = []
            style_std = []
            for i in range(len(x_normed)):
                style = self.generator.sample()
                style = torch.reshape(style, (2, -1))
                style_mean.append(style[0])
                style_std.append(style[1])

            const_mean = torch.vstack(style_mean).float()
            const_std = torch.vstack(style_std).float()

            const_mean = torch.reshape(const_mean, (const_mean.shape[0], const_mean.shape[1], 1, 1)).to('cuda')
            const_std = torch.reshape(const_std, (const_std.shape[0], const_std.shape[1], 1, 1)).to('cuda')

        else:
            # student训练：域风格采样 | 每张图片独立域风格
            b, feat_c, h, w = x_normed.shape
            const_mean, const_std = torch.zeros(x_normed.shape), torch.zeros(x_normed.shape)
            for img_idx in range(b):
                label_img = label[img_idx, 0]  # h,w
                classes_in_img = torch.unique(label_img)
                for cls in classes_in_img:
                    cls_int = cls.item()
                    mask = (label_img == cls_int)  # h,w
                    style = self.generator.sample() if cls_int == 255 else self.generator_cls[int(cls)].sample()
                    style = torch.reshape(style, (2, -1)).float()  # 2,c
                    const_mean[img_idx, :, mask] = style[0].unsqueeze(1)
                    const_std[img_idx, :, mask] = style[1].unsqueeze(1)

            const_mean, const_std = const_mean.to('cuda'), const_std.to('cuda')

        beta = const_std
        gamma = const_mean
        out = x_normed * beta + gamma

        return (out, w_sp)


class OptionalConstStyle(MultiConstStyle):
    """选择性调用CS和MaskCS"""
    def __init__(self, idx, num_clusters=6, num_cls=4, eps=1e-6):
        super().__init__(idx, num_clusters, num_cls, eps)
        # 用于mcs的域风格
        self.mean_mcs = []
        self.std_mcs = []
        self.const_mean_mcs = None
        self.const_cov_mcs = None
        self.generator_mcs = None

    def clear_memory(self):
        super().clear_memory()
        self.mean_mcs = []
        self.std_mcs = []

    def get_global_cls_style(self, x, label, save=True, cs=False):
        """
        single path: teacher, student前L层，测试
        save==True:
            保存域风格：【每张图片】独立计算均值和标准差
            保存类域风格：【每张图片】+【每种类别】独立计算均值和标准差 (student后L层所特有)
        cs==True:
            mcs
            可靠类特征：(F^c - mu^c) / sig^c
            可靠单一类特征或者不可靠特征：(F - mu) / sig
            cs
            所有特征：(F - mu) / sig
        param:
        x: feature of b,c,h,w
        label: gt+pl of b,1,h,w {0,1,2,3,255}
        save: whether save style statistic or not
        cs: whether apply cs or not

        return: x_normed, dual_path
        """
        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        mcs = False if label is None else True
        x_normed = None
        if not mcs:
            # CS
            if save:
                self.mean.extend(mu.squeeze().cpu().numpy())
                self.std.extend(sig.squeeze().cpu().numpy())
            if cs:
                x_normed = (x - mu) / sig
            return x_normed, mcs

        if save:
            self.mean_mcs.extend(mu.squeeze().cpu().numpy())
            self.std_mcs.extend(sig.squeeze().cpu().numpy())
            self.get_x_normed(x, label, mu, sig, save, cs=False)  # 仅保存类域风格

        if cs:
            x_normed = self.get_x_normed(x, label, mu, sig, save=False, cs=cs)

        return x_normed, mcs

    def cal_mean_std(self, idx, epoch):
        super().cal_mean_std(idx, epoch)
        # 统计mcs的域风格
        if len(self.mean_mcs) > 0:
            self.const_mean_mcs, self.const_cov_mcs = self.cal_single_mean_std(mean=self.mean_mcs, std=self.std_mcs,
                                                                               const_mean=self.const_mean_mcs,
                                                                               const_cov=self.const_cov_mcs)
            self.generator_mcs = torch.distributions.MultivariateNormal(loc=self.const_mean_mcs,
                                                                        covariance_matrix=self.const_cov_mcs)

    def sample_cs_style(self, x_normed, is_test=False):
        if is_test:
            const_value = torch.reshape(self.const_mean, (2, -1))
            const_mean = const_value[0].float().to('cuda')
            const_std = const_value[1].float().to('cuda')
            const_mean = torch.reshape(const_mean, (1, const_mean.shape[0], 1, 1))
            const_std = torch.reshape(const_std, (1, const_std.shape[0], 1, 1))
            return const_mean, const_std

        style_mean = []
        style_std = []
        for i in range(len(x_normed)):
            style = self.generator.sample()
            style = torch.reshape(style, (2, -1))
            style_mean.append(style[0])
            style_std.append(style[1])

        const_mean = torch.vstack(style_mean).float()
        const_std = torch.vstack(style_std).float()

        const_mean = torch.reshape(const_mean, (const_mean.shape[0], const_mean.shape[1], 1, 1)).to('cuda')
        const_std = torch.reshape(const_std, (const_std.shape[0], const_std.shape[1], 1, 1)).to('cuda')

        return const_mean, const_std

    def sample_mcs_style(self, x_normed, label, is_test=False):
        # student训练：域风格采样 | 每张图片独立域风格
        const_mean, const_std = torch.zeros(x_normed.shape), torch.zeros(x_normed.shape)
        # const_mean, const_std = torch.zeros_like(x_normed), torch.zeros_like(x_normed)
        for img_idx in range(x_normed.shape[0]):
            label_img = label[img_idx, 0]  # h,w
            classes_in_img = torch.unique(label_img)
            for cls in classes_in_img:
                cls_int = cls.item()
                mask = (label_img == cls_int)  # h,w
                if is_test:
                    style = self.const_mean_mcs if cls_int == 255 else self.const_mean_cls[int(cls)]
                else:
                    style = self.generator_mcs.sample() if cls_int == 255 else self.generator_cls[int(cls)].sample()
                style = torch.reshape(style, (2, -1)).float()  # 2,c
                const_mean[img_idx, :, mask] = style[0].unsqueeze(1)
                const_std[img_idx, :, mask] = style[1].unsqueeze(1)

        const_mean, const_std = const_mean.to('cuda'), const_std.to('cuda')

        return const_mean, const_std

    def forward(self, x, domain, store_feature=False, apply_conststyle=False, is_test=False, label=None):
        if store_feature or apply_conststyle:
            x_normed, mcs = self.get_global_cls_style(x.clone(), label, save=store_feature, cs=apply_conststyle)

        if not apply_conststyle:
            return (x, torch.ones(x.shape[0]).to('cuda'))

        w_sp = torch.ones(x.shape[0]).to('cuda')
        if not mcs:
            gamma, beta = self.sample_cs_style(x_normed, is_test=is_test)
            out = x_normed * beta + gamma
        else:
            gamma, beta = self.sample_mcs_style(x_normed, label, is_test=is_test)
            out = x_normed * beta + gamma

        return (out, w_sp)




      
