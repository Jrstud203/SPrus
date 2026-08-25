""" Full assembly of the parts to form the complete network """
from .conststyle import ConstStyle, OptionalConstStyle
from .unet_parts import *


class EmbeddingHead(nn.Module):
    def __init__(self, dim_in, embed_dim=256, embed='convmlp'):
        super(EmbeddingHead, self).__init__()
        if embed == 'linear':
            self.embed = nn.Conv2d(dim_in, embed_dim, kernel_size=1)
        elif embed == 'convmlp':
            self.embed = nn.Sequential(
                nn.Conv2d(dim_in, dim_in, kernel_size=1),
                nn.BatchNorm2d(dim_in),
                nn.ReLU(),
                nn.Conv2d(dim_in, embed_dim, kernel_size=1)
            )

    def forward(self, x):
        return F.normalize(self.embed(x), p=2, dim=1)


class ConstStyleClModel(nn.Module):
    def __init__(self, n_channels, n_classes, num_conststyle=3, n_cluster=6, cl_layer=-2,
                 sp=False, proj=True, max_epoch=120, args=None, bilinear=False):
        super(ConstStyleClModel, self).__init__()
        self.num_conststyle = num_conststyle
        self.conststyle = [ConstStyle(i, num_clusters=n_cluster) for i in range(self.num_conststyle)]
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.cl_layer = cl_layer
        self.bilinear = bilinear
        self.sp = sp
        self.proj = proj

        self.inc = DoubleConv(self.n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, self.n_classes)

        if proj:
            encoder_channels = (64, 128, 256, 512, 1024)
            self.embed_head = EmbeddingHead(dim_in=encoder_channels[self.cl_layer], embed_dim=256)

    def forward(self, x, domain, return_feature=False, store_feature=False, apply_conststyle=False, is_test=False):
        batch_w = 0.
        x1 = self.inc(x)
        if self.num_conststyle >= 1:
            x1 = self.conststyle[0](x1, domain, store_feature=store_feature, apply_conststyle=apply_conststyle,
                                    is_test=is_test)
            if self.sp:
                x1, w1 = x1
                batch_w += w1
        x2 = self.down1(x1)
        if self.num_conststyle >= 2:
            x2 = self.conststyle[1](x2, domain, store_feature=store_feature, apply_conststyle=apply_conststyle,
                                    is_test=is_test)
            if self.sp:
                x2, w2 = x2
                batch_w += w2
        x3 = self.down2(x2)
        if self.num_conststyle >= 3:
            x3 = self.conststyle[2](x3, domain, store_feature=store_feature, apply_conststyle=apply_conststyle,
                                    is_test=is_test)
            if self.sp:
                x3, w3 = x3
                batch_w += w3
        x4 = self.down3(x3)
        if self.num_conststyle >= 4:
            x4 = self.conststyle[3](x4, domain, store_feature=store_feature, apply_conststyle=apply_conststyle,
                                    is_test=is_test)
            if self.sp:
                x4, w4 = x4
                # batch_w += w4
        x5 = self.down4(x4)
        if self.num_conststyle >= 5:
            x5 = self.conststyle[4](x5, domain, store_feature=store_feature, apply_conststyle=apply_conststyle,
                                    is_test=is_test)
            if self.sp:
                x5, w5 = x5
                # batch_w += w5
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)

        feat = [x1, x2, x3, x4, x5]
        embed = self.embed_head(feat[self.cl_layer]) if self.proj else feat[self.cl_layer]

        if return_feature:
            return logits, embed, x
        else:
            return (logits, embed) if not self.sp else (logits, embed, batch_w / 3)  # batch_w / self.num_conststyle


class OptionalConstStyleModel(ConstStyleClModel):
    def __init__(self, n_channels, n_classes, num_conststyle=3, n_cluster=6, cl_layer=-2,
                 sp=False, proj=True, max_epoch=120, args=None, bilinear=False):
        super().__init__(n_channels, n_classes, num_conststyle, n_cluster, cl_layer,
                         sp, proj, max_epoch, args, bilinear)
        self.conststyle = [OptionalConstStyle(i, num_clusters=n_cluster, num_cls=n_classes)
                           for i in range(self.num_conststyle)]

    def forward(self, x, domain, label=None, return_feature=False,
                store_feature=False, apply_conststyle=False, is_test=False):
        batch_w = 0.
        x1 = self.inc(x)
        if self.num_conststyle >= 1:
            x1 = self.conststyle[0](x1, domain, store_feature=store_feature, apply_conststyle=apply_conststyle,
                                    is_test=is_test, label=None)
            if self.sp:
                x1, w1 = x1
                batch_w += w1
        x2 = self.down1(x1)
        if self.num_conststyle >= 2:
            x2 = self.conststyle[1](x2, domain, store_feature=store_feature, apply_conststyle=apply_conststyle,
                                    is_test=is_test, label=None)
            if self.sp:
                x2, w2 = x2
                # batch_w += w2
        x3 = self.down2(x2)
        if self.num_conststyle >= 3:
            label_small = None if label is None else F.interpolate(label.float(), size=x3.shape[2:], mode='nearest')
            x3 = self.conststyle[2](x3, domain, store_feature=store_feature, apply_conststyle=apply_conststyle,
                                    is_test=is_test, label=label_small)
            if self.sp:
                x3, w3 = x3
                # batch_w += w3
        x4 = self.down3(x3)
        if self.num_conststyle >= 4:
            label_small = None if label is None else F.interpolate(label.float(), size=x4.shape[2:], mode='nearest')
            x4 = self.conststyle[3](x4, domain, store_feature=store_feature, apply_conststyle=apply_conststyle,
                                    is_test=is_test, label=label_small)
            if self.sp:
                x4, w4 = x4
                # batch_w += w4
        x5 = self.down4(x4)
        if self.num_conststyle >= 5:
            label_small = None if label is None else F.interpolate(label.float(), size=x5.shape[2:], mode='nearest')
            x5 = self.conststyle[4](x5, domain, store_feature=store_feature, apply_conststyle=apply_conststyle,
                                    is_test=is_test, label=label_small)
            if self.sp:
                x5, w5 = x5
                # batch_w += w5
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)  # b,c,h,w

        feat = [x1, x2, x3, x4, x5]
        embed = self.embed_head(feat[self.cl_layer]) if self.proj else feat[self.cl_layer]

        if return_feature:
            return logits, embed, x
        else:
            return (logits, embed) if not self.sp else (logits, embed, batch_w / 1)
