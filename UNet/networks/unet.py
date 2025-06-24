# -*- coding: utf-8 -*-
"""
The implementation is borrowed from: https://github.com/HiLab-git/PyMIC
"""
from __future__ import division, print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.uniform import Uniform


class ConvBlock(nn.Module):
    """two convolution layers with batch norm and leaky relu"""

    def __init__(self, in_channels, out_channels, dropout_p, norm):
        super(ConvBlock, self).__init__()
        if norm == 'BN':
            self.conv_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(),
                nn.Dropout(dropout_p),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU()
            )
        elif norm == 'IN':
            self.conv_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.InstanceNorm2d(out_channels),
                nn.LeakyReLU(),
                nn.Dropout(dropout_p),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.InstanceNorm2d(out_channels),
                nn.LeakyReLU()
            )

    def forward(self, x):
        return self.conv_conv(x)


class DownBlock(nn.Module):
    """Downsampling followed by ConvBlock"""

    def __init__(self, in_channels, out_channels, dropout_p, norm):
        super(DownBlock, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels, dropout_p, norm)

        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpBlock(nn.Module):
    """Upssampling followed by ConvBlock"""

    def __init__(self, in_channels1, in_channels2, out_channels, dropout_p, bilinear=True, norm='BN'):
        super(UpBlock, self).__init__()
        self.bilinear = bilinear
        if bilinear:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels1, in_channels2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels2 * 2, out_channels, dropout_p, norm)

    def forward(self, x1, x2):
        if self.bilinear:
            x1 = self.conv1x1(x1)
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear_encoder']
        self.dropout = self.params['dropout_encoder']
        self.norm = self.params['norm']
        assert (len(self.ft_chns) == 5)
        self.in_conv = ConvBlock(self.in_chns, self.ft_chns[0], self.dropout[0], self.norm)
        self.down1 = DownBlock(self.ft_chns[0], self.ft_chns[1], self.dropout[1], self.norm)
        self.down2 = DownBlock(self.ft_chns[1], self.ft_chns[2], self.dropout[2], self.norm)
        self.down3 = DownBlock(self.ft_chns[2], self.ft_chns[3], self.dropout[3], self.norm)
        self.down4 = DownBlock(self.ft_chns[3], self.ft_chns[4], self.dropout[4], self.norm)

    def forward(self, x):
        x0 = self.in_conv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        return [x0, x1, x2, x3, x4]


class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.dropout = self.params['dropout_decoder']
        self.bilinear = self.params['bilinear_decoder']
        self.norm = self.params['norm']
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock(self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=self.dropout, bilinear=self.bilinear, norm=self.norm)
        self.up2 = UpBlock(self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=self.dropout, bilinear=self.bilinear, norm=self.norm)
        self.up3 = UpBlock(self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=self.dropout, bilinear=self.bilinear, norm=self.norm)
        self.up4 = UpBlock(self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=self.dropout, bilinear=self.bilinear, norm=self.norm)

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class, kernel_size=3, padding=1)

    def forward(self, feature):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)
        output = self.out_conv(x)
        return output


class Decoder_DS(nn.Module):
    def __init__(self, params):
        super(Decoder_DS, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock(
            self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0)
        self.up2 = UpBlock(
            self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0)
        self.up3 = UpBlock(
            self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0)
        self.up4 = UpBlock(
            self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0)

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class,
                                  kernel_size=3, padding=1)
        self.out_conv_dp4 = nn.Conv2d(self.ft_chns[4], self.n_class,
                                      kernel_size=3, padding=1)
        self.out_conv_dp3 = nn.Conv2d(self.ft_chns[3], self.n_class,
                                      kernel_size=3, padding=1)
        self.out_conv_dp2 = nn.Conv2d(self.ft_chns[2], self.n_class,
                                      kernel_size=3, padding=1)
        self.out_conv_dp1 = nn.Conv2d(self.ft_chns[1], self.n_class,
                                      kernel_size=3, padding=1)

    def forward(self, feature, shape):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]
        x = self.up1(x4, x3)
        dp3_out_seg = self.out_conv_dp3(x)
        dp3_out_seg = torch.nn.functional.interpolate(dp3_out_seg, shape)

        x = self.up2(x, x2)
        dp2_out_seg = self.out_conv_dp2(x)
        dp2_out_seg = torch.nn.functional.interpolate(dp2_out_seg, shape)

        x = self.up3(x, x1)
        dp1_out_seg = self.out_conv_dp1(x)
        dp1_out_seg = torch.nn.functional.interpolate(dp1_out_seg, shape)

        x = self.up4(x, x0)
        dp0_out_seg = self.out_conv(x)
        return dp0_out_seg, dp1_out_seg, dp2_out_seg, dp3_out_seg


class Decoder_URDS(nn.Module):
    def __init__(self, params):
        super(Decoder_URDS, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock(
            self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0)
        self.up2 = UpBlock(
            self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0)
        self.up3 = UpBlock(
            self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0)
        self.up4 = UpBlock(
            self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0)

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class,
                                  kernel_size=3, padding=1)
        self.out_conv_dp4 = nn.Conv2d(self.ft_chns[4], self.n_class,
                                      kernel_size=3, padding=1)
        self.out_conv_dp3 = nn.Conv2d(self.ft_chns[3], self.n_class,
                                      kernel_size=3, padding=1)
        self.out_conv_dp2 = nn.Conv2d(self.ft_chns[2], self.n_class,
                                      kernel_size=3, padding=1)
        self.out_conv_dp1 = nn.Conv2d(self.ft_chns[1], self.n_class,
                                      kernel_size=3, padding=1)
        self.feature_noise = FeatureNoise()

    def forward(self, feature, shape):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]
        x = self.up1(x4, x3)
        if self.training:
            dp3_out_seg = self.out_conv_dp3(Dropout(x, p=0.5))
        else:
            dp3_out_seg = self.out_conv_dp3(x)
        dp3_out_seg = torch.nn.functional.interpolate(dp3_out_seg, shape)

        x = self.up2(x, x2)
        if self.training:
            dp2_out_seg = self.out_conv_dp2(FeatureDropout(x))
        else:
            dp2_out_seg = self.out_conv_dp2(x)
        dp2_out_seg = torch.nn.functional.interpolate(dp2_out_seg, shape)

        x = self.up3(x, x1)
        if self.training:
            dp1_out_seg = self.out_conv_dp1(self.feature_noise(x))
        else:
            dp1_out_seg = self.out_conv_dp1(x)
        dp1_out_seg = torch.nn.functional.interpolate(dp1_out_seg, shape)

        x = self.up4(x, x0)
        dp0_out_seg = self.out_conv(x)
        return dp0_out_seg, dp1_out_seg, dp2_out_seg, dp3_out_seg


def Dropout(x, p=0.5):
    x = torch.nn.functional.dropout2d(x, p)
    return x


def FeatureDropout(x):
    attention = torch.mean(x, dim=1, keepdim=True)
    max_val, _ = torch.max(attention.view(
        x.size(0), -1), dim=1, keepdim=True)
    threshold = max_val * np.random.uniform(0.7, 0.9)
    threshold = threshold.view(x.size(0), 1, 1, 1).expand_as(attention)
    drop_mask = (attention < threshold).float()
    x = x.mul(drop_mask)
    return x


class FeatureNoise(nn.Module):
    def __init__(self, uniform_range=0.3):
        super(FeatureNoise, self).__init__()
        self.uni_dist = Uniform(-uniform_range, uniform_range)

    def feature_based_noise(self, x):
        noise_vector = self.uni_dist.sample(
            x.shape[1:]).to(x.device).unsqueeze(0)
        x_noise = x.mul(noise_vector) + x
        return x_noise

    def forward(self, x):
        x = self.feature_based_noise(x)
        return x

class Lambda(nn.Module):
    def __init__(self, func, **kwargs):
        super().__init__()

        self.func = func
        self.kwargs = kwargs

    def forward(self, *args, **kwargs):
        return self.func(*args, **kwargs, **self.kwargs)
    

class UNet(nn.Module):
    def __init__(self, in_chns, class_num, bilinear_encoder=False, bilinear_decoder=True, norm='BN', multiscale=False,
                 embed_dim_sam=2816, embed_dim_unet=448, proj_dim=256):
        super(UNet, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout_encoder': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'dropout_decoder': 0.0,
                  'class_num': class_num,
                  'bilinear_encoder': bilinear_encoder,
                  'bilinear_decoder': bilinear_decoder,
                  'acti_func': 'relu',
                  'norm': norm}

        self.encoder = Encoder(params)
        self.decoder = Decoder(params)

        if multiscale:
            self.projection_head_sam = nn.Sequential(
                nn.Flatten(start_dim=0, end_dim=-2),
                nn.Linear(embed_dim_sam, embed_dim_sam),
                nn.BatchNorm1d(embed_dim_sam),
                nn.ReLU(),
                nn.Linear(embed_dim_sam, embed_dim_sam),
                nn.BatchNorm1d(embed_dim_sam),
                nn.ReLU(),
                nn.Linear(embed_dim_sam, proj_dim),
                Lambda(F.normalize)
            )
            self.projection_head_unet = nn.Sequential(
                nn.Flatten(start_dim=0, end_dim=-2),
                nn.Linear(embed_dim_unet, embed_dim_unet),
                nn.BatchNorm1d(embed_dim_unet),
                nn.ReLU(),
                nn.Linear(embed_dim_unet, embed_dim_unet),
                nn.BatchNorm1d(embed_dim_unet),
                nn.ReLU(),
                nn.Linear(embed_dim_unet, proj_dim),
                Lambda(F.normalize)
            )

    def forward(self, x, kd=False):
        feature = self.encoder(x)
        output = self.decoder(feature)
        if kd:
            return feature, output
        else:
            return output
        
class UNet_multiscale_nonorm(nn.Module):
    def __init__(self, in_chns, class_num, bilinear_encoder=False, bilinear_decoder=True, norm='BN', embed_dim_sam=2816, embed_dim_unet=448, proj_dim=256):
        super(UNet_multiscale_nonorm, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout_encoder': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'dropout_decoder': 0.0,
                  'class_num': class_num,
                  'bilinear_encoder': bilinear_encoder,
                  'bilinear_decoder': bilinear_decoder,
                  'acti_func': 'relu',
                  'norm': norm}

        self.encoder = Encoder(params)
        self.decoder = Decoder(params)
        # self.projection_head_sam = nn.Sequential(
        #     nn.Flatten(start_dim=0, end_dim=-2),
        #     nn.Linear(embed_dim_sam, embed_dim_sam),
        #     nn.BatchNorm1d(embed_dim_sam),
        #     nn.ReLU(),
        #     nn.Linear(embed_dim_sam, proj_dim)
        # )
        # self.projection_head_unet = nn.Sequential(
        #     nn.Flatten(start_dim=0, end_dim=-2),
        #     nn.Linear(embed_dim_unet, embed_dim_unet),
        #     nn.BatchNorm1d(embed_dim_unet),
        #     nn.ReLU(),
        #     nn.Linear(embed_dim_unet, proj_dim)
        # )        

        self.projection_head_sam = nn.Sequential(
            nn.Flatten(start_dim=0, end_dim=-2),
            nn.Linear(embed_dim_sam, embed_dim_sam),
            nn.BatchNorm1d(embed_dim_sam),
            nn.ReLU(),
            nn.Linear(embed_dim_sam, embed_dim_sam),
            nn.BatchNorm1d(embed_dim_sam),
            nn.ReLU(),
            nn.Linear(embed_dim_sam, proj_dim)
        )
        self.projection_head_unet = nn.Sequential(
            nn.Flatten(start_dim=0, end_dim=-2),
            nn.Linear(embed_dim_unet, embed_dim_unet),
            nn.BatchNorm1d(embed_dim_unet),
            nn.ReLU(),
            nn.Linear(embed_dim_unet, embed_dim_unet),
            nn.BatchNorm1d(embed_dim_unet),
            nn.ReLU(),
            nn.Linear(embed_dim_unet, proj_dim)
        )
        
    def forward(self, x, kd=False):
        feature = self.encoder(x)
        output = self.decoder(feature)
        if kd:
            return feature, output
        else:
            return output
        

class UNet_projection_head(nn.Module):
    def __init__(self, in_chns, class_num, bilinear_encoder=False, bilinear_decoder=True, norm='BN', embed_dim_sam=256, embed_dim_unet=256, proj_dim=256):
        super(UNet_projection_head, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout_encoder': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'dropout_decoder': 0.0,
                  'class_num': class_num,
                  'bilinear_encoder': bilinear_encoder,
                  'bilinear_decoder': bilinear_decoder,
                  'acti_func': 'relu',
                  'norm': norm}

        self.encoder = Encoder(params)
        self.decoder = Decoder(params)

        self.projection_head_sam = nn.Sequential(
            nn.Flatten(start_dim=0, end_dim=-2),
            nn.Linear(embed_dim_sam, embed_dim_sam),
            nn.BatchNorm1d(embed_dim_sam),
            nn.ReLU(),
            nn.Linear(embed_dim_sam, proj_dim)
        )
        self.projection_head_unet = nn.Sequential(
            nn.Flatten(start_dim=0, end_dim=-2),
            nn.Linear(embed_dim_unet, embed_dim_unet),
            nn.BatchNorm1d(embed_dim_unet),
            nn.ReLU(),
            nn.Linear(embed_dim_unet, proj_dim)
        )
        
    def forward(self, x, kd=False):
        feature = self.encoder(x)
        output = self.decoder(feature)
        if kd:
            return feature, output
        else:
            return output
        
class UNet_DS(nn.Module):
    def __init__(self, in_chns, class_num):
        super(UNet_DS, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}
        self.encoder = Encoder(params)
        self.decoder = Decoder_DS(params)

    def forward(self, x):
        shape = x.shape[2:]
        feature = self.encoder(x)
        dp0_out_seg, dp1_out_seg, dp2_out_seg, dp3_out_seg = self.decoder(
            feature, shape)
        return dp0_out_seg, dp1_out_seg, dp2_out_seg, dp3_out_seg


class UNet_CCT(nn.Module):
    def __init__(self, in_chns, class_num):
        super(UNet_CCT, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}
        self.encoder = Encoder(params)
        self.main_decoder = Decoder(params)
        self.aux_decoder1 = Decoder(params)

    def forward(self, x):
        feature = self.encoder(x)
        main_seg = self.main_decoder(feature)
        aux1_feature = [Dropout(i) for i in feature]
        aux_seg1 = self.aux_decoder1(aux1_feature)
        return main_seg, aux_seg1


class UNet_CCT_3H(nn.Module):
    def __init__(self, in_chns, class_num):
        super(UNet_CCT_3H, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}
        self.encoder = Encoder(params)
        self.main_decoder = Decoder(params)
        self.aux_decoder1 = Decoder(params)
        self.aux_decoder2 = Decoder(params)

    def forward(self, x):
        feature = self.encoder(x)
        main_seg = self.main_decoder(feature)
        aux1_feature = [Dropout(i) for i in feature]
        aux_seg1 = self.aux_decoder1(aux1_feature)
        aux2_feature = [FeatureNoise()(i) for i in feature]
        aux_seg2 = self.aux_decoder1(aux2_feature)
        return main_seg, aux_seg1, aux_seg2
    
    


class BiNet(nn.Module):
    def __init__(self, in_chns=1, class_num=2):
        super(BiNet, self).__init__()
        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}

        self.encoder1 = Encoder(params)
        self.decoder1 = Decoder(params)

        self.encoder2 = Encoder(params)
        self.decoder2 = Decoder(params)
        
    def forward(self, x, training=False):
        feature1 = self.encoder1(x)
        out1 = self.decoder1(feature1)

        feature2 = self.encoder2(x)
        out2 = self.decoder2(feature2)

        if (training):
          return out1, out2
        else:
          return (out1 + out2) / 2
    
    
class TriNet(nn.Module):
    def __init__(self, in_chns, class_num):
        super(TriNet, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}

        self.encoder1 = Encoder(params)
        self.decoder1 = Decoder(params)

        self.encoder2 = Encoder(params)
        self.decoder2 = Decoder(params)

        self.encoder3 = Encoder(params)
        self.decoder3 = Decoder(params)

    def forward(self, x, training=False):
        feature1 = self.encoder1(x)
        output1 = self.decoder1(feature1)

        feature2 = self.encoder2(x)
        output2 = self.decoder2(feature2)

        feature3 = self.encoder3(x)
        output3 = self.decoder3(feature3)

        if (training):
          return output1, output2, output3
        else:
          return (output1 + output2 + output3) / 3
    