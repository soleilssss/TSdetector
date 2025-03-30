#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import torch
import torch.nn as nn

from .darknet import BaseConv, CSPDarknet, CSPLayer, DWConv
import torch.nn.functional as F
from .tadaconv import TAda
class CondConv2d(nn.Module):
    

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=1, dilation=1, groups=1, bias=True,
                 num_experts=1):
        super(CondConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.num_experts = num_experts

        self.avgpool = nn.AdaptiveAvgPool3d((None,1,1))
        self.temporalconv=nn.Conv3d(in_channels, in_channels, (3,1,1))
        self.fc=nn.Conv3d(in_channels, 1, (3,1,1))


        self.weight = nn.Parameter(
            torch.Tensor(1,1,out_channels, in_channels // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1,1,out_channels))
        else:
            self.register_parameter('bias', None)
        
        for m in self.modules():
            if isinstance(m,nn.Conv3d):
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)
			
    def generateweight(self,xet):
	    #校准权重
	
        xet=xet.permute(0,2,1,3,4)  #x BxCxLxHxW
        xet=self.avgpool(xet) #x BxCxLx1x1
        
        
        allxet=torch.cat((xet[:,:,0,:,:].unsqueeze(2),xet[:,:,0,:,:].unsqueeze(2),xet),2)
        calibration=self.temporalconv(allxet)
        
        finalweight=self.weight*(calibration+1).unsqueeze(0).permute(1,3,0,2,4,5)
		
        bias=self.bias*(self.fc(allxet)+1).squeeze().unsqueeze(-1)
        
        
        return finalweight,bias,allxet
    
    def forward(self, x): #x B*L*C*W*H
	    #校准权重
        finalweight,finalbias,_=self.generateweight(x)
        
        b,l, c_in, h, w = x.size()
        
        
        x=x.reshape(1,-1,h,w)
        finalweight=finalweight.reshape(-1,self.in_channels,self.kernel_size ,self.kernel_size )
        finalbias=finalbias.view(-1)
        		
        	
        if self.bias is not None:
            
            output = F.conv2d(
                x, weight=finalweight, bias=finalbias, stride=self.stride, padding=self.padding,
                dilation=self.dilation, groups=b*l)
        else:
            output = F.conv2d(
                x, weight=finalweight, bias=None, stride=self.stride, padding=self.padding,
                dilation=self.dilation, groups=b*l)
        
        output = output.view(-1, self.out_channels, output.size(-2), output.size(-1))
        return output

def cubes_2_maps(cubes):
    b, c, d, h, w = cubes.shape
    cubes = cubes.permute(0, 2, 1, 3, 4)

    return cubes.contiguous().view(b*d, c, h, w), b, d

def maps_2_cubes(x, b, d):
    x_b, x_c, x_h, x_w = x.shape
    x = x.contiguous().view(b, d, x_c, x_h, x_w)

    return x.permute(0, 2, 1, 3, 4)

def rev_maps(maps, b, d):
    """reverse maps temporarily."""
    cubes = maps_2_cubes(maps, b, d).flip(dims=[2])

    return cubes_2_maps(cubes)[0]

class R_3(nn.Module):
    def __init__(self, block_channel, use_bn=False):
        super(R_3, self).__init__()

        num_features = int(block_channel+8)
        self.conv0 = nn.Conv2d(num_features, num_features,
                               kernel_size=5, stride=1, padding=2, bias=False)
        self.bn0 = nn.BatchNorm2d(num_features)

        self.conv1 = nn.Conv2d(num_features, num_features,
                               kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features)

        # self.conv2 = nn.Conv2d(
        #     num_features, 1, kernel_size=5, stride=1, padding=2, bias=True)

        self.conv2 = nn.Conv2d(
            num_features, int(block_channel), kernel_size=5, stride=1, padding=2, bias=True)

        self.convh = nn.Conv2d(
            num_features, 8, kernel_size=3, stride=1, padding=1, bias=True)

        self.use_bn = use_bn

    def forward(self, x):

        x0 = self.conv0(x)
        if self.use_bn:
            x0 = self.bn0(x0)
        x0 = F.relu(x0)

        x1 = self.conv1(x0)
        if self.use_bn:
            x1 = self.bn1(x1)
        x1 = F.relu(x1)

        h = self.convh(x1)
        out = self.conv2(x1)

        return h, out

class R_CLSTM_5(nn.Module):
    def __init__(self, block_channel, use_bn=False):
        super(R_CLSTM_5, self).__init__()
        num_features = int(block_channel)
        self.Refine = R_3(block_channel, use_bn=use_bn)
        self.F_t = nn.Sequential(
            nn.Conv2d(in_channels=int(num_features + 8),
                      out_channels=8,
                      kernel_size=3,
                      padding=1,
                      ),
            nn.Sigmoid()
        )
        self.I_t = nn.Sequential(
            nn.Conv2d(in_channels=int(num_features + 8),
                      out_channels=8,
                      kernel_size=3,
                      padding=1,
                      ),
            nn.Sigmoid()
        )
        self.C_t = nn.Sequential(
            nn.Conv2d(in_channels=int(num_features + 8),
                      out_channels=8,
                      kernel_size=3,
                      padding=1,
                      ),
            nn.Tanh()
        )
        self.Q_t = nn.Sequential(
            nn.Conv2d(in_channels=int(num_features + 8),
                      out_channels=num_features,
                      kernel_size=3,
                      padding=1,
                      ),
            nn.Sigmoid()
        )

    def forward(self, input_tensor, b, d):
        input_tensor = maps_2_cubes(input_tensor, b, d)
        b, c, d, h, w = input_tensor.shape
        h_state_init = torch.zeros(b, 8, h, w).to('cuda')
        c_state_init = torch.zeros(b, 8, h, w).to('cuda')

        seq_len = d

        h_state, c_state = h_state_init, c_state_init
        output_inner = []
        for t in range(seq_len):
            input_cat = torch.cat((input_tensor[:, :, t, :, :], h_state), dim=1)
            c_state = self.F_t(input_cat) * c_state + self.I_t(input_cat) * self.C_t(input_cat)

            h_state, p_depth = self.Refine(torch.cat((c_state, self.Q_t(input_cat)), 1))

            output_inner.append(p_depth)

        layer_output = torch.stack(output_inner, dim=2)

        return layer_output

class YOLOXHead(nn.Module):
    def __init__(self, num_classes, width = 1.0, in_channels = [256, 512, 1024], act = "silu", depthwise = False,):
        super().__init__()
        Conv            = DWConv if depthwise else BaseConv
        
        self.cls_convs  = nn.ModuleList()
        self.reg_convs  = nn.ModuleList()
        self.cls_preds  = nn.ModuleList()
        self.reg_preds  = nn.ModuleList()
        self.obj_preds  = nn.ModuleList()
        self.stems      = nn.ModuleList()

        for i in range(len(in_channels)):
            self.stems.append(BaseConv(in_channels = int(in_channels[i] * width), out_channels = int(256 * width), ksize = 1, stride = 1, act = act))
            self.cls_convs.append(nn.Sequential(*[
                Conv(in_channels = int(256 * width), out_channels = int(256 * width), ksize = 3, stride = 1, act = act), 
                Conv(in_channels = int(256 * width), out_channels = int(256 * width), ksize = 3, stride = 1, act = act), 
            ]))
            self.cls_preds.append(
                nn.Conv2d(in_channels = int(256 * width), out_channels = num_classes, kernel_size = 1, stride = 1, padding = 0)
            )
            

            self.reg_convs.append(nn.Sequential(*[
                Conv(in_channels = int(256 * width), out_channels = int(256 * width), ksize = 3, stride = 1, act = act), 
                Conv(in_channels = int(256 * width), out_channels = int(256 * width), ksize = 3, stride = 1, act = act)
            ]))
            self.reg_preds.append(
                nn.Conv2d(in_channels = int(256 * width), out_channels = 4, kernel_size = 1, stride = 1, padding = 0)
            )
            self.obj_preds.append(
                nn.Conv2d(in_channels = int(256 * width), out_channels = 1, kernel_size = 1, stride = 1, padding = 0)
            )

    def forward(self, inputs):
        #---------------------------------------------------#
        #   inputs输入
        #   P3_out  80, 80, 256
        #   P4_out  40, 40, 512
        #   P5_out  20, 20, 1024
        #---------------------------------------------------#
        outputs = []
        for k, x in enumerate(inputs):
            #---------------------------------------------------#
            #   利用1x1卷积进行通道整合
            #---------------------------------------------------#
            x       = self.stems[k](x)
            #---------------------------------------------------#
            #   利用两个卷积标准化激活函数来进行特征提取
            #---------------------------------------------------#
            cls_feat    = self.cls_convs[k](x)
            #---------------------------------------------------#
            #   判断特征点所属的种类
            #   80, 80, num_classes
            #   40, 40, num_classes
            #   20, 20, num_classes
            #---------------------------------------------------#
            cls_output  = self.cls_preds[k](cls_feat)

            #---------------------------------------------------#
            #   利用两个卷积标准化激活函数来进行特征提取
            #---------------------------------------------------#
            reg_feat    = self.reg_convs[k](x)
            #---------------------------------------------------#
            #   特征点的回归系数
            #   reg_pred 80, 80, 4
            #   reg_pred 40, 40, 4
            #   reg_pred 20, 20, 4
            #---------------------------------------------------#
            reg_output  = self.reg_preds[k](reg_feat)
            #---------------------------------------------------#
            #   判断特征点是否有对应的物体
            #   obj_pred 80, 80, 1
            #   obj_pred 40, 40, 1
            #   obj_pred 20, 20, 1
            #---------------------------------------------------#
            obj_output  = self.obj_preds[k](reg_feat)

            output      = torch.cat([reg_output, obj_output, cls_output], 1)
            outputs.append(output)
        return outputs

class YOLOPAFPN(nn.Module):
    def __init__(self, depth = 1.0, width = 1.0, in_features = ("dark3", "dark4", "dark5"), in_channels = [256, 512, 1024], depthwise = False, act = "silu"):
        super().__init__()
        Conv                = DWConv if depthwise else BaseConv
        self.backbone       = CSPDarknet(depth, width, depthwise = depthwise, act = act)
        self.in_features    = in_features

        self.upsample       = nn.Upsample(scale_factor=2, mode="nearest")

        #-------------------------------------------#
        #   20, 20, 1024 -> 20, 20, 512
        #-------------------------------------------#
        self.lateral_conv0  = BaseConv(int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act)
    
        #-------------------------------------------#
        #   40, 40, 1024 -> 40, 40, 512
        #-------------------------------------------#
        self.C3_p4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise = depthwise,
            act = act,
        )  

        #-------------------------------------------#
        #   40, 40, 512 -> 40, 40, 256
        #-------------------------------------------#
        self.reduce_conv1   = BaseConv(int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act)
        #-------------------------------------------#
        #   80, 80, 512 -> 80, 80, 256
        #-------------------------------------------#
        self.C3_p3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise = depthwise,
            act = act,
        )

        #-------------------------------------------#
        #   80, 80, 256 -> 40, 40, 256
        #-------------------------------------------#
        self.bu_conv2       = Conv(int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act)
        #-------------------------------------------#
        #   40, 40, 256 -> 40, 40, 512
        #-------------------------------------------#
        self.C3_n3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise = depthwise,
            act = act,
        )

        #-------------------------------------------#
        #   40, 40, 512 -> 20, 20, 512
        #-------------------------------------------#
        self.bu_conv1       = Conv(int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act)
        #-------------------------------------------#
        #   20, 20, 1024 -> 20, 20, 1024
        #-------------------------------------------#
        self.C3_n4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[2] * width),
            round(3 * depth),
            False,
            depthwise = depthwise,
            act = act,
        )

        # self.temporalconv3 = CondConv2d(int(in_channels[0] * width), int(in_channels[0] * width), kernel_size=3)
        
        # self.b_f3=  nn.Sequential(
        #     nn.BatchNorm2d(int(in_channels[0] * width)),
        #     nn.ReLU(inplace=True))
        self.tada_3 = TAda(int(in_channels[0] * width), int(in_channels[0] * width),stride=[1,1,1])
        self.R_fwd3 = R_CLSTM_5(in_channels[0] * width)
        self.reduce2_0 = Conv(in_channels=int(in_channels[0] * width),out_channels=int(in_channels[0] * width) // 2,ksize=1,stride=1,act=act,)
        self.reduce2_1 = Conv(in_channels=int(in_channels[0] * width),out_channels=int(in_channels[0] * width) // 2,ksize=1,stride=1,act=act,)
        self.reduce2_2 = Conv(in_channels=int(in_channels[0] * width),out_channels=int(in_channels[0] * width) // 2,ksize=1,stride=1,act=act,)
        self.reduce2_3 = Conv(in_channels=int(in_channels[0] * width),out_channels=int(in_channels[0] * width) // 2,ksize=1,stride=1,act=act,)
        self.reduce2_4 = Conv(in_channels=int(in_channels[0] * width),out_channels=int(in_channels[0] * width) // 2,ksize=1,stride=1,act=act,)
        self.reduce2_5 = Conv(in_channels=int(in_channels[0] * width),out_channels=int(in_channels[0] * width) // 2,ksize=1,stride=1,act=act,)

        # self.temporalconv4 = CondConv2d(int(in_channels[1] * width), int(in_channels[1] * width), kernel_size=3)
        
        # self.b_f4=  nn.Sequential(
        #     nn.BatchNorm2d(int(in_channels[1] * width)),
        #     nn.ReLU(inplace=True))
        self.tada_4 = TAda(int(in_channels[1] * width), int(in_channels[1] * width),stride=[1,1,1])
        self.R_fwd4 = R_CLSTM_5(in_channels[1] * width)
        self.reduce1_0 = Conv(in_channels=int(in_channels[1] * width),out_channels=int(in_channels[1] * width) // 2,ksize=1,stride=1,act=act,)
        self.reduce1_1 = Conv(in_channels=int(in_channels[1] * width),out_channels=int(in_channels[1] * width) // 2,ksize=1,stride=1,act=act,)
        self.reduce1_2 = Conv(in_channels=int(in_channels[1] * width),out_channels=int(in_channels[1] * width) // 2,ksize=1,stride=1,act=act,)
        self.reduce1_3 = Conv(in_channels=int(in_channels[1] * width),out_channels=int(in_channels[1] * width) // 2,ksize=1,stride=1,act=act,)
        self.reduce1_4 = Conv(in_channels=int(in_channels[1] * width),out_channels=int(in_channels[1] * width) // 2,ksize=1,stride=1,act=act,)
        self.reduce1_5 = Conv(in_channels=int(in_channels[1] * width),out_channels=int(in_channels[1] * width) // 2,ksize=1,stride=1,act=act,)

        # self.temporalconv5 = CondConv2d(int(in_channels[2] * width),int(in_channels[2] * width), kernel_size=3)
        
        # self.b_f5=  nn.Sequential(
        #     nn.BatchNorm2d(int(in_channels[2] * width)),
        #     nn.ReLU(inplace=True))
        self.tada_5 = TAda(int(in_channels[2] * width), int(in_channels[2] * width),stride=[1,1,1])
        self.R_fwd5 = R_CLSTM_5(in_channels[2] * width)
        self.reduce0_0 = Conv(in_channels=int(in_channels[2] * width),out_channels=int(in_channels[2] * width) // 2,ksize=1,stride=1,act=act,)
        self.reduce0_1 = Conv(in_channels=int(in_channels[2] * width),out_channels=int(in_channels[2] * width) // 2,ksize=1,stride=1,act=act,)
        self.reduce0_2 = Conv(in_channels=int(in_channels[2] * width),out_channels=int(in_channels[2] * width) // 2,ksize=1,stride=1,act=act,)
        self.reduce0_3 = Conv(in_channels=int(in_channels[2] * width),out_channels=int(in_channels[2] * width) // 2,ksize=1,stride=1,act=act,)
        self.reduce0_4 = Conv(in_channels=int(in_channels[2] * width),out_channels=int(in_channels[2] * width) // 2,ksize=1,stride=1,act=act,)
        self.reduce0_5 = Conv(in_channels=int(in_channels[2] * width),out_channels=int(in_channels[2] * width) // 2,ksize=1,stride=1,act=act,)

    def forward(self, input, previous_x):
        presearch= torch.cat((previous_x, input.unsqueeze(1)),1)
        presearch = presearch.view(-1,presearch.size(-3),presearch.size(-2),presearch.size(-1))
        out_features            = self.backbone.forward(presearch)
        [feat1, feat2, feat3]   = [out_features[f] for f in self.in_features]

        # feat3_      = feat3.view(feat3.size(0)//4, int(4), feat3.size(-3),feat3.size(-2),feat3.size(-1))
        # feat3_  = self.temporalconv5(feat3_)
        # feat3_  = self.b_f5(feat3_)

        # feat2_      = feat2.view(feat2.size(0)//4, int(4), feat2.size(-3),feat2.size(-2),feat2.size(-1))
        # feat2_  = self.temporalconv4(feat2_)
        # feat2_  = self.b_f4(feat2_)

        # feat1_      = feat1.view(feat1.size(0)//4, int(4), feat1.size(-3),feat1.size(-2),feat1.size(-1))
        # feat1_  = self.temporalconv3(feat1_)
        # feat1_  = self.b_f3(feat1_)
        feat3_  = self.tada_5(feat3) 

        feat2_  = self.tada_4(feat2) 

        feat1_  = self.tada_3(feat1) 

        #-------------------------------------------#
        #   20, 20, 1024 -> 20, 20, 512
        #-------------------------------------------#
        P5          = self.lateral_conv0(feat3_)
        #-------------------------------------------#
        #  20, 20, 512 -> 40, 40, 512
        #-------------------------------------------#
        P5_upsample = self.upsample(P5)
        #-------------------------------------------#
        #  40, 40, 512 + 40, 40, 512 -> 40, 40, 1024
        #-------------------------------------------#
        P5_upsample = torch.cat([P5_upsample, feat2_], 1)
        #-------------------------------------------#
        #   40, 40, 1024 -> 40, 40, 512
        #-------------------------------------------#
        P5_upsample = self.C3_p4(P5_upsample)

        #-------------------------------------------#
        #   40, 40, 512 -> 40, 40, 256
        #-------------------------------------------#
        P4          = self.reduce_conv1(P5_upsample) 
        #-------------------------------------------#
        #   40, 40, 256 -> 80, 80, 256
        #-------------------------------------------#
        P4_upsample = self.upsample(P4) 
        #-------------------------------------------#
        #   80, 80, 256 + 80, 80, 256 -> 80, 80, 512
        #-------------------------------------------#
        P4_upsample = torch.cat([P4_upsample, feat1_], 1) 
        #-------------------------------------------#
        #   80, 80, 512 -> 80, 80, 256
        #-------------------------------------------#
        P3_out      = self.C3_p3(P4_upsample)
        b,d =  P3_out.size(0)//4, 4
        fwd_out3 = self.R_fwd3(P3_out, b, d)
        P3_out_      = fwd_out3.permute(0,2,1,3,4)
        P3_out_search, P3_out_pre = P3_out_[:,-1,:,:,:], P3_out_.permute(1,0,2,3,4)
        P3_out_pre_0 = torch.cat((self.reduce2_0(P3_out_pre[0]),self.reduce2_1(P3_out_pre[1])),dim=1)+P3_out_pre[1]
        P3_out_pre_1 = torch.cat((self.reduce2_2(P3_out_pre_0),self.reduce2_3(P3_out_pre[2])),dim=1)+P3_out_pre[2]
        P3_out_pre_2 = torch.cat((self.reduce2_4(P3_out_pre_1),self.reduce2_5(P3_out_pre[3])),dim=1)+P3_out_search
        
        #-------------------------------------------#
        #   80, 80, 256 -> 40, 40, 256
        #-------------------------------------------#
        P3_downsample   = self.bu_conv2(P3_out) 
        #-------------------------------------------#
        #   40, 40, 256 + 40, 40, 256 -> 40, 40, 512
        #-------------------------------------------#
        P3_downsample   = torch.cat([P3_downsample, P4], 1) 
        #-------------------------------------------#
        #   40, 40, 256 -> 40, 40, 512
        #-------------------------------------------#
        P4_out          = self.C3_n3(P3_downsample) 
        b,d =  P4_out.size(0)//4, 4
        fwd_out4 = self.R_fwd4(P4_out, b, d)
        P4_out_      = fwd_out4.permute(0,2,1,3,4)
        P4_out_search, P4_out_pre = P4_out_[:,-1,:,:,:], P4_out_.permute(1,0,2,3,4)
        # P4_pre = []
        # for i in range(len(P4_out_pre)):
        #     res4=self.reduce1(P4_out_pre[i])
        #     P4_pre.append(res4)
        # P4_pre = torch.cat(P4_pre, dim=1) + P4_out_search
        P4_out_pre_0 = torch.cat((self.reduce1_0(P4_out_pre[0]),self.reduce1_1(P4_out_pre[1])),dim=1)+P4_out_pre[1]
        P4_out_pre_1 = torch.cat((self.reduce1_2(P4_out_pre_0),self.reduce1_3(P4_out_pre[2])),dim=1)+P4_out_pre[2]
        P4_out_pre_2 = torch.cat((self.reduce1_4(P4_out_pre_1),self.reduce1_5(P4_out_pre[3])),dim=1)+P4_out_search
        #-------------------------------------------#
        #   40, 40, 512 -> 20, 20, 512
        #-------------------------------------------#
        P4_downsample   = self.bu_conv1(P4_out)
        #-------------------------------------------#
        #   20, 20, 512 + 20, 20, 512 -> 20, 20, 1024
        #-------------------------------------------#
        P4_downsample   = torch.cat([P4_downsample, P5], 1)
        #-------------------------------------------#
        #   20, 20, 1024 -> 20, 20, 1024
        #-------------------------------------------#
        P5_out          = self.C3_n4(P4_downsample)
        b,d =  P5_out.size(0)//4, 4
        fwd_out5 = self.R_fwd5(P5_out, b, d)
        P5_out_      = fwd_out5.permute(0,2,1,3,4)
        P5_out_search, P5_out_pre = P5_out_[:,-1,:,:,:], P5_out_.permute(1,0,2,3,4)
        P5_out_pre_0 = torch.cat((self.reduce0_0(P5_out_pre[0]),self.reduce0_1(P5_out_pre[1])),dim=1)+P5_out_pre[1]
        P5_out_pre_1 = torch.cat((self.reduce0_2(P5_out_pre_0),self.reduce0_3(P5_out_pre[2])),dim=1)+P5_out_pre[2]
        P5_out_pre_2 = torch.cat((self.reduce0_4(P5_out_pre_1),self.reduce0_5(P5_out_pre[3])),dim=1)+P5_out_search

        return (P3_out_pre_2, P4_out_pre_2, P5_out_pre_2)

class YoloBody(nn.Module):
    def __init__(self, num_classes, phi):
        super().__init__()
        depth_dict = {'nano': 0.33, 'tiny': 0.33, 's' : 0.33, 'm' : 0.67, 'l' : 1.00, 'x' : 1.33,}
        width_dict = {'nano': 0.25, 'tiny': 0.375, 's' : 0.50, 'm' : 0.75, 'l' : 1.00, 'x' : 1.25,}
        depth, width    = depth_dict[phi], width_dict[phi]
        depthwise       = True if phi == 'nano' else False 

        self.backbone   = YOLOPAFPN(depth, width, depthwise=depthwise)
        self.head       = YOLOXHead(num_classes, width, depthwise=depthwise)

    def forward(self, x, previous_x):
        fpn_outs    = self.backbone.forward(x, previous_x)
        outputs     = self.head.forward(fpn_outs)
        return outputs
    

