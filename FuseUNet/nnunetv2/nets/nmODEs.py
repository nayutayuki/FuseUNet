import torch
import torch.nn as nn
from torch.nn.modules.conv import _ConvNd,_ConvTransposeNd
from typing import Type
from torch.nn import functional as F

up_conv = 1

def do_nothing(data):
    return data

def convert_conv_op_to_dim(conv_op: Type[_ConvNd]) -> int:
    """
    :param conv_op: conv class
    :return: dimension: 1, 2 or 3
    """
    if conv_op == nn.Conv1d:
        return 1
    elif conv_op == nn.Conv2d:
        return 2
    elif conv_op == nn.Conv3d:
        return 3
    else:
        raise ValueError("Unknown dimension. Only 1d 2d and 3d conv are supported. got %s" % str(conv_op))

def get_matching_upsample(conv_op: Type[_ConvNd] = None, dimension: int = None) -> Type[_ConvTransposeNd]:
    """
    You MUST set EITHER conv_op OR dimension. Do not set both!

    :param conv_op:
    :param dimension:
    :return:
    """
    assert not ((conv_op is not None) and (dimension is not None)), \
        "You MUST set EITHER conv_op OR dimension. Do not set both!"
    if conv_op is not None:
        dimension = convert_conv_op_to_dim(conv_op)
    assert dimension in [1, 2, 3], 'Dimension must be 1, 2 or 3'
    if dimension == 1:
        return upsample1d
    elif dimension == 2:
        return upsample2d
    elif dimension == 3:
        return upsample3d

class upsample1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,scale_factor,bias=False):
        super(upsample1d, self).__init__()
        
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2,bias=bias)
        self.scale_factor = scale_factor

    def forward(self, x):
        if up_conv:
            x = self.conv(x)
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='linear', align_corners=False)
        return x
    
class upsample2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,scale_factor,bias=False):
        super(upsample2d, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2,bias=bias)
        self.scale_factor = scale_factor

    def forward(self, x):
        if up_conv:
            x = self.conv(x)
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        return x
    
class upsample3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,scale_factor,bias=False):
        super(upsample3d, self).__init__()
        
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2,bias=bias)
        self.scale_factor = scale_factor

    def forward(self, x):
        if up_conv:
            x = self.conv(x)
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='trilinear', align_corners=False)
        return x
    
class add_op(torch.nn.Module):
    def __init__(self,op):
        super(add_op,self).__init__()
        self.op = op
    def forward(self,x,y):
        # pdb.set_trace()
        y = x + y
        y = self.op(y)
        return y

class single_decoder(torch.nn.Module):
    def __init__(self,num_stage,current_stage,opx,opx_upper_layer,opy,opxy):
        super(single_decoder,self).__init__()
        self.num_stage = num_stage
        self.current_stage = current_stage
        self.opx = opx
        self.opx_upper_layer = opx_upper_layer
        self.opy = opy
        self.opxy = opxy
        self.step = 1/self.num_stage

    def ODE_eq(self,x,y):
        return -self.opy(y)+self.opxy(self.opx(x),self.opy(y))

    def ODE_eq_pre(self,x,y):
        return -self.opy(y)+self.opxy(self.opx_upper_layer(x),self.opy(y))

    def step1_order1_explicit(self,x1,y1):
        f1 = self.ODE_eq(x1,y1)
        y2 = y1 + self.step*f1
        return y2,f1

    def step1_order2_implicit(self,x1,y1,x2):
        y2_pre,f1 = self.step1_order1_explicit(x1,y1)
        f2_pre = self.ODE_eq_pre(x2,y2_pre)
        y2 = y1 + (self.step/2)*(f1 + f2_pre)
        return y2,f1

    def steps2_order2_explicit(self,f1,x2,y2):
        f2 = self.ODE_eq(x2,y2)
        y3 = y2 + (self.step/2)*(3*f2 - f1)
        return y3,f2

    def steps2_order3_implicit(self,f1,x2,y2,x3):
        y3_pre,f2 = self.steps2_order2_explicit(f1,x2,y2)
        f3_pre = self.ODE_eq_pre(x3,y3_pre)
        y3 = y2 + (self.step/12)*(5*f3_pre + 8*f2 - f1)
        return y3,f2

    def steps3_order3_explicit(self,f1,f2,x3,y3):
        f3 = self.ODE_eq(x3,y3)
        y4 = y3 + (self.step/12)*(23*f3-16*f2+5*f1)
        return y4,f3

    def steps3_order4_implicit(self,f1,f2,x3,y3,x4):
        y4_pre,f3 = self.steps3_order3_explicit(f1,f2,x3,y3)
        f4_pre = self.ODE_eq_pre(x4,y4_pre)
        y4 = y3 + (self.step/24)*(9*f4_pre + 19*f3 - 5*f2 +f1)
        return y4,f3

    def steps4_order4_explicit(self,f1,f2,f3,x4,y4):
        f4 = self.ODE_eq(x4,y4)
        y5 = y4 + (self.step/24)*(55*f4 - 59*f3 + 37*f2 - 9*f1)
        return y5,f4

    def steps4_order4_implicit(self,f1,f2,f3,x4,y4,x5):
        y5_pre,f4 = self.steps4_order4_explicit(f1,f2,f3,x4,y4)
        f5_pre = self.ODE_eq_pre(x5,y5_pre)
        y5 = y4 + (self.step/24)*(9*f5_pre + 19*f4 - 5*f3 +f2)
        return y5,f4
    
    def forward(self,f,x,y):
        if self.current_stage == self.num_stage:
            if self.current_stage == 2:
                return self.steps2_order2_explicit(f[0],x[-self.current_stage],y)
            elif self.current_stage == 3:
                return self.steps3_order3_explicit(f[0],f[1],x[-self.current_stage],y)
            else:
                return self.steps4_order4_explicit(f[0],f[1],f[2],x[-self.current_stage],y)
        else:
            if self.current_stage == 1:
                return self.step1_order2_implicit(x[-self.current_stage],y,x[-self.current_stage-1])
            elif self.current_stage == 2:
                return self.steps2_order3_implicit(f[0],x[-self.current_stage],y,x[-self.current_stage-1])
            elif self.current_stage == 3:
                return self.steps3_order4_implicit(f[0],f[1],x[-self.current_stage],y,x[-self.current_stage-1])
            else:
                return self.steps4_order4_implicit(f[-self.current_stage+4],f[-self.current_stage+3],f[-self.current_stage+2],
                                                   x[-self.current_stage],y,x[-self.current_stage-1])

    # def forward(self,f,x,y):  #1
    #     return self.step1_order1_explicit(x[-self.current_stage],y)
    
    # def forward(self,f,x,y):  #2
    #     if self.current_stage == self.num_stage:
    #         return self.steps2_order2_explicit(f[-1],x[-self.current_stage],y)
    #     else:
    #         return self.step1_order2_implicit(x[-self.current_stage],y,x[-self.current_stage-1])

    # def forward(self,f,x,y): #3
    #     if self.current_stage == self.num_stage:
    #         if self.current_stage == 2:
    #             return self.steps2_order2_explicit(f[-1],x[-self.current_stage],y)
    #         else:
    #             return self.steps3_order3_explicit(f[-2],f[-1],x[-self.current_stage],y)
            
    #     else:
    #         if self.current_stage == 1:
    #             return self.step1_order2_implicit(x[-self.current_stage],y,x[-self.current_stage-1])
    #         else:
    #             return self.steps2_order3_implicit(f[-1],x[-self.current_stage],y,x[-self.current_stage-1])

