import numpy as np
import torch
from torch import nn
from typing import Union, List, Tuple, Type
import pdb
from torch.nn.modules.dropout import _DropoutNd

from dynamic_network_architectures.building_blocks.simple_conv_blocks import StackedConvBlocks
from dynamic_network_architectures.building_blocks.helper import get_matching_convtransp
from dynamic_network_architectures.building_blocks.residual_encoders import ResidualEncoder
from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder
from nnunetv2.nets.nmODEs import do_nothing,get_matching_upsample,single_decoder,add_op

init_scale = 2
    
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


class MyDecoder(nn.Module):
    def __init__(self,
                 encoder: Union[PlainConvEncoder, ResidualEncoder],
                 num_classes: int,
                 n_conv_per_stage: Union[int, Tuple[int, ...], List[int]],
                 deep_supervision,
                 nonlin_first: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 conv_bias: bool = None
                 ):
        """
        This class needs the skips of the encoder as input in its forward.

        the encoder goes all the way to the bottleneck, so that's where the decoder picks up. stages in the decoder
        are sorted by order of computation, so the first stage has the lowest resolution and takes the bottleneck
        features and the lowest skip as inputs
        the decoder has two (three) parts in each stage:
        1) conv transpose to upsample the feature maps of the stage below it (or the bottleneck in case of the first stage)
        2) n_conv_per_stage conv blocks to let the two inputs get to know each other and merge
        3) (optional if deep_supervision=True) a segmentation output Todo: enable upsample logits?
        :param encoder:
        :param num_classes:
        :param n_conv_per_stage:
        :param deep_supervision:
        """
        super().__init__()
        self.deep_supervision = deep_supervision
        self.encoder = encoder
        self.num_classes = num_classes
        n_stages_encoder = len(encoder.output_channels)
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * (n_stages_encoder - 1)
        assert len(n_conv_per_stage) == n_stages_encoder - 1, "n_conv_per_stage must have as many entries as we have " \
                                                          "resolution stages - 1 (n_stages in encoder - 1), " \
                                                          "here: %d" % n_stages_encoder

        # upsample_op = get_matching_convtransp(conv_op=encoder.conv_op)
        upsample_op = get_matching_upsample(conv_op=encoder.conv_op)
        conv_bias = encoder.conv_bias if conv_bias is None else conv_bias
        norm_op = encoder.norm_op if norm_op is None else norm_op
        norm_op_kwargs = encoder.norm_op_kwargs if norm_op_kwargs is None else norm_op_kwargs
        dropout_op = encoder.dropout_op if dropout_op is None else dropout_op
        dropout_op_kwargs = encoder.dropout_op_kwargs if dropout_op_kwargs is None else dropout_op_kwargs
        nonlin = encoder.nonlin if nonlin is None else nonlin
        nonlin_kwargs = encoder.nonlin_kwargs if nonlin_kwargs is None else nonlin_kwargs


        # we start with the bottleneck and work out way up
        stages = []
        upsamples = []
        seg_layers = []
        for s in range(1, n_stages_encoder+1):
            input_features_below = encoder.output_channels[-s]
            # input_features_skip = encoder.output_channels[-(s + 1)] #ori
            input_features_skip = init_scale*self.num_classes if self.num_classes >= 3 else init_scale*3 ###################改，初始化y通道数
            # stride_for_upsample = encoder.strides[-s] #ori
            scale_for_upsample =[1] * len(encoder.strides[-s])
            for j in range(len(encoder.output_channels)-s+1):  
                scale_for_upsample = [scale_for_upsample[k] * encoder.strides[j][k] for k in range(len(encoder.strides[j]))]
            upsamples.append(upsample_op(input_features_below, input_features_skip, 1, scale_for_upsample,bias=conv_bias))
            # input features to conv is 2x input_features_skip (concat input_features_skip with upsample output)
            # n_conv_per_stage[s-1]  encoder.kernel_sizes[-(s + 1)]
        for s in range(1, n_stages_encoder+1):   
            if s != n_stages_encoder:
                up_upperlayer = upsamples[s]
            else: up_upperlayer=None
            stages.append(single_decoder(n_stages_encoder,s,upsamples[s-1],up_upperlayer,do_nothing,add_op(nn.ReLU())))

            # we always build the deep supervision outputs so that we can always load parameters. If we don't do this
            # then a model trained with deep_supervision=True could not easily be loaded at inference time where
            # deep supervision is not needed. It's just a convenience thing
            seg_layers.append(encoder.conv_op(input_features_skip, num_classes, 1, 1, 0, bias=True))

        self.stages = nn.ModuleList(stages)
        self.upsamples = nn.ModuleList(upsamples)
        # pdb.set_trace()
        self.seg_layers = nn.ModuleList(seg_layers)

    def forward(self, ori_input,skips):
        """
        we expect to get the skips in the order they were computed, so the bottleneck should be the last entry
        :param skips:
        :return:
        """
        # lres_input = skips[-1] #ori
        # print('this is mod decoder')
        # print(len(skips),len(self.stages))
        b,_,h,w,z = ori_input.shape #11.13
        # b,_,h,w,z = skips[-1].shape
        c = init_scale*self.num_classes if self.num_classes >= 3 else init_scale*3 ######改，同上input_features_skip
        y = torch.zeros(b,c,h,w,z).cuda()
        seg_outputs = []
        f = []
        for s in range(0,len(self.stages)):
            # decoder = single_decoder(len(self.stages),s,self.upsamples[s-1],self.upsamples[s],do_nothing,add_op(self.stages[s-1])) #11.13
            # decoder = single_decoder(len(self.stages),s,do_nothing,do_nothing,self.upsamples[s],add_op(self.stages[s-1]))
            y,f_current = self.stages[s](f,skips,y)
            f.append(f_current)
            # x = self.upsamples[s-1](skips[-s])
            # y = torch.cat((y, x), 1)
            # y = self.stages[s-1](y)
            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s-1](y))
            elif s == (len(self.stages)-1):
                seg_outputs.append(self.seg_layers[-1](y))
        # invert seg outputs so that the largest segmentation prediction is returned first
        seg_outputs = seg_outputs[::-1]
        if not self.deep_supervision:
            r = seg_outputs[0]
        else:
            r = seg_outputs
        return r

    def compute_conv_feature_map_size(self, input_size):
        """
        IMPORTANT: input_size is the input_size of the encoder!
        :param input_size:
        :return:
        """
        # first we need to compute the skip sizes. Skip bottleneck because all output feature maps of our ops will at
        # least have the size of the skip above that (therefore -1)
        skip_sizes = []
        for s in range(len(self.encoder.strides) - 1):
            skip_sizes.append([i // j for i, j in zip(input_size, self.encoder.strides[s])])
            input_size = skip_sizes[-1]
        # print(skip_sizes)

        assert len(skip_sizes) == len(self.stages)

        # our ops are the other way around, so let's match things up
        output = np.int64(0)
        for s in range(len(self.stages)):
            # print(skip_sizes[-(s+1)], self.encoder.output_channels[-(s+2)])
            # conv blocks
            output += self.stages[s].compute_conv_feature_map_size(skip_sizes[-(s+1)])
            # trans conv
            output += np.prod([self.encoder.output_channels[-(s+2)], *skip_sizes[-(s+1)]], dtype=np.int64)
            # segmentation
            if self.deep_supervision or (s == (len(self.stages) - 1)):
                output += np.prod([self.num_classes, *skip_sizes[-(s+1)]], dtype=np.int64)
        return output