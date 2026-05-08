# -*- coding: utf-8 -*-
"""
Adapted for SuperNetwork (SA) Architecture
Strictly aligned with the TL baseline's mathematical logic.
Includes bmanoij dimension fix and dynamic batch processing.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from PIL import Image

class Fconv_PCA(nn.Module):

    def __init__(self,  sizeP, inNum, outNum, tranNum=4, inP = None, padding=None, ifIni=0, bias=True, Smooth = True, iniScale = 1.0, stride = 1):
       
        super(Fconv_PCA, self).__init__()
        if inP==None:
            inP = sizeP
        self.tranNum = tranNum
        self.outNum = outNum
        self.inNum = inNum
        self.sizeP = sizeP
        self.stride = stride
        self.GetBasis = GetBasis(sizeP, tranNum, inP)   
        
        self.ifbias = bias
        if ifIni:
            expand = 1
        else:
            expand = tranNum
        self.expand = expand

        iniw = Getini_reg(inP*inP, inNum, outNum, self.expand)*iniScale
        self.weights = nn.Parameter(iniw, requires_grad=True)

        if padding == None:
            self.padding = 0
        else:
            self.padding = padding
        
        self.c = nn.Parameter(torch.zeros(1,outNum,1,1), requires_grad=bias)

    def forward(self, input, Cx, Cy, theta0):
        B, C, H, W = input.size()
        
        tranNum = self.tranNum
        outNum = self.outNum
        inNum = self.inNum
        expand = self.expand
        
        Basis = self.GetBasis(Cx, Cy, theta0)
        tempW = torch.einsum('bijok,mnak->bmonaij', Basis, self.weights)
        
        Num = tranNum // expand
        tempWList = [torch.cat([tempW[:, :, i*Num:(i+1)*Num, :, -i:, :, :], tempW[:, :, i*Num:(i+1)*Num, :, :-i, :, :]], dim=4) for i in range(expand)]   
        tempW = torch.cat(tempWList, dim=2)
        
        _filter = tempW.reshape([B, outNum*tranNum, inNum*self.expand, self.sizeP, self.sizeP])
        if self.ifbias:
            _bias = self.c.repeat([1,1,tranNum,1]).reshape([1,outNum*tranNum,1,1])
            
        input = input.reshape(1, B * C, H, W)
        _filter = _filter.reshape(B * outNum * tranNum, inNum * self.expand, self.sizeP, self.sizeP)
        
        output = F.conv2d(input, _filter,
                        stride=self.stride,
                        padding=self.padding,
                        dilation=1,
                        groups=B)
        output = output.reshape(B, -1, output.size(2), output.size(3))
        
        if self.ifbias:
            output = output + _bias
        return output 


class Fconv_PCA_out(nn.Module):
    
    def __init__(self,  sizeP, inNum, outNum, tranNum=4, inP = None, padding=None, ifIni=0, bias=True, Smooth = True, iniScale = 1.0, stride = 1):
       
        super(Fconv_PCA_out, self).__init__()
        if inP==None:
            inP = sizeP
        self.tranNum = tranNum
        self.outNum = outNum
        self.inNum = inNum
        self.sizeP = sizeP
        self.stride = stride
        self.GetBasis = GetBasis(sizeP,  tranNum, inP)  

        self.weights = nn.Parameter(torch.Tensor(outNum, inNum, 1, inP*inP), requires_grad=True)

        if padding == None:
            self.padding = 0
        else:
            self.padding = padding
        self.ifbias = bias
        
        if bias:
            self.c = nn.Parameter(torch.Tensor(1,outNum,1,1))
        else:
            self.register_parameter('c', None)
        self.reset_parameters()
        
    def forward(self, input, Cx, Cy, theta0):
        B, C, H, W = input.size()

        tranNum = self.tranNum
        outNum = self.outNum
        inNum = self.inNum
        Basis = self.GetBasis(Cx, Cy, theta0)
        
        # [核心防爆机制] 必须是 bmanoij，防止通道展平错位
        tempW = torch.einsum('bijok,mnak->bmanoij', Basis, self.weights)
        _filter = tempW.reshape([B, outNum, inNum*tranNum, self.sizeP, self.sizeP])
        _bias = self.c

        input = input.reshape(1, B * C, H, W)
        _filter = _filter.reshape(B * outNum, inNum * tranNum, self.sizeP, self.sizeP)
        
        output = F.conv2d(input, _filter,
                        stride=self.stride,
                        padding=self.padding,
                        dilation=1,
                        groups=B)
        output = output.reshape(B, -1, output.size(2), output.size(3))
        return output + _bias
    
    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        if self.c is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.c, -bound, bound)
            

class FconvTranspose_PCA(nn.Module):

    def __init__(self,  sizeP, inNum, outNum, tranNum=4, inP = None, padding=None, ifIni=0, bias=True, Smooth = True, iniScale = 1.0, stride = 1):
       
        super(FconvTranspose_PCA, self).__init__()
        if inP==None:
            inP = sizeP
        self.tranNum = tranNum
        self.outNum = outNum
        self.inNum = inNum
        self.sizeP = sizeP
        self.stride = stride
        self.GetBasis = GetBasis(sizeP,  tranNum, inP)  
        
        self.ifbias = bias
        if ifIni:
            expand = 1
        else:
            expand = tranNum
        self.expand = expand
        
        self.weights = nn.Parameter(torch.Tensor(outNum, inNum, expand, inP*inP), requires_grad=True)
        
        if padding == None:
            self.padding = 0
        else:
            self.padding = padding
        if bias:
            self.c = nn.Parameter(torch.Tensor(1,outNum,1,1))
        else:
            self.register_parameter('c', None)
        self.reset_parameters()
        
    def forward(self, input, Cx, Cy, theta0):
        B, C, H, W = input.size()
        
        tranNum = self.tranNum
        outNum = self.outNum
        inNum = self.inNum
        expand = self.expand
        Basis = self.GetBasis(Cx, Cy, theta0)
        
        # [防爆机制] Transpose 需要 bnamoij
        tempW = torch.einsum('bijok,mnak->bnamoij', Basis, self.weights)
        
        Num = tranNum // expand
        tempWList = [torch.cat([tempW[:, :, i*Num:(i+1)*Num, :, -i:, :, :], tempW[:, :, i*Num:(i+1)*Num, :, :-i, :, :]], dim=4) for i in range(expand)]   
        tempW = torch.cat(tempWList, dim=2)
        
        _filter = tempW.reshape([B, inNum*self.expand, outNum*tranNum, self.sizeP, self.sizeP])
        if self.ifbias:
            _bias = self.c.repeat([1,1,tranNum,1]).reshape([1,outNum*tranNum,1,1])

        input = input.reshape(1, B * C, H, W)
        _filter = _filter.reshape(B * inNum * self.expand, outNum * tranNum, self.sizeP, self.sizeP)
        
        output = F.conv_transpose2d(input, _filter,
                        stride=self.stride,
                        padding=self.padding,
                        output_padding=1,
                        dilation=1,
                        groups=B)
        output = output.reshape(B, outNum * tranNum, output.size(2), output.size(3))
        
        if self.ifbias:
            output = output + _bias
        return output 
    
    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        if self.c is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.c, -bound, bound)
   

class Fconv_1X1(nn.Module):
    
    def __init__(self, inNum, outNum, tranNum=4, ifIni=0, bias=True, Smooth = True, iniScale = 1.0, stride=1):
       
        super(Fconv_1X1, self).__init__()

        self.tranNum = tranNum
        self.outNum = outNum
        self.inNum = inNum
        self.stride = stride
                
        if ifIni:
            self.expand = 1
        else:
            self.expand = tranNum
        iniw = Getini_reg(1, inNum, outNum, self.expand)*iniScale
        self.weights = nn.Parameter(iniw, requires_grad=True)

        self.padding = 0
        self.bias = bias

        if bias:
            self.c = nn.Parameter(torch.zeros(1,outNum,1,1), requires_grad=True)
        else:
            self.c = torch.zeros(1,outNum,1,1)

    def forward(self, input, Cx=None, Cy=None, theta0=None):
        tranNum = self.tranNum
        outNum = self.outNum
        inNum = self.inNum
        expand = self.expand
        tempW = self.weights.unsqueeze(4).unsqueeze(1).repeat([1,tranNum,1,1,1,1])
        
        Num = tranNum//expand
        tempWList = [torch.cat([tempW[:,i*Num:(i+1)*Num,:,-i:,...],tempW[:,i*Num:(i+1)*Num,:,:-i,...]], dim = 3) for i in range(expand)]   
        tempW = torch.cat(tempWList, dim = 1)

        _filter = tempW.reshape([outNum*tranNum, inNum*self.expand, 1, 1 ])
        bias = self.c.repeat([1,1,tranNum,1]).reshape([1,outNum*tranNum,1,1])

        output = F.conv2d(input, _filter,
                        stride = self.stride,
                        padding=self.padding,
                        dilation=1,
                        groups=1)
        return output+bias  


class F_relu(nn.Module):
    def __init__(self, inplace=True):
        super(F_relu, self).__init__()
        self.relu = nn.ReLU(inplace=inplace)
    def forward(self, X, Cx=None, Cy=None, theta0=None):
        X = self.relu(X)
        return X 


class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size, tranNum=8, inP = None, 
        bias=True, bn=False, act=F_relu(True), res_scale=1,  Smooth = True, iniScale = 1.0):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(kernel_size, n_feats, n_feats, tranNum=tranNum, inP = inP, padding=(kernel_size-1)//2,  bias=bias, Smooth = Smooth, iniScale = iniScale))
            if bn:
                m.append(F_BN(n_feats, tranNum))
            if i == 0:
                m.append(act)

        self.body = nn.ModuleList(m)
        self.res_scale = res_scale

    def forward(self, x, Cx, Cy, theta0):
        res = x
        for layer in self.body:
            res = layer(res, Cx, Cy, theta0)
        res = res.mul(self.res_scale)
        res += x
        return res
    

def Getini_reg(nNum, inNum, outNum,expand, weight = 1): 
    A = (np.random.rand(outNum,inNum,expand,nNum)-0.5)*2*2.4495/np.sqrt((inNum)*nNum)*np.expand_dims(np.expand_dims(np.expand_dims(weight, axis = 0),axis = 0),axis = 0)
    return torch.FloatTensor(A)


class GetBasis(nn.Module):
    def __init__(self, sizeP, tranNum=4, inP=None):
        super(GetBasis,self).__init__()
        self.sizeP = sizeP
        self.tranNum = tranNum
        
        inX, inY, Mask = MaskC(sizeP, tranNum)
        inX = torch.FloatTensor(inX)
        inY = torch.FloatTensor(inY)
        Mask = torch.FloatTensor(Mask) 
        
        if inP==None:
            inP = sizeP
            
        self.Rank = inP * inP
        self.inp = inP//2

        self.register_buffer("inX", inX.reshape(1, sizeP,sizeP,1,1,1))  
        self.register_buffer("inY", inY.reshape(1, sizeP,sizeP,1,1,1)) 
        self.register_buffer("Mask", Mask.reshape(1, sizeP,sizeP,1,1,1)) 
        
        self.Cx = nn.Parameter(torch.ones(1))
        self.Cy = nn.Parameter(torch.ones(1))
        self.theta0 = nn.Parameter(torch.zeros(1))
        
        v = torch.pi/inP*(inP-1)
        U = Matrix_PCA(sizeP, tranNum, inP=None, Smooth = True)
        self.register_buffer("U", U) 
        
        k = torch.arange(-(inP//2),inP//2+1, dtype=torch.float32).reshape(1, 1, 1, 1, inP, 1)*v
        l = torch.arange(-(inP//2),inP//2+1, dtype=torch.float32).reshape(1, 1, 1, 1, 1, inP)*v
        self.register_buffer("k", k)
        self.register_buffer("l", l)
        
        theta = torch.arange(tranNum, dtype=torch.float32)/tranNum*2*math.pi
        self.register_buffer("theta", theta.reshape(1, 1, 1, tranNum, 1, 1))

        self.scale = 1.0
        
    def forward(self, Cx, Cy, theta0):
        B = Cx.size(0)

        Cx = self.scale * Cx.view(B, 1, 1, 1, 1, 1) + self.Cx
        Cy = self.scale * Cy.view(B, 1, 1, 1, 1, 1) + self.Cy
        theta0 = self.scale * theta0.view(B, 1, 1, 1, 1, 1) + self.theta0

        X = torch.cos(theta0)*self.inX-torch.sin(theta0)*self.inY
        Y = torch.cos(theta0)*self.inY+torch.sin(theta0)*self.inX

        X1 = X * Cx
        Y1 = Y * Cy

        X = torch.cos(self.theta)*X1-torch.sin(self.theta)*Y1
        Y = torch.cos(self.theta)*Y1+torch.sin(self.theta)*X1

        BasisC = torch.cos(self.k*X+self.l*Y)
        BasisS = torch.sin(self.k*X+self.l*Y)

        
        BasisC = BasisC.reshape(B, BasisC.size(1), BasisC.size(2), BasisC.size(3), -1)
        BasisS = BasisS.reshape(B, BasisS.size(1), BasisS.size(2), BasisS.size(3), -1)

        BasisR = torch.cat((BasisC,BasisS), dim=4)

        
        BasisR = torch.einsum('bxytd,dr->bxytr', BasisR, self.U)
        return BasisR


def Matrix_PCA(sizeP, tranNum=4, inP=None, Smooth = True):
    if inP==None:
        inP = sizeP
    inX, inY, Mask = MaskC(sizeP, tranNum)
    
    # 转为 numpy 进行处理，对齐 TL 逻辑
    inX = inX.numpy()
    inY = inY.numpy()
    Mask = Mask.numpy()
    
    X0 = np.expand_dims(inX,2)
    Y0 = np.expand_dims(inY,2)
    Mask = np.expand_dims(Mask,2)
    theta = np.arange(tranNum)/tranNum*2*np.pi
    theta = np.expand_dims(np.expand_dims(theta,axis=0),axis=0)

    X = np.cos(theta)*X0-np.sin(theta)*Y0
    Y = np.cos(theta)*Y0+np.sin(theta)*X0

    X = np.expand_dims(np.expand_dims(X,3),4)
    Y = np.expand_dims(np.expand_dims(Y,3),4)
    v = np.pi/inP*(inP-1)
    
    k = np.reshape(np.arange(-(inP//2),inP//2+1), [1,1,1,inP,1])*v
    l = np.reshape(np.arange(-(inP//2),inP//2+1), [1,1,1,1,inP])*v
    
    BasisC = np.cos(k*X+l*Y)
    BasisS = np.sin(k*X+l*Y)

    # 严格遵循 TL 版本，不乘 Mask
    BasisC = np.reshape(BasisC,[sizeP, sizeP, tranNum, inP*inP])
    BasisS = np.reshape(BasisS,[sizeP, sizeP, tranNum, inP*inP])

    BasisC = np.reshape(BasisC,[sizeP*sizeP*tranNum, inP*inP])
    BasisS = np.reshape(BasisS,[sizeP*sizeP*tranNum, inP*inP])

    BasisR = np.concatenate((BasisC, BasisS), axis = 1)
    
    U,S,VT = np.linalg.svd(np.matmul(BasisR.T,BasisR))

    Rank   = np.sum(S>0.0001)
    BasisR = np.matmul(np.matmul(BasisR,U[:,:Rank]),np.diag(1/np.sqrt(S[:Rank]+0.0000000001))) 
    BasisR = np.reshape(BasisR,[sizeP, sizeP, tranNum, Rank])
    
    temp = np.reshape(BasisR, [sizeP*sizeP, tranNum, Rank])
    var = (np.std(np.sum(temp, axis = 0)**2, axis=0)+np.std(np.sum(temp**2*sizeP*sizeP, axis = 0),axis = 0))/np.mean(np.sum(temp, axis = 0)**2+np.sum(temp**2*sizeP*sizeP, axis = 0),axis = 0)

    Weight = 1/np.maximum(var, 0.04)/25
    if Smooth:
        BasisR = np.expand_dims(np.expand_dims(np.expand_dims(Weight,0),0),0)*BasisR
    S = 1/np.sqrt(S[:Rank]*Weight+0.0000000001)

    U = U[:,:Rank]*np.expand_dims(S,0)
    return torch.FloatTensor(U)


def MaskC(SizeP, tranNum):
        p = (SizeP-1)/2
        x = np.arange(-p,p+1)/p
        X,Y  = np.meshgrid(x,x)
        C    =X**2+Y**2
        if tranNum ==4 or tranNum==2 or tranNum==1:
            Mask = np.ones([SizeP, SizeP])
        else:
            if SizeP>4:
                Mask = np.exp(-np.maximum(C-1,0)/0.2)
            else:
                Mask = np.exp(-np.maximum(C-1,0)/2)
        return torch.FloatTensor(X), torch.FloatTensor(Y), torch.FloatTensor(Mask)


class PointwiseAvgPoolAntialiased(nn.Module):
    
    def __init__(self, sizeF, stride, padding=None ):
        super(PointwiseAvgPoolAntialiased, self).__init__()
        sigma = (sizeF-1)/2/3
        self.kernel_size = (sizeF, sizeF)
        if isinstance(stride, int):
            self.stride = (stride, stride)
        elif stride is None:
            self.stride = self.kernel_size
        else:
            self.stride = stride
        
        if padding is None:
            padding = int((sizeF-1)//2)
            
        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding

        grid_x = torch.arange(sizeF).repeat(sizeF).view(sizeF, sizeF)
        grid_y = grid_x.t()
        grid = torch.stack([grid_x, grid_y], dim=-1)
        mean = (sizeF - 1) / 2.
        variance = sigma ** 2.
        r = -torch.sum((grid - mean) ** 2., dim=-1, dtype=torch.get_default_dtype())
        _filter = torch.exp(r / (2 * variance))
        _filter /= torch.sum(_filter)
        _filter = _filter.view(1, 1, sizeF, sizeF)
        self.filter = nn.Parameter(_filter, requires_grad=False)
    
    def forward(self, input, Cx=None, Cy=None, theta0=None):
        _filter = self.filter.repeat((input.shape[1], 1, 1, 1))
        output = F.conv2d(input, _filter, stride=self.stride, padding=self.padding, groups=input.shape[1])        
        return output
        
class F_BN(nn.Module):
    def __init__(self,channels, tranNum=4, affine=True, track_running_stats: bool = True):
        super(F_BN, self).__init__()
        self.BN = nn.BatchNorm2d(num_features=channels,affine=affine,track_running_stats=track_running_stats)
        self.tranNum = tranNum
    def forward(self, X, Cx=None, Cy=None, theta0=None):
        X = self.BN(X.reshape([X.size(0), int(X.size(1)/self.tranNum), self.tranNum*X.size(2), X.size(3)]))
        return X.reshape([X.size(0), self.tranNum*X.size(1),int(X.size(2)/self.tranNum), X.size(3)])

class F_IN(nn.Module):
    def __init__(self,channels, tranNum=4, affine=False, track_running_stats: bool = False):
        super(F_IN, self).__init__()
        self.IN = nn.InstanceNorm2d(num_features=channels,affine=affine,track_running_stats=track_running_stats)
        self.tranNum = tranNum
    def forward(self, X, Cx=None, Cy=None, theta0=None):
        X = self.IN(X.reshape([X.size(0), int(X.size(1)/self.tranNum), self.tranNum*X.size(2), X.size(3)]))
        return X.reshape([X.size(0), self.tranNum*X.size(1),int(X.size(2)/self.tranNum), X.size(3)])

class F_Dropout(nn.Module):
    def __init__(self,zero_prob = 0.5,  tranNum=8):
        self.tranNum = tranNum
        super(F_Dropout, self).__init__()
        self.Dropout = nn.Dropout2d(zero_prob)
    def forward(self, X, Cx=None, Cy=None, theta0=None):
        X = self.Dropout(X.reshape([X.size(0), int(X.size(1)/self.tranNum), self.tranNum*X.size(2), X.size(3)]))
        return X.reshape([X.size(0), self.tranNum*X.size(1),int(X.size(2)/self.tranNum), X.size(3)])

def build_mask(s, margin=2, dtype=torch.float32):
    mask = torch.zeros(1, 1, s, s, dtype=dtype)
    c = (s-1) / 2
    t = (c - margin/100.*c)**2
    sig = 2.
    for x in range(s):
        for y in range(s):
            r = (x - c) ** 2 + (y - c) ** 2
            if r > t:
                mask[..., x, y] = math.exp((t - r)/sig**2)
            else:
                mask[..., x, y] = 1.
    return mask

class MaskModule(nn.Module):
    def __init__(self, S: int, margin: float = 0.):
        super(MaskModule, self).__init__()
        self.margin = margin
        self.mask = torch.nn.Parameter(build_mask(S, margin=margin), requires_grad=False)

    def forward(self, input, Cx=None, Cy=None, theta0=None):
        assert input.shape[2:] == self.mask.shape[2:]
        out = input * self.mask
        return out

class GroupPooling(nn.Module):
    def __init__(self, tranNum=4):
        super(GroupPooling, self).__init__()
        self.tranNum = tranNum
        
    def forward(self, input, Cx=None, Cy=None, theta0=None):
        output = input.reshape([input.size(0), -1, self.tranNum, input.size(2), input.size(3)]) 
        output = torch.max(output,2).values
        return output
    
class GroupMeanPooling(nn.Module):
    def __init__(self, tranNum):
        super(GroupMeanPooling, self).__init__()
        self.tranNum = tranNum
        
    def forward(self, input, Cx=None, Cy=None, theta0=None):
        output = input.reshape([input.size(0), -1, self.tranNum, input.size(2), input.size(3)]) 
        output = torch.mean(output,2)
        return output