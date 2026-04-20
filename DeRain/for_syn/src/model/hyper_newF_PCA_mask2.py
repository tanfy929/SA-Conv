# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 15:58:33 2021

@author: XieQi
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as  F
#import MyLibForSteerCNN as ML
# import scipy.io as sio    
import math
from PIL import Image

class Fconv_PCA(nn.Module):

    def __init__(self,  sizeP, inNum, outNum, tranNum=8, inP = None, padding=None, ifIni=0, bias=True, Smooth = True, iniScale = 1.0, stride = 1):
       
        super(Fconv_PCA, self).__init__()
        if inP==None:
            inP = sizeP
        self.tranNum = tranNum
        self.outNum = outNum
        self.inNum = inNum
        self.sizeP = sizeP
        self.stride = stride
        self.GetBasis = GetBasis(sizeP, tranNum, inP)   
        # self.register_buffer("Basis", Basis)#.cuda())        
        self.ifbias = bias
        if ifIni:
            expand = 1
        else:
            expand = tranNum
        # iniw = Getini_reg(Basis.size(3), inNum, outNum, self.expand, weight)*iniScale
        self.expand = expand
        self.weights = nn.Parameter(torch.Tensor(outNum, inNum, expand, inP*inP), requires_grad=True)
        # iniw = Getini_reg(inP*inP, inNum, outNum, self.expand)*iniScale
        # self.weights = nn.Parameter(iniw, requires_grad=True)

        # self.weights = nn.Parameter(torch.Tensor(outNum, inNum, expand, inP*inP), requires_grad=True)
        # # nn.init.kaiming_uniform_(self.weights, a=0,mode='fan_in', nonlinearity='leaky_relu')
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
        B,C,H,W = input.size()
        
        tranNum = self.tranNum
        outNum = self.outNum
        inNum = self.inNum
        expand = self.expand
        Basis = self.GetBasis(Cx, Cy, theta0)
        tempW = torch.einsum('bijok,mnak->bmonaij', Basis, self.weights)
        
        Num = tranNum//expand
        tempWList = [torch.cat([tempW[:, :,i*Num:(i+1)*Num,:,-i:,:,:],tempW[:, :,i*Num:(i+1)*Num,:,:-i,:,:]], dim = 4) for i in range(expand)]   
        tempW = torch.cat(tempWList, dim = 2)
        
        _filter = tempW.reshape([B, outNum*tranNum, inNum*self.expand, self.sizeP, self.sizeP ])
        if self.ifbias:
            _bias = self.c.repeat([1,1,tranNum,1]).reshape([1,outNum*tranNum,1,1])
            # self.register_buffer("bias", _bias)
        # group conv
        input = input.reshape(1, input.size(0) * input.size(1), input.size(2), input.size(3))
        _filter = _filter.reshape(_filter.size(0)*_filter.size(1), _filter.size(2), _filter.size(3), _filter.size(4))
        output = F.conv2d(input, _filter,
                        stride = self.stride,
                        padding=self.padding,
                        dilation=1,
                        groups=B)
        output = output.reshape(B,-1,H,W)
        if self.ifbias:
            output = output+_bias
        return output   
    
    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        if self.c is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.c, -bound, bound)

    
class Fconv_PCA_out(nn.Module):
    
    def __init__(self,  sizeP, inNum, outNum, tranNum=4, inP = None, padding=None, ifIni=0, bias=True, Smooth = True,iniScale = 1.0, stride = 1):
       
        super(Fconv_PCA_out, self).__init__()
        if inP==None:
            inP = sizeP
        self.tranNum = tranNum
        self.outNum = outNum
        self.inNum = inNum
        self.sizeP = sizeP
        self.stride = stride
        self.GetBasis = GetBasis(sizeP, tranNum, inP)  
        # Basis, Rank, weight = GetBasis_PCA(sizeP,tranNum,inP, Smooth = Smooth)        
        # self.register_buffer("Basis", Basis)#.cuda())        
        self.weights = nn.Parameter(torch.Tensor(outNum, inNum, 1, inP*inP), requires_grad=True)
        # nn.init.kaiming_uniform_(self.weights, a=0,mode='fan_in', nonlinearity='leaky_relu')

        # iniw = Getini_reg(Basis.size(3), inNum, outNum, 1, weight)*iniScale
        # self.weights = nn.Parameter(iniw, requires_grad=True)
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
        B,C,H,W = input.size()

        tranNum = self.tranNum
        outNum = self.outNum
        inNum = self.inNum
        Basis = self.GetBasis(Cx, Cy, theta0)
        tempW = torch.einsum('bijok,mnak->bmonaij', Basis, self.weights)
        _filter = tempW.reshape([B, outNum, inNum*tranNum , self.sizeP, self.sizeP ])
        _bias = self.c

        input = input.reshape(1, input.size(0) * input.size(1), input.size(2), input.size(3))
        _filter = _filter.reshape(_filter.size(0)*_filter.size(1), _filter.size(2), _filter.size(3), _filter.size(4))
        output = F.conv2d(input, _filter,
                        stride = self.stride,
                        padding=self.padding,
                        dilation=1,
                        groups=B)
        output = output.reshape(B,-1,H,W)
        return output + _bias
        
    # def train(self, mode=True):
    #     if mode:
    #         # TODO thoroughly check this is not causing problems
    #         if hasattr(self, "filter"):
    #             del self.filter
    #     elif self.training:
    #         # avoid re-computation of the filter and the bias on multiple consecutive calls of `.eval()`
    #         tranNum = self.tranNum
    #         tranNum = self.tranNum
    #         outNum = self.outNum
    #         inNum = self.inNum
    #         Basis = self.GetBasis()
    #         tempW = torch.einsum('ijok,mnak->manoij', Basis, self.weights)
            
    #         _filter = tempW.reshape([outNum, inNum*tranNum , self.sizeP, self.sizeP ])
    #         self.register_buffer("filter", _filter)
    #     return super(Fconv_PCA_out, self).train(mode)      
    
    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        if self.c is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.c, -bound, bound)
    
# class FconvTranspose_PCA(nn.Module):

#     def __init__(self,  sizeP, inNum, outNum, tranNum=8, inP = None, padding=None, ifIni=0, bias=True, Smooth = True, iniScale = 1.0, stride = 1):
       
#         super(FconvTranspose_PCA, self).__init__()
#         if inP==None:
#             inP = sizeP
#         self.tranNum = tranNum
#         self.outNum = outNum
#         self.inNum = inNum
#         self.sizeP = sizeP
#         self.stride = stride
#         self.GetBasis = GetBasis(sizeP,  tranNum, inP)  
#         # Basis, Rank, weight = GetBasis_PCA(sizeP,tranNum,inP,Smooth = Smooth)        
#         # self.register_buffer("Basis", Basis)#.cuda())        
#         self.ifbias = bias
#         if ifIni:
#             expand = 1
#         else:
#             expand = tranNum
#         # iniw = Getini_reg(Basis.size(3), inNum, outNum, self.expand, weight)*iniScale
#         self.expand = expand
#         self.weights = nn.Parameter(torch.Tensor(outNum, inNum, expand, inP*inP), requires_grad=True)
#         # nn.init.kaiming_uniform_(self.weights, a=0,mode='fan_in', nonlinearity='leaky_relu')
#         if padding == None:
#             self.padding = 0
#         else:
#             self.padding = padding
#         if bias:
#             self.c = nn.Parameter(torch.Tensor(1,outNum,1,1))
#         else:
#             self.register_parameter('c', None)
#         self.reset_parameters()
#     def forward(self, input):
    
#         if self.training:
#             tranNum = self.tranNum
#             outNum = self.outNum
#             inNum = self.inNum
#             expand = self.expand
#             Basis = self.GetBasis()
#             tempW = torch.einsum('ijok,mnak->namoij', Basis, self.weights)
#             # tempW = torch.einsum('ijok,mnak->monaij', [self.Basis, self.weights])   # for torch<1.0
            
#             Num = tranNum//expand
#             tempWList = [torch.cat([tempW[:,i*Num:(i+1)*Num,:,-i:,:,:],tempW[:,i*Num:(i+1)*Num,:,:-i,:,:]], dim = 3) for i in range(expand)]   
#             tempW = torch.cat(tempWList, dim = 1)
            
#             _filter = tempW.reshape([inNum*self.expand, outNum*tranNum, self.sizeP, self.sizeP ])
#             if self.ifbias:
#                 _bias = self.c.repeat([1,1,tranNum,1]).reshape([1,outNum*tranNum,1,1])
#                 # self.register_buffer("bias", _bias)
#         else:
#             _filter = self.filter
#             if self.ifbias:
#                 _bias   = self.bias
#         output = F.conv_transpose2d(input, _filter,
#                         stride = self.stride,
#                         padding=self.padding,
#                         output_padding=1,
#                         dilation=1,
#                         groups=1)
#         if self.ifbias:
#             output = output+_bias
#         return output 
        
#     def train(self, mode=True):
#         if mode:
#             # TODO thoroughly check this is not causing problems
#             if hasattr(self, "filter"):
#                 del self.filter
#                 if self.ifbias:
#                     del self.bias
#         elif self.training:
#             # avoid re-computation of the filter and the bias on multiple consecutive calls of `.eval()`
#             tranNum = self.tranNum
#             outNum = self.outNum
#             inNum = self.inNum
#             expand = self.expand
#             Basis = self.GetBasis()
#             tempW = torch.einsum('ijok,mnak->namoij', Basis, self.weights)
#             # tempW = torch.einsum('ijok,mnak->monaij', [self.Basis, self.weights])   # for torch<1.0
#             Num = tranNum//expand
#             tempWList = [torch.cat([tempW[:,i*Num:(i+1)*Num,:,-i:,:,:],tempW[:,i*Num:(i+1)*Num,:,:-i,:,:]], dim = 3) for i in range(expand)]   
#             tempW = torch.cat(tempWList, dim = 1)
#             _filter = tempW.reshape([inNum*self.expand, outNum*tranNum, self.sizeP, self.sizeP ])
#             # _bias = self.c.repeat([1,1,tranNum,1]).reshape([1,outNum*tranNum,1,1])
#             self.register_buffer("filter", _filter)
#             if self.ifbias:
#                 _bias = self.c.repeat([1,1,tranNum,1]).reshape([1,outNum*tranNum,1,1])
#                 self.register_buffer("bias", _bias)

#         return super(FconvTranspose_PCA, self).train(mode)  
    
#     def reset_parameters(self) -> None:
#         nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
#         if self.c is not None:
#             fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
#             bound = 1 / math.sqrt(fan_in)
#             nn.init.uniform_(self.c, -bound, bound)
   

class Fconv_1X1(nn.Module):
    
    def __init__(self, inNum, outNum, tranNum=8, ifIni=0, bias=True, Smooth = True, iniScale = 1.0, stride=1):
       
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
                
        bias = self.c.repeat([1,1,tranNum,1]).reshape([1,outNum*tranNum,1,1])#.cuda()

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
        bias=True, bn=False, act = F_relu(True), res_scale=1,  Smooth = True, iniScale = 1.0):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(kernel_size, n_feats, n_feats, tranNum=tranNum, inP = inP, padding=(kernel_size-1)//2,  bias=bias, Smooth = Smooth, iniScale = iniScale))
            if bn:
                m.append(F_BN(n_feats, tranNum))
            if i == 0:
                m.append(act)

        # self.body = nn.Sequential(*m)
        self.body = nn.ModuleList(m)
        self.res_scale = res_scale

    def forward(self, x, Cx, Cy, theta0):
        # res = self.body(x, Cx, Cy, theta0).mul(self.res_scale)
        # res += x
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
    def __init__(self, sizeP, tranNum=8, inP=None):
        super(GetBasis,self).__init__()
        self.sizeP = sizeP
        self.tranNum = tranNum
        inX, inY = MaskC_ini(sizeP, tranNum)
        inX = torch.FloatTensor(inX)
        inY = torch.FloatTensor(inY)
        if inP==None:
            inP = sizeP
        self.Rank = inP * inP
        self.inp = inP//2

        self.register_buffer("inX", inX.reshape(1, sizeP,sizeP,1,1,1))  
        self.register_buffer("inY", inY.reshape(1, sizeP,sizeP,1,1,1)) 
       
        v = torch.pi/inP*(inP-1)
        U = Matrix_PCA(sizeP, tranNum, inP=None, Smooth = True)
        self.register_buffer("U", U) 

        k = torch.arange(-(inP//2),inP//2+1).reshape(1, 1, 1, 1, inP, 1)*v
        l = torch.arange(-(inP//2),inP//2+1).reshape(1, 1, 1, 1, 1, inP)*v
        self.register_buffer("k", k)
        self.register_buffer("l", l)
        
        theta = torch.arange(tranNum)/tranNum*2*torch.pi
        self.register_buffer("theta", theta.reshape(1, 1, 1, tranNum, 1, 1))

        ini_Cx = torch.ones(1)
        ini_Cy = torch.ones(1)
        ini_theta0 = torch.zeros(1)
        self.register_buffer("ini_Cx", ini_Cx)
        self.register_buffer("ini_Cy", ini_Cy)
        self.register_buffer("ini_theta0",ini_theta0)
    def forward(self, Cx_, Cy_, theta0_):
        B = Cx_.size(0)

        Cx = self.ini_Cx + 0.01 * Cx_
        Cy = self.ini_Cy + 0.01 * Cy_
        theta0 = self.ini_theta0 + 0.01 * theta0_

        Cx = Cx.view(B, 1, 1, 1, 1, 1)
        Cy = Cy.view(B, 1, 1, 1, 1, 1)
        theta0 = theta0.view(B, 1, 1, 1, 1, 1)

        X = torch.cos(theta0)*self.inX-torch.sin(theta0)*self.inY
        Y = torch.cos(theta0)*self.inY+torch.sin(theta0)*self.inX

        X1 = X * Cx
        Y1 = Y * Cy

        X = torch.cos(self.theta)*X1-torch.sin(self.theta)*Y1
        Y = torch.cos(self.theta)*Y1+torch.sin(self.theta)*X1

        BasisC = torch.cos(self.k*X+self.l*Y)
        BasisS = torch.sin(self.k*X+self.l*Y)
        # p = self.inP / 2
        # BasisC = torch.cos((self.k-self.inP*(self.k>p))*self.v*X+(self.l-self.inP*(self.l>p))*self.v*Y)
        # BasisS = torch.sin((self.k-self.inP*(self.k>p))*self.v*X+(self.l-self.inP*(self.l>p))*self.v*Y)

        Mask_ = MaskC(self.sizeP, self.tranNum, theta0, Cx, Cy)
        Mask = Mask_.detach().view(B, self.sizeP,self.sizeP,1,1,1) 

        BasisC = BasisC * Mask  
        BasisS = BasisS * Mask

        BasisC = BasisC.reshape(B, BasisC.size(1),BasisC.size(2),BasisC.size(3),-1)
        BasisS = BasisS.reshape(B, BasisS.size(1),BasisS.size(2),BasisS.size(3),-1)

        BasisR = torch.cat((BasisC,BasisS),dim = 4)
     
        # BasisR = torch.einsum('rabcd,de->rabce', BasisR, self.U)
        BasisR = torch.matmul(BasisR, self.U)
        return BasisR


def Matrix_PCA(sizeP, tranNum=8, inP=None, Smooth = True):
    if inP==None:
        inP = sizeP
    inX, inY, Mask = MaskC_ori(sizeP, tranNum)
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
    p = inP/2
    
    k = np.reshape(np.arange(-(inP//2),inP//2+1), [1,1,1,inP,1])*v
    l = np.reshape(np.arange(-(inP//2),inP//2+1), [1,1,1,1,inP])*v
    
    BasisC = np.cos(k*X+l*Y)
    BasisS = np.sin(k*X+l*Y)

    BasisC = np.reshape(BasisC,[sizeP, sizeP, tranNum, inP*inP])*np.expand_dims(Mask,3)
    BasisS = np.reshape(BasisS,[sizeP, sizeP, tranNum, inP*inP])*np.expand_dims(Mask,3)

    BasisC = np.reshape(BasisC,[sizeP*sizeP*tranNum, inP*inP])
    BasisS = np.reshape(BasisS,[sizeP*sizeP*tranNum, inP*inP])

    BasisR = np.concatenate((BasisC, BasisS), axis = 1)
    # print('------BasisR',BasisR.shape) # (100, 50)
    U,S,VT = np.linalg.svd(np.matmul(BasisR.T,BasisR))

    Rank   = np.sum(S>0.0001)
    BasisR = np.matmul(np.matmul(BasisR,U[:,:Rank]),np.diag(1/np.sqrt(S[:Rank]+0.0000000001))) 
    BasisR = np.reshape(BasisR,[sizeP, sizeP, tranNum, Rank])
    # print('*******BasisR',BasisR.shape) #  (5, 5, 4, 25)
    temp = np.reshape(BasisR, [sizeP*sizeP, tranNum, Rank])
    var = (np.std(np.sum(temp, axis = 0)**2, axis=0)+np.std(np.sum(temp**2*sizeP*sizeP, axis = 0),axis = 0))/np.mean(np.sum(temp, axis = 0)**2+np.sum(temp**2*sizeP*sizeP, axis = 0),axis = 0)
    # Trod = 1
    # Ind = var<Trod
    # Rank = np.sum(Ind)
    # Weight = 1/np.maximum(var, 0.04)/25
    # if Smooth:
    #     BasisR = np.expand_dims(np.expand_dims(np.expand_dims(Weight,0),0),0)*BasisR
    S = 1/np.sqrt(S[:Rank]+0.0000000001)
    # print('U', U.shape) #(18,18)
    # print('S', S.shape) # (9,9)
    U = U[:,:Rank]*np.expand_dims(S,0)
    return torch.FloatTensor(U)

def MaskC_ini(SizeP, tranNum):
        p = (SizeP-1)/2
        x = np.arange(-p,p+1)/p
        X,Y  = np.meshgrid(x,x)
        return X, Y

def MaskC(SizeP, tranNum, theta0, Cx, Cy):
        device = theta0.device 
        p = (SizeP-1)/2

        x = torch.arange(-p, p + 1, dtype=torch.float32, device=device) / p
        X, Y = torch.meshgrid(x, x, indexing='ij')

        X = X.reshape(1, SizeP, SizeP, 1, 1, 1)
        Y = Y.reshape(1, SizeP, SizeP, 1, 1, 1)

        X = torch.cos(theta0)*X-torch.sin(theta0)*Y
        Y = torch.cos(theta0)*Y+torch.sin(theta0)*X
        X1 = X*(Cx)
        Y1 = Y*(Cy)

        C = X1**2+Y1**2
        if tranNum ==4 or tranNum==2 or tranNum==1:
            Mask = torch.ones([SizeP, SizeP])
        else:
            if SizeP>4:
                Mask = torch.exp(-torch.maximum(C-1,torch.zeros_like(C))/0.2)
            else:
                Mask = torch.exp(-torch.maximum(C-1,torch.zeros_like(C))/2)
        return Mask

def MaskC_ori(SizeP, tranNum):
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
        return X, Y, Mask

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

        # Build the Gaussian smoothing filter
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
        #self.register_buffer("filter", _filter)
    
    def forward(self, input):
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
    def __init__(self,channels, tranNum=8, affine=False, track_running_stats: bool = False):
        super(F_IN, self).__init__()
        # print('channel', channels)
        self.IN = nn.InstanceNorm2d(num_features=channels,affine=affine,track_running_stats=track_running_stats)
        self.tranNum = tranNum
    def forward(self, X):
        # print('X', X.size(1))
        X = self.IN(X.reshape([X.size(0), int(X.size(1)/self.tranNum), self.tranNum*X.size(2), X.size(3)]))
        return X.reshape([X.size(0), self.tranNum*X.size(1),int(X.size(2)/self.tranNum), X.size(3)])

class F_Dropout(nn.Module):
    def __init__(self,zero_prob = 0.5,  tranNum=8):
        # nn.Dropout2d
        self.tranNum = tranNum
        super(F_Dropout, self).__init__()
        self.Dropout = nn.Dropout2d(zero_prob)
    def forward(self, X):
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


    def forward(self, input):

        assert input.shape[2:] == self.mask.shape[2:]

        out = input * self.mask
        return out

class GroupPooling(nn.Module):
    def __init__(self, tranNum=8):
        super(GroupPooling, self).__init__()
        self.tranNum = tranNum
        
    def forward(self, input):
        output = input.reshape([input.size(0), -1, self.tranNum, input.size(2), input.size(3)]) 
        output = torch.max(output,2).values
        return output
    
    
class GroupMeanPooling(nn.Module):
    def __init__(self, tranNum):
        super(GroupMeanPooling, self).__init__()
        self.tranNum = tranNum
        
    def forward(self, input):
        
        output = input.reshape([input.size(0), -1, self.tranNum, input.size(2), input.size(3)]) 
        output = torch.mean(output,2)
        return output
        
def Getini(sizeP, inNum, outNum, expand):
    
    inX, inY, Mask = MaskC(sizeP)
    X0 = np.expand_dims(inX,2)
    Y0 = np.expand_dims(inY,2)
    X0 = np.expand_dims(np.expand_dims(np.expand_dims(np.expand_dims(X0,0),0),4),0)
    y  = Y0[:,1]
    y = np.expand_dims(np.expand_dims(np.expand_dims(np.expand_dims(y,0),0),3),0)

    orlW = np.zeros([outNum,inNum,expand,sizeP,sizeP,1,1])
    for i in range(outNum):
        for j in range(inNum):
            for k in range(expand):
                temp = np.array(Image.fromarray(((np.random.randn(3,3))*2.4495/np.sqrt((inNum)*sizeP*sizeP))).resize((sizeP,sizeP)))
                orlW[i,j,k,:,:,0,0] = temp
             
    v = np.pi/sizeP*(sizeP-1)
    k = np.reshape((np.arange(sizeP)),[1,1,1,1,1,sizeP,1])
    l = np.reshape((np.arange(sizeP)),[1,1,1,1,1,sizeP])

    tempA =  np.sum(np.cos(k*v*X0)*orlW,4)/sizeP
    tempB = -np.sum(np.sin(k*v*X0)*orlW,4)/sizeP
    A     =  np.sum(np.cos(l*v*y)*tempA+np.sin(l*v*y)*tempB,3)/sizeP
    B     =  np.sum(np.cos(l*v*y)*tempB-np.sin(l*v*y)*tempA,3)/sizeP 
    A     = np.reshape(A, [outNum,inNum,expand,sizeP*sizeP])
    B     = np.reshape(B, [outNum,inNum,expand,sizeP*sizeP]) 
    iniW  = np.concatenate((A,B), axis = 3)
    return torch.FloatTensor(iniW)

