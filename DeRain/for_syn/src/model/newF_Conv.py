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
        # self.register_buffer("Basis", Basis)#.cuda())        
        self.ifbias = bias
        if ifIni:
            expand = 1
        else:
            expand = tranNum
        # iniw = Getini_reg(Basis.size(3), inNum, outNum, self.expand, weight)*iniScale
        self.expand = expand
        self.weights = nn.Parameter(torch.Tensor(outNum, inNum, expand, inP*inP), requires_grad=True)
        # nn.init.kaiming_uniform_(self.weights, a=0,mode='fan_in', nonlinearity='leaky_relu')
        if padding == None:
            self.padding = 0
        else:
            self.padding = padding
        if bias:
            self.c = nn.Parameter(torch.Tensor(1,outNum,1,1))
        else:
            self.register_parameter('c', None)
        self.reset_parameters()
    def forward(self, input):
    
        if self.training:
            tranNum = self.tranNum
            outNum = self.outNum
            inNum = self.inNum
            expand = self.expand
            Basis = self.GetBasis()
            # print('Basis', Basis.size())
            tempW = torch.einsum('ijok,mnak->monaij', Basis, self.weights)
            # tempW = torch.einsum('ijok,mnak->monaij', [self.Basis, self.weights])   # for torch<1.0
            
            Num = tranNum//expand
            tempWList = [torch.cat([tempW[:,i*Num:(i+1)*Num,:,-i:,:,:],tempW[:,i*Num:(i+1)*Num,:,:-i,:,:]], dim = 3) for i in range(expand)]   
            tempW = torch.cat(tempWList, dim = 1)
            
            _filter = tempW.reshape([outNum*tranNum, inNum*self.expand, self.sizeP, self.sizeP ])
            if self.ifbias:
                _bias = self.c.repeat([1,1,tranNum,1]).reshape([1,outNum*tranNum,1,1])
                # self.register_buffer("bias", _bias)
        else:
            _filter = self.filter
            if self.ifbias:
                _bias   = self.bias
        output = F.conv2d(input, _filter,
                        stride = self.stride,
                        padding=self.padding,
                        dilation=1,
                        groups=1)
        if self.ifbias:
            output = output+_bias
        return output 
        
    def train(self, mode=True):
        if mode:
            # TODO thoroughly check this is not causing problems
            if hasattr(self, "filter"):
                del self.filter
                if self.ifbias:
                    del self.bias
        elif self.training:
            # avoid re-computation of the filter and the bias on multiple consecutive calls of `.eval()`
            tranNum = self.tranNum
            outNum = self.outNum
            inNum = self.inNum
            expand = self.expand
            Basis = self.GetBasis()
            tempW = torch.einsum('ijok,mnak->monaij', Basis, self.weights)
            # tempW = torch.einsum('ijok,mnak->monaij', [self.Basis, self.weights])   # for torch<1.0
            Num = tranNum//expand
            tempWList = [torch.cat([tempW[:,i*Num:(i+1)*Num,:,-i:,:,:],tempW[:,i*Num:(i+1)*Num,:,:-i,:,:]], dim = 3) for i in range(expand)]   
            tempW = torch.cat(tempWList, dim = 1)
            _filter = tempW.reshape([outNum*tranNum, inNum*self.expand, self.sizeP, self.sizeP ])
            # _bias = self.c.repeat([1,1,tranNum,1]).reshape([1,outNum*tranNum,1,1])
            self.register_buffer("filter", _filter)
            if self.ifbias:
                _bias = self.c.repeat([1,1,tranNum,1]).reshape([1,outNum*tranNum,1,1])
                self.register_buffer("bias", _bias)

        return super(Fconv_PCA, self).train(mode)  
    
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
        self.GetBasis = GetBasis(sizeP,  tranNum, inP)  
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
        
    def forward(self, input):
    
        if self.training:
            tranNum = self.tranNum
            outNum = self.outNum
            inNum = self.inNum
            Basis = self.GetBasis()
            tempW = torch.einsum('ijok,mnak->manoij', Basis, self.weights)
            _filter = tempW.reshape([outNum, inNum*tranNum , self.sizeP, self.sizeP ])
        else:
            _filter = self.filter
        _bias = self.c
        output = F.conv2d(input, _filter,
                        stride = self.stride,
                        padding=self.padding,
                        dilation=1,
                        groups=1)
        return output + _bias
        
    def train(self, mode=True):
        if mode:
            # TODO thoroughly check this is not causing problems
            if hasattr(self, "filter"):
                del self.filter
        elif self.training:
            # avoid re-computation of the filter and the bias on multiple consecutive calls of `.eval()`
            tranNum = self.tranNum
            tranNum = self.tranNum
            outNum = self.outNum
            inNum = self.inNum
            Basis = self.GetBasis()
            tempW = torch.einsum('ijok,mnak->manoij', Basis, self.weights)
            
            _filter = tempW.reshape([outNum, inNum*tranNum , self.sizeP, self.sizeP ])
            self.register_buffer("filter", _filter)
        return super(Fconv_PCA_out, self).train(mode)      
    
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
        # Basis, Rank, weight = GetBasis_PCA(sizeP,tranNum,inP,Smooth = Smooth)        
        # self.register_buffer("Basis", Basis)#.cuda())        
        self.ifbias = bias
        if ifIni:
            expand = 1
        else:
            expand = tranNum
        # iniw = Getini_reg(Basis.size(3), inNum, outNum, self.expand, weight)*iniScale
        self.expand = expand
        self.weights = nn.Parameter(torch.Tensor(outNum, inNum, expand, inP*inP), requires_grad=True)
        # nn.init.kaiming_uniform_(self.weights, a=0,mode='fan_in', nonlinearity='leaky_relu')
        if padding == None:
            self.padding = 0
        else:
            self.padding = padding
        if bias:
            self.c = nn.Parameter(torch.Tensor(1,outNum,1,1))
        else:
            self.register_parameter('c', None)
        self.reset_parameters()
    def forward(self, input):
    
        if self.training:
            tranNum = self.tranNum
            outNum = self.outNum
            inNum = self.inNum
            expand = self.expand
            Basis = self.GetBasis()
            tempW = torch.einsum('ijok,mnak->namoij', Basis, self.weights)
            # tempW = torch.einsum('ijok,mnak->monaij', [self.Basis, self.weights])   # for torch<1.0
            
            Num = tranNum//expand
            tempWList = [torch.cat([tempW[:,i*Num:(i+1)*Num,:,-i:,:,:],tempW[:,i*Num:(i+1)*Num,:,:-i,:,:]], dim = 3) for i in range(expand)]   
            tempW = torch.cat(tempWList, dim = 1)
            
            _filter = tempW.reshape([inNum*self.expand, outNum*tranNum, self.sizeP, self.sizeP ])
            if self.ifbias:
                _bias = self.c.repeat([1,1,tranNum,1]).reshape([1,outNum*tranNum,1,1])
                # self.register_buffer("bias", _bias)
        else:
            _filter = self.filter
            if self.ifbias:
                _bias   = self.bias
        output = F.conv_transpose2d(input, _filter,
                        stride = self.stride,
                        padding=self.padding,
                        output_padding=1,
                        dilation=1,
                        groups=1)
        if self.ifbias:
            output = output+_bias
        return output 
        
    def train(self, mode=True):
        if mode:
            # TODO thoroughly check this is not causing problems
            if hasattr(self, "filter"):
                del self.filter
                if self.ifbias:
                    del self.bias
        elif self.training:
            # avoid re-computation of the filter and the bias on multiple consecutive calls of `.eval()`
            tranNum = self.tranNum
            outNum = self.outNum
            inNum = self.inNum
            expand = self.expand
            Basis = self.GetBasis()
            tempW = torch.einsum('ijok,mnak->namoij', Basis, self.weights)
            # tempW = torch.einsum('ijok,mnak->monaij', [self.Basis, self.weights])   # for torch<1.0
            Num = tranNum//expand
            tempWList = [torch.cat([tempW[:,i*Num:(i+1)*Num,:,-i:,:,:],tempW[:,i*Num:(i+1)*Num,:,:-i,:,:]], dim = 3) for i in range(expand)]   
            tempW = torch.cat(tempWList, dim = 1)
            _filter = tempW.reshape([inNum*self.expand, outNum*tranNum, self.sizeP, self.sizeP ])
            # _bias = self.c.repeat([1,1,tranNum,1]).reshape([1,outNum*tranNum,1,1])
            self.register_buffer("filter", _filter)
            if self.ifbias:
                _bias = self.c.repeat([1,1,tranNum,1]).reshape([1,outNum*tranNum,1,1])
                self.register_buffer("bias", _bias)

        return super(FconvTranspose_PCA, self).train(mode)  
    
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

    def forward(self, input):
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
    
class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size, tranNum=8, inP = None, 
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1,  Smooth = True, iniScale = 1.0):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(kernel_size, n_feats, n_feats, tranNum=tranNum, inP = inP, padding=(kernel_size-1)//2,  bias=bias, Smooth = Smooth, iniScale = iniScale))
            if bn:
                m.append(F_BN(n_feats, tranNum))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res
    
def Getini_reg(nNum, inNum, outNum,expand, weight = 1): 
    A = (np.random.rand(outNum,inNum,expand,nNum)-0.5)*2*2.4495/np.sqrt((inNum)*nNum)*np.expand_dims(np.expand_dims(np.expand_dims(weight, axis = 0),axis = 0),axis = 0)
    return torch.FloatTensor(A)


class GetBasis(nn.Module):
    def __init__(self, sizeP, tranNum=4, inP=None):
        super(GetBasis,self).__init__()
        inX, inY, Mask = MaskC(sizeP, tranNum)
        if inP==None:
            inP = sizeP
        device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
        # self.inP = inP
        # self.register_buffer()会将张量与模型状态一起保存并在模型移动到GPU时，这些缓冲区会自动移动到相同的设备
        self.register_buffer("inX", inX.reshape(sizeP,sizeP,1,1,1))  
        self.register_buffer("inY", inY.reshape(sizeP,sizeP,1,1,1)) 
        self.register_buffer("Mask", Mask.reshape(sizeP,sizeP,1,1,1)) 
        # self.Cx = nn.Parameter(torch.ones(1))
        # self.Cy = nn.Parameter(torch.ones(1))
        # self.theta0 = nn.Parameter(torch.zeros(1))
        self.Cx = torch.ones(1).to(device)
        self.Cy = torch.ones(1).to(device)
        self.theta0 = torch.zeros(1).to(device)
        v = torch.pi/inP*(inP-1)
        U = Matrix_PCA(sizeP, tranNum=4, inP=None, Smooth = True)
        self.register_buffer("U", U) 
        # self.register_buffer("Weight", Weight) 
        
        self.k = torch.arange(-(inP//2),inP//2+1, device=device).reshape(1,1,1,inP,1)*v
        self.l = torch.arange(-(inP//2),inP//2+1, device=device).reshape(1,1,1,1,inP)*v
        # k = np.reshape(np.arange(inP),[1,1,1,inP,1])
        # l = np.reshape(np.arange(inP),[1,1,1,1,inP])
        # self.register_buffer("k",torch.FloatTensor(k)) 
        # self.register_buffer("l",torch.FloatTensor(l)) 
        theta = torch.arange(tranNum)/tranNum*2*torch.pi
        self.theta = theta.reshape(1,1,tranNum,1,1).to(device)

    def forward(self):
        # print('inX', self.inX.device) # cuda
        # print('Cx', self.Cx.device) # cuda
        X = torch.cos(self.theta0)*self.inX-torch.sin(self.theta0)*self.inY
        Y = torch.cos(self.theta0)*self.inY+torch.sin(self.theta0)*self.inX
        
        X1 = X*self.Cx
        Y1 = Y*self.Cy

        # print('theta0', self.theta0.device) # cuda
        # X2 = torch.cos(self.theta0)*X1+torch.sin(self.theta0)*Y1  # cuda
        # Y2 = torch.cos(self.theta0)*Y1-torch.sin(self.theta0)*X1

        X = torch.cos(self.theta)*X1-torch.sin(self.theta)*Y1
        Y = torch.cos(self.theta)*Y1+torch.sin(self.theta)*X1

        BasisC = torch.cos(self.k*X+self.l*Y)
        BasisS = torch.sin(self.k*X+self.l*Y)
        # p = self.inP / 2
        # BasisC = torch.cos((self.k-self.inP*(self.k>p))*self.v*X+(self.l-self.inP*(self.l>p))*self.v*Y)
        # BasisS = torch.sin((self.k-self.inP*(self.k>p))*self.v*X+(self.l-self.inP*(self.l>p))*self.v*Y)

        BasisC = BasisC.reshape(BasisC.size(0),BasisC.size(1),BasisC.size(2),BasisC.size(3)*BasisC.size(4) )
        BasisS = BasisS.reshape(BasisS.size(0),BasisS.size(1),BasisS.size(2),BasisS.size(3)*BasisS.size(4) )

        BasisR = torch.cat((BasisC,BasisS),dim = 3)

        # BasisR = BasisR.reshape(BasisR.size(0)*BasisR.size(1)*BasisR.size(2), BasisR.size(3))

        # BasisR = torch.matmul(BasisR,self.U) 
        BasisR = torch.einsum('abcd,de->abce', BasisR, self.U)
        # BasisR = torch.matmul(torch.matmul(BasisR,self.U),self.S) 
        # BasisR = BasisR.reshape([BasisC.size(0), BasisC.size(1), BasisC.size(2), BasisC.size(3)]) #xiaoxin BasisC.size(3) = Rank
        # print('######## Basis1 ########', self.Basis1)
        # print('######## Basis2 ########', BasisR)
        return BasisR




def Matrix_PCA(sizeP, tranNum=4, inP=None, Smooth = True):
    if inP==None:
        inP = sizeP
    inX, inY, Mask = MaskC(sizeP, tranNum)
    X0 = np.expand_dims(inX,2)
    Y0 = np.expand_dims(inY,2)
    Mask = np.expand_dims(Mask,2)
    theta = np.arange(tranNum)/tranNum*2*np.pi
    theta = np.expand_dims(np.expand_dims(theta,axis=0),axis=0)
#    theta = torch.FloatTensor(theta)
    X = np.cos(theta)*X0-np.sin(theta)*Y0
    Y = np.cos(theta)*Y0+np.sin(theta)*X0
#    X = X.unsqueeze(3).unsqueeze(4)
    X = np.expand_dims(np.expand_dims(X,3),4)
    Y = np.expand_dims(np.expand_dims(Y,3),4)
    v = np.pi/inP*(inP-1)
    p = inP/2
    
    # k = np.reshape(np.arange(inP),[1,1,1,inP,1])
    # l = np.reshape(np.arange(inP),[1,1,1,1,inP])
    
    
    # BasisC = np.cos((k-inP*(k>p))*v*X+(l-inP*(l>p))*v*Y)
    # BasisS = np.sin((k-inP*(k>p))*v*X+(l-inP*(l>p))*v*Y)
    
    k = np.reshape(np.arange(-(inP//2),inP//2+1), [1,1,1,inP,1])*v
    l = np.reshape(np.arange(-(inP//2),inP//2+1), [1,1,1,1,inP])*v
    
    BasisC = np.cos(k*X+l*Y)
    BasisS = np.sin(k*X+l*Y)


    BasisC = np.reshape(BasisC,[sizeP, sizeP, tranNum, inP*inP])#*np.expand_dims(Mask,3)
    BasisS = np.reshape(BasisS,[sizeP, sizeP, tranNum, inP*inP])#*np.expand_dims(Mask,3)

    BasisC = np.reshape(BasisC,[sizeP*sizeP*tranNum, inP*inP])
    BasisS = np.reshape(BasisS,[sizeP*sizeP*tranNum, inP*inP])

    BasisR = np.concatenate((BasisC, BasisS), axis = 1)
    
    U,S,VT = np.linalg.svd(np.matmul(BasisR.T,BasisR))

    Rank   = np.sum(S>0.0001)
    BasisR = np.matmul(np.matmul(BasisR,U[:,:Rank]),np.diag(1/np.sqrt(S[:Rank]+0.0000000001))) 
    BasisR = np.reshape(BasisR,[sizeP, sizeP, tranNum, Rank])
    
    temp = np.reshape(BasisR, [sizeP*sizeP, tranNum, Rank])
    var = (np.std(np.sum(temp, axis = 0)**2, axis=0)+np.std(np.sum(temp**2*sizeP*sizeP, axis = 0),axis = 0))/np.mean(np.sum(temp, axis = 0)**2+np.sum(temp**2*sizeP*sizeP, axis = 0),axis = 0)
    # Trod = 1
    # Ind = var<Trod
    # Rank = np.sum(Ind)
    Weight = 1/np.maximum(var, 0.04)/25
    if Smooth:
        BasisR = np.expand_dims(np.expand_dims(np.expand_dims(Weight,0),0),0)*BasisR
    S = 1/np.sqrt(S[:Rank]*Weight+0.0000000001)
    # print('U', U.shape) #(18,18)
    # print('S', S.shape) # (9,9)
    U = U[:,:Rank]*np.expand_dims(S,0)
    # print('######### BasisR1 ##########', BasisR)
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
    def forward(self, X):
        X = self.BN(X.reshape([X.size(0), int(X.size(1)/self.tranNum), self.tranNum*X.size(2), X.size(3)]))
        return X.reshape([X.size(0), self.tranNum*X.size(1),int(X.size(2)/self.tranNum), X.size(3)])

class F_IN(nn.Module):
    def __init__(self,channels, tranNum=4, affine=False, track_running_stats: bool = False):
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
    def __init__(self, tranNum=4):
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

