'''
 # @ Author: Takuho Ri (Zehao Li)
 # @ Modified time: 2025-01-15 15:33:59
 # @ Description: 
 '''

import numpy as np
import pandas as pd
import torch as th
from torch import nn
from kan import KAN
from fastkan import FastKAN


class KAN_Layer(nn.Module):
    def __init__(self, width:list, mode, grid=20, k=3):
        super().__init__()
        self.width = width
        self.mode = mode
        layers = [KAN(width=self.width, grid=grid, k=k, auto_save=False)]
        if self.mode == "c":
            layers.append(nn.Sigmoid())
        self.kan = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.kan:
            x = layer(x)
        return x

class FastKAN_Layer(nn.Module):
    def __init__(self, width:list, mode, grid_min, grid_max, num_grids):
        super().__init__()
        self.width = width
        self.mode = mode
        layers = [FastKAN(width, grid_min=grid_min, grid_max=grid_max, num_grids=num_grids)]
        if self.mode == "c":
            layers.append(nn.Sigmoid())
        self.kan = nn.ModuleList(layers)
    
    def forward(self, x):
        for layer in self.kan:
            x = layer(x)
        return x

# develop based on https://github.com/GistNoesis/FourierKAN.git
class _NaiveFourierKANLayer(th.nn.Module):
    def __init__( self, inputdim, outdim, gridsize, addbias=True, smooth_initialization=False):
        super(_NaiveFourierKANLayer,self).__init__()
        self.gridsize= gridsize
        self.addbias = addbias
        self.inputdim = inputdim
        self.outdim = outdim
        
        # With smooth_initialization, fourier coefficients are attenuated by the square of their frequency.
        # This makes KAN's scalar functions smooth at initialization.
        # Without smooth_initialization, high gridsizes will lead to high-frequency scalar functions,
        # with high derivatives and low correlation between similar inputs.
        self.grid_norm_factor = (th.arange(gridsize) + 1)**2 if smooth_initialization else np.sqrt(gridsize)
        
        #The normalization has been chosen so that if given inputs where each coordinate is of unit variance,
        #then each coordinates of the output is of unit variance 
        #independently of the various sizes
        self.fouriercoeffs = th.nn.Parameter( th.randn(2,outdim,inputdim,gridsize) / 
                                                (np.sqrt(inputdim) * self.grid_norm_factor ) )
        if( self.addbias ):
            self.bias  = th.nn.Parameter( th.zeros(1,outdim))

    #x.shape ( ... , indim ) 
    #out.shape ( ..., outdim)
    def forward(self,x):
        xshp = x.shape
        outshape = xshp[0:-1]+(self.outdim,)
        x = th.reshape(x,(-1,self.inputdim))
        #Starting at 1 because constant terms are in the bias
        k = th.reshape( th.arange(1,self.gridsize+1,device=x.device),(1,1,1,self.gridsize))
        xrshp = th.reshape(x,(x.shape[0],1,x.shape[1],1) ) 
        #This should be fused to avoid materializing memory
        c = th.cos( k*xrshp )
        s = th.sin( k*xrshp )
        #We compute the interpolation of the various functions defined by their fourier coefficient for each input coordinates and we sum them 
        y =  th.sum( c*self.fouriercoeffs[0:1],(-2,-1)) 
        y += th.sum( s*self.fouriercoeffs[1:2],(-2,-1))
        if( self.addbias):
            y += self.bias
        #End fuse
        '''
        #You can use einsum instead to reduce memory usage
        #It stills not as good as fully fused but it should help
        #einsum is usually slower though
        c = th.reshape(c,(1,x.shape[0],x.shape[1],self.gridsize))
        s = th.reshape(s,(1,x.shape[0],x.shape[1],self.gridsize))
        y2 = th.einsum( "dbik,djik->bj", th.concat([c,s],axis=0) ,self.fouriercoeffs )
        if( self.addbias):
            y2 += self.bias
        diff = th.sum((y2-y)**2)
        print("diff")
        print(diff) #should be ~0
        '''
        y = th.reshape( y, outshape)
        return y

class FourierKAN_Layer(nn.Module):
    def __init__(self, width:list, num_grids, dropout=0, smooth_initialization=False):
        super().__init__()
        self.width = width
        self.num_grids = num_grids
        num_layers = len(self.width) - 1
        layers = [_NaiveFourierKANLayer(inputdim, outdim, self.num_grids, addbias=True, smooth_initialization=smooth_initialization)
                                                for inputdim, outdim in zip(width[:-1], width[1:])]
        for i in range(num_layers-1):
            layers.insert(i*2+1, nn.Dropout(dropout))
        self.kan = nn.ModuleList(layers)
    
    def reset_parameters(self):
        for layer in self.kan:
            if isinstance(layer, _NaiveFourierKANLayer):
                layer.fouriercoeffs = th.nn.Parameter( th.randn(2,layer.outdim,layer.inputdim,layer.gridsize) / (np.sqrt(layer.inputdim) * layer.grid_norm_factor ) )
                if( layer.addbias ):
                    layer.bias  = th.nn.Parameter( th.zeros(1,layer.outdim))

    def forward(self, x):
        for layer in self.kan:
            x = layer(x)
        return x


class MLP_Layer(nn.Module):
    def __init__(self, width:list, dropout=0):
        super().__init__()
        self.width = width
        num_layers = len(self.width) - 1
        layers = [nn.Linear(self.width[i], self.width[i+1]) for i in range(num_layers)]
        for i in range(num_layers-1):
            layers.insert(i*2+1, nn.SiLU())
        for i in range(num_layers-1):
            layers.insert(i*3+2, nn.Dropout(dropout))
        self.mlp = nn.ModuleList(layers)
    
    def reset_parameters(self):
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()
    
    def forward(self, x):
        for layer in self.mlp:
            x = layer(x)
        return x