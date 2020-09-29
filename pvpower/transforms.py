from torch import Tensor, normal
import torch
import numpy as np
from skimage.util import apply_parallel
from tqdm.auto import trange
import random

class PerImageNormalization:

    def __init__(self):
        pass

    def __call__(self, x: Tensor) -> Tensor:
        mean = x.mean(dim=(1,2), keepdim=True)
        std = x.std(dim=(1,2), keepdim=True)
        return (x-mean) / std
    
#class Identity:
#    
#    def __init__(self):
#        pass
#    
#    def __call__(self, x):
#        return x

    
class PerImageZCAWhitening:
    
    def __init__(self, nb_samples=100, patch_size=8):
        self._patch_size = patch_size
        self._nb_samples = nb_samples
    
    def _get_patch(self, x: Tensor, x0, y0):
        size = x.size()
        x1 = x0+self._patch_size
        y1 = y0+self._patch_size
        if size[1] > x1 and size[2] > y1:
            return x[:,x0:x1,y0:y1], x1, y1
        else:
            x1 = min(size[1], x1)
            y1 = min(size[2], y1)
            patch = torch.zeros((size[0],self._patch_size,self._patch_size))
            patch[:,:x1-x0,:y1-y0] = x[:,x0:x1:,y0:y1]
            return patch, x1, y1

    def _compute_whiten_transform(self, x: Tensor):
        size = x.size()
        
        # accumulate patches
        patches = list()
        for i in range(self._nb_samples):
            x0 = random.randint(0, size[1]-self._patch_size)
            y0 = random.randint(0, size[2]-self._patch_size)
            patch, _, _ = self._get_patch(x, x0, y0)
            patches.append(patch.flatten())
        X = torch.stack(patches, dim=1)   # samples are columns
        
        # compute W
        X = X+1e-5
        sigma = X.matmul(X.transpose(0,1)) / self._nb_samples  # covariance
        U,S,V = sigma.svd()
        D = torch.diag(1.0/torch.sqrt(S + 1e-5))
        DU = D.matmul(U.transpose(0,1))
        W = U.matmul(DU)
        
        return W
    
    def _apply_transform(self, x: Tensor, W: Tensor):
        size = x.size()
        N = size[1] // self._patch_size + (1 if size[1] % self._patch_size != 0 else 0)
        M = size[2] // self._patch_size + (1 if size[2] % self._patch_size != 0 else 0)
        
        # apply patchwise
        for i in range(N):
            for j in range(M):
                x0 = i*self._patch_size
                y0 = j*self._patch_size
                patch, x1, y1 = self._get_patch(x, x0, y0)
                patch = W.matmul(patch.flatten()).reshape(patch.size())
                x[:,x0:x1,y0:y1] = patch[:,:x1-x0,:y1-y0]
        
        return x

    def __call__(self, x:Tensor) -> Tensor:
        # de-mean first
        mean = x.mean(2).mean(1)
        x -= mean.reshape(-1,1,1)
        
        W = self._compute_whiten_transform(x)
        x = self._apply_transform(x, W)
        
        return x
    
class GaussianNoise:
    
    def __init__(self, std):
        self._std = std
        
    def __call__(self, x:Tensor) -> Tensor:
        return x + normal(0.0, self._std, size=x.size())
        
