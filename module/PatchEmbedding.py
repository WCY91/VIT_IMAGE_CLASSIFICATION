import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torchvision import transforms
from torchinfo import summary
import engine
import data_setup


class PatchEmbedding(nn.Module):
    
    def __init__(self,in_channels = 3,patch_size = 16,embedding_dim = 768):
        super(PatchEmbedding,self).__init__()
        self.patcher = nn.Conv2d(in_channels=in_channels,out_channels=embedding_dim,kernel_size = patch_size,stride = patch_size,padding= 0)
        self.flatten  = nn.Flatten(start_dim = 2,end_dim = 3)
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim

    def forward(self,x):
        image_resolution = x.shape[-1]
        assert image_resolution % self.patch_size == 0, f"Input image size must be divisible by patch size, image shape: {image_resolution}, patch size: {patch_size}"
        
        x_patched = self.patcher(x)
        x_flattened = self.flatten(x_patched)

        return x_flattened.permute(0, 2, 1) # adjust so the embedding is on the final dimension [batch_size, P^2•C, N] -> [batch_size, N, P^2•C]