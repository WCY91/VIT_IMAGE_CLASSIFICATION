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
from module import PatchEmbedding
import random

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

setup_seed(3407)
device = "cuda" if torch.cuda.is_available() else "cpu"

train_dir = "./data/train"
test_dir = "./data/test"

# because the vit is need the image width and height is 224 so need to preprocess
IMG_SIZE = 224
BATCH_SIZE = 32

manual_transforms = transforms.Compose([transforms.Resize((IMG_SIZE,IMG_SIZE)),transforms.ToTensor()])

train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    transform=manual_transforms, # use manually created transforms
    batch_size=BATCH_SIZE
)

height = 224
width = 224
channels = 3
patch_size = 16 #可以想成一個小圖片的長 寬 因為其會透過多張小圖表達大圖片

number_of_patches = int((height * width) /( patch_size ** 2))
assert IMG_SIZE % patch_size == 0, "Image size must be divisible by patch size"

embedding_layer_input_shape = (height,width,channels) # input shape
embedding_layer_output_shape = (number_of_patches,patch_size**2*channels)
print("original image size input shape is ",embedding_layer_input_shape)
print("the paper need shape is ", embedding_layer_output_shape)

patchify = PatchEmbedding.PatchEmbedding(in_channels=3,patch_size=16,embedding_dim=768)




