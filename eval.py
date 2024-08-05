import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import optim
from torch.optim.lr_scheduler import StepLR

# dataset and transformation
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os

# display images
from torchvision import utils
# import matplotlib.pyplot as plt

# utils
import numpy as np
import math

import time
import copy
from tqdm import tqdm

from utils.pytorch_utils import accuracy, AverageMeter
from models.resnets import ResNet50

torch.cuda.empty_cache()

dataset_path = '/workspace/dataset/imagenet'

input_image_size = 224
batch_size = 1

val_dir = os.path.join(dataset_path, 'val')  
input_image_crop = 0.875    # imagenet dataset DEFAULT        # ImageNet 의 기본 input size: 256x256 이다

resize_image_size = int(math.ceil(input_image_size / input_image_crop)) #math.ceil은 실수를 입력하면 올림하여 정수를 반환하는 함수   224 -> 256
transforms_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform_list = [transforms.Resize(resize_image_size), transforms.CenterCrop(input_image_size), transforms.ToTensor(), transforms_normalize]
transformer = transforms.Compose(transform_list)

val_dataset = datasets.ImageFolder(val_dir, transformer)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                            num_workers=4, pin_memory=True, sampler=None)

    
# load model
# torch.backends.cudnn.benchmark = True
model = ResNet50()
# model_state_dict = torch.load('/workspace/resnet_tutorial/saved_models/SGD_model_epoch_15.pth') # 모델의 가중치 로드
model_state_dict = torch.load('/workspace/resnet_tutorial/saved_models/model_epoch_5.pth') # 모델의 가중치 로드
print(model_state_dict)
#model.load_state_dict(model_state_dict, strict=True) # strict=True 는 모델의 구조와 가중치가 정확히 일치해야 함을 의미
model.load_state_dict(model_state_dict, strict=True)
model = model.cuda('cuda:0') #모델을 첫 번째 CUDA device(GPU)로 이동

# from torchsummary import summary
# summary(model, (3, 224, 224))


model.eval()
# print(model)
# model.requires_grad_(False)

acc1_sum = 0
acc5_sum = 0

losses = AverageMeter()
top1 = AverageMeter()
top5 = AverageMeter()
timer_start = time.time()


device = 'cuda:0'
criterion = nn.CrossEntropyLoss().to(device)

with torch.no_grad():
    with tqdm(total=len(val_loader), desc="Validate") as t:
        for i, (input, target) in enumerate(val_loader):
            input = input.to(device=device, non_blocking=True)
            target = target.to(device=device, non_blocking=True)
            output = model(input)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            loss = criterion(output, target)

            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0].item(), input.size(0))
            top5.update(acc5[0].item(), input.size(0))

            #if i % 100 == 0:
                 #print('mini_batch {}, top-1 acc={:4g}%, top-5 acc={:4g}%'.format(i, acc1[0], acc5[0]))
            t.set_postfix(
                {
                    "loss": losses.avg,
                    "top1": top1.avg,
                    "top5": top5.avg,
                    "img_size": input.size(2),
                }
            )
            t.update(1)


timer_end = time.time()
#latency = float() / (timer_end - timer_start)

#print('*** validation top-1 acc={}%, top-5 acc={}%, latency={:4g} img/s'.format(
    #top1, top5, latency))

