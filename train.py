import argparse    # argparse 1

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

from PIL import Image

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


def build_train_transform(input_image_size = 224):
    train_transforms = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    return transforms.Compose(train_transforms)


def main():
    # argparse 2
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--dataset_path', default='/workspace/dataset/imagenet', type=str)
    parser.add_argument('--input_image_size', default=224, type=int)

    parser.add_argument('--lr_step_size', default=30, type=int)
    parser.add_argument('--lr_gamma', default=0.1, type=float)

    batch_size = 128

    num_epochs = 30
    # argparse 3
    args = parser.parse_args()

    train_dir = os.path.join(args.dataset_path,'train')
    save_dir = '/workspace/resnet_tutorial/saved_models'


    # input_image_crop = 0.875

    # resize_image_size = int(math.ceil(args.input_image_size/input_image_crop))
    # transforms_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])     # argparse 4: args. 이름에 붙여주기
    # train_transform = transforms.Compose([
    #     transforms.Resize(resize_image_size), transforms.CenterCrop(args.input_image_size), transforms.ToTensor(), transforms_normalize])
    train_transform = build_train_transform()

    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    # class_names = train_dataset.classes
    # print(f"Classes: {class_names}")

    # for images, labels in train_loader:
    #     print(f"Batch of images has shape: {images.shape}")
    #     print(f"Batch of labels has shape: {labels.shape}")
    #     print(labels)
    #     break
    
    
    model = ResNet50()

    model = model.cuda('cuda:0')

    # #debug
    # from torchsummary import summary
    # summary(model, (3, 224, 224))


    device = 'cuda:0'
    criterion = nn.CrossEntropyLoss().to(device)
    #optimizer = optim.Adam(model.parameters(), lr=0.2, betas=(0.9, 0.999), weight_decay=0.01)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

    # optimizer가 가지고 있는 LR값을 바꿔줌
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)



    for epoch in range(num_epochs):
        model.train()
        # running_loss=0.0
        # epoch_loss=0.0
        print(f'start epoch: {epoch}/{num_epochs - 1}')

        # debug
        # for name, params in model.named_parameters():
        #     print(f'{name}: {params.requires_grad}')
        # exit(0)

        # 출력 확인용 1
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        
        # tqdm 1
        with tqdm(total=len(train_loader), desc="Train") as t:

            for batch_idx, (inputs, labels) in enumerate(train_loader):   # input보다는 image로 표기 많이 함
                    

                inputs, labels = inputs.to(device), labels.to(device)     # target으로도 많이 씀    
                labels = labels.to(device)

                # tensor = inputs[0]
                # # 텐서 형태를 (높이, 너비, 채널)로 변환
                # tensor = tensor.cpu()
                # tensor = tensor.permute(1, 2, 0)
                # # 텐서를 NumPy 배열로 변환
                # np_array = tensor.numpy()
                # # 값을 [0, 255] 범위로 스케일링
                # np_array = (np_array * 255).astype(np.uint8)
                # # NumPy 배열을 PIL 이미지로 변환
                # image = Image.fromarray(np_array)
                # image.save('image.png',path='/workspace/resnet_tutorial/temp')
                # print(labels)


                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # 출력 확인용 2
                acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
                losses.update(loss.item(), inputs.size(0))
                top1.update(acc1[0], inputs.size(0))
                top5.update(acc5[0], inputs.size(0))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # running_loss += loss.item()
                # epoch_loss += loss.item()

                t.set_postfix(
                    {
                        "loss": losses.avg,    # AverageMeter 메서드
                        "top1": top1.avg,
                        "top5": top5.avg,
                        "img_size": inputs.size(2),
                    }
                )

                # if batch_idx % 1000 == 999:
                #     print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx + 1}/{len(train_loader)}], Loss: {running_loss / 1000:.4f}')
            
                t.update(1) # tqdm 2
        
            # print(f'epoch_{epoch+1}_loss = {epoch_loss/len(train_loader)}')
        lr_scheduler.step()
        torch.save(model.state_dict(), os.path.join(save_dir,f'SGD_model_epoch_{epoch+1}.pth'))

    print('Finished Training')

if __name__=="__main__":
    main()