# Helper function for extracting features from pre-trained models
import sys, os
import argparse
import torch
import torch.nn as nn
import cv2
import numpy as np
import glob

from attacker import Attacker
from loader import ImageNet_A
from utils.Resnet import resnet152_denoise, resnet101_denoise, resnet152
from utils.Normalize import Normalize, Permute


def load_model():
    pretrained_model1 = resnet101_denoise()
    loaded_state_dict = torch.load(os.path.join('weight', 'Adv_Denoise_Resnext101.pytorch'))
    pretrained_model1.load_state_dict(loaded_state_dict, strict=True)

    pretrained_model2 = resnet152_denoise()
    loaded_state_dict = torch.load(os.path.join('weight', 'Adv_Denoise_Resnet152.pytorch'))
    pretrained_model2.load_state_dict(loaded_state_dict)

    pretrained_model3 = resnet152()
    loaded_state_dict = torch.load(os.path.join('weight', 'Adv_Resnet152.pytorch'))
    pretrained_model3.load_state_dict(loaded_state_dict)

    model1 = nn.Sequential(
            Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            Permute([2, 1, 0]),
            pretrained_model1
        )
    
    model2 = nn.Sequential(
            Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            Permute([2, 1, 0]),
            pretrained_model2
        )

    model3 = nn.Sequential(
            Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            Permute([2, 1, 0]),
            pretrained_model3
        )

    model1.cuda()
    model1.eval()

    model2.cuda()
    model2.eval()

    model3.cuda()
    model3.eval()

    for _, param in enumerate(model1.parameters()):
        param.requires_grad = False
    for _, param in enumerate(model2.parameters()):
        param.requires_grad = False
    for _, param in enumerate(model3.parameters()):
        param.requires_grad = False

    return model1, model2, model3


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default='./data/', type=str, help='path to data')
    parser.add_argument('--output_dir', default='./results/', type=str, help='path to results')
    parser.add_argument('--batch_size', default=4, type=int, help='mini-batch size')
    parser.add_argument('--steps', default=100, type=int, help='iteration steps')
    parser.add_argument('--max_norm', default=32, type=float, help='Linf limit')
    parser.add_argument('--div_prob', default=0.9, type=float, help='probability of diversity')
    args = parser.parse_args()

    model1, model2, model3 = load_model()

    output_dir = os.path.join(args.output_dir, 'images')

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    dataset = ImageNet_A(args.input_dir)
    loader = torch.utils.data.DataLoader(dataset, 
                                         batch_size=args.batch_size, 
                                         shuffle=False)

    attacker = Attacker(steps=args.steps, 
                        max_norm=args.max_norm/255.0,
                        div_prob=args.div_prob,
                        device=torch.device('cuda'))

    for ind, (img, label_true, label_target, filenames) in enumerate(loader):
        img_g, label_true_g, label_target_g = img.cuda(), label_true.cuda(), label_target.cuda()

        adv = attacker.attack(model1, model2, model3, img_g, label_true_g, label_target_g)

        for bind, filename in enumerate(filenames):
            out_img = adv[bind].detach().cpu().numpy()
            delta_img = np.abs(out_img - img[bind].numpy()) * 255.0

            print('Attack on {}:'.format(os.path.split(filename)[-1]))
            print('Max: {0:.0f}, Mean: {1:.2f}'.format(
                (np.max(delta_img)),
                (np.mean(delta_img)) 
            ))

            out_img = np.transpose(out_img, axes=[1, 2, 0]) * 255.0
            out_img = out_img[:, :, ::-1]

            out_filename = os.path.join(output_dir, os.path.split(filename)[-1])
            cv2.imwrite(out_filename, out_img)