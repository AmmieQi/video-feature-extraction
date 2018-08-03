__author__ = 'qianyu'

import os
import numpy as np
import torch
from torch.autograd import Variable
from config import parse_opts
from utils import video_loader
from transforms import *
from create_model import *


def feature_extract(opt):
    for video_name in os.listdir(opt.frame_root):
        frame_path = os.path.join(opt.frame_root, video_name)
        feats = []

        frame_indices = range(1, len(os.listdir(frame_path)) + 1)
        print(frame_path)
        frames = video_loader(frame_path, frame_indices)

        if opt.feature_type == '2D':
            # preprocess
            transforms = Compose([Scale(opt.sample_size),
                                  CenterCrop(opt.sample_size),
                                  ToTensor(),
                                  Normalize([114.7748 / 255, 107.7354 / 255, 99.4750 / 255],
                                            [1, 1, 1])])
            frames = [transforms(img) for img in frames]
            frames = torch.stack(frames, 0).permute(1, 0, 2, 3)
            # load model
            m2d = {}
            m2d['model'] = 'vgg16'
            model = generate_2D_model(m2d)
            print('loading model {}'.format(opt.model))
            # model_data = torch.load(opt.model)
            # model.load_state_dict(model_data['state_dict'])
            model.eval()

            video_features = []
            for i, inputs in enumerate(frames):
                with torch.no_grad():
                    inputs = Variable(inputs)
                outputs = model(inputs)

                video_features.append(outputs.cpu().data)

            video_features = torch.cat(video_features)

        elif opt.feature_type == '3D':
            # preprocess
            transforms = Compose([Scale(opt.sample_size),
                                  CenterCrop(opt.sample_size),
                                  ToTensor(),
                                  Normalize(opt.mean, [1, 1, 1])])
            frames = [transforms(img) for img in frames]
            frames = torch.stack(frames, 0).permute(1, 0, 2, 3)
            # load model
            model = generate_3D_model(opt)
            print('loading model {}'.format(opt.model))
            model_data = torch.load(opt.model)
            model.load_state_dict(model_data['state_dict'])
            model.eval()

            video_features = []
            for i, inputs in enumerate(frames):
                with torch.no_grad():
                    inputs = Variable(inputs)
                outputs = model(inputs)

                video_features.append(outputs.cpu().data)

            video_features = torch.cat(video_features)

        elif opt.feature_type == 'OF':
            # preprocess
            transforms = Compose([Scale(opt.sample_size),
                                  CenterCrop(opt.sample_size),
                                  ToTensor(),
                                  Normalize(opt.mean, [1, 1, 1])])
            frames = [transforms(img) for img in frames]
            frames = torch.stack(frames, 0).permute(1, 0, 2, 3)

            video_feature = get_OF_feature(frames)

        feats.append(video_feature)
    return feats


def get_OF_feature(frames, opt):
    video_features = []
    return video_features


if __name__ == "__main__":
    # test
    opt = parse_opts()
    opt.frame_root = "/home/qianyu/task_video/data/ucf/ucf101-frames/"
    opt.sample_size = 16
    feats = feature_extract(opt)
    print(len(feats))