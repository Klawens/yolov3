import math
import os
import time

import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torch.autograd import Variable
from torchvision.ops import nms


class DecodeBox(nn.Module):
    def __init__(self, anchors, num_classes, img_size):
        super(DecodeBox, self).__init__()
        self.anchors = anchors  # config.py: anchors
        self.num_anchors = len(anchors) # 3
        self.num_classes = num_classes  # config.py: classes
        self.bbox_attrs = 5 + num_classes   #  4 adjust params, 1 has_object(confidence), num_classes
        self.img_size = img_size

    # take 13x13 for example
    def forward(self, input):
        # input: (bs, 3*(4+1+num_classes), 13(or 26 or 52), 13(or 26 or 52))
        #         bs, 3: num_anchors, 4: adjust params, 1: has_object
        batch_size = input.size(0)
        input_height = input.size(2)
        input_width = input.size(3)

        # one pixel on feature map represents 32 pixels on original image
        # 416 / 13 = 32. Receptive field
        stride_h = self.img_size[1] / input_height
        stride_w = self.img_size[0] / input_width

        # scale the anchors to feater map size
        # ex: smallest anchor is (32/116x32/90) = 0.2759x0.3556 big on a 13x13 feature map
        scaled_anchors = [(anchor_width / stride_w, anchor_height / stride_h) for anchor_width, anchor_height in self.anchors]
        # bs, 3*(4+1+num_classes), 13, 13 ----> bs, 3, 13, 13, (4+1+num_classes)
        prediction = input.view(batch_size, self.num_anchors,
                                self.bbox_attrs, input_height, input_width).permute(0, 1, 3, 4, 2).contiguous()
        
        # Adjust centre, distance from gt centre
        x = t.sigmoid(prediction[..., 0])   # self.bbox_attrs[0]
        y = t.sigmoid(prediction[..., 1])   # self.bbox_attrs[1]
        # Adjust height&width
        w = prediction[..., 2]  # self.bbox_attrs[2]
        h = prediction[..., 3]  # self.bbox_attrs[3]
        # confidence of has_object
        conf = t.sigmoid(prediction[..., 4])
        # confidence of classes, from self.bbox_attrs[5] to self.bbox_attrs[85] if it's coco dataset
        pred_cls = t.sigmoid(prediction[..., 5:])

        FloatTensor = t.cuda.FloatTensor if x.is_cuda else t.FloatTensor
        LongTensor = t.cuda.LongTensor if x.is_cuda else t.LongTensor

        # get anchor centre, Top-left of a cell, (batch_size, 3, 13, 13)
        # linspace(start, end, step), to even a seq linearly
        grid_x = t.linspace(0, input_width - 1, input_width).repeat(input_height, 1).repeat(
            batch_size * self.num_anchors, 1, 1).view(x.shape).type(FloatTensor)
        grid_y = t.linspace(0, input_height - 1, input_height).repeat(input_width, 1).t().repeat(
            batch_size * self.num_anchors, 1, 1).view(y.shape).type(FloatTensor)
        # get anchors' w&h
        anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor[0])
        anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor[1])
        anchor_w = anchor_w.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(w.shape)
        anchor_h = anchor_h.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(h.shape)
        # calculate adjusted centre&w&h
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + grid_x
        pred_boxes[..., 1] = y.data + grid_y
        pred_boxes[..., 2] = t.exp(w.data) * anchor_w
        pred_boxes[..., 3] = t.exp(h.data) * anchor_h

        _scale = t.Tensor([stride_w, stride_h] * 2).type(FloatTensor)
        output = t.cat((pred_boxes.view(batch_size, -1, 4) * _scale,
                            conf.view(batch_size, -1, 1), pred_cls.view(batch_size, -1, self.num_classes)), -1)
        return output.data
