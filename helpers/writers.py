##########################################################################
# NNAD (Neural Networks for Automated Driving) training scripts          #
# Copyright (C) 2019 FZI Research Center for Information Technology      #
#                                                                        #
# This program is free software: you can redistribute it and/or modify   #
# it under the terms of the GNU General Public License as published by   #
# the Free Software Foundation, either version 3 of the License, or      #
# (at your option) any later version.                                    #
#                                                                        #
# This program is distributed in the hope that it will be useful,        #
# but WITHOUT ANY WARRANTY; without even the implied warranty of         #
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the          #
# GNU General Public License for more details.                           #
#                                                                        #
# You should have received a copy of the GNU General Public License      #
# along with this program.  If not, see <https://www.gnu.org/licenses/>. #
##########################################################################

import numpy as np
import scipy as sp
import skimage.draw
import PIL
import os
import json

from helpers.helpers import *

LABEL_COLOR_LUT = np.array([(128, 64,128), (244, 35,232), ( 70, 70, 70), (102,102,156), (190,153,153), (153,153,153),
                            (250,170, 30), (220,220, 0), (107,142, 35), (152,251,152), ( 70,130,180), (220, 20, 60),
                            (255, 0, 0), ( 0, 0,142), ( 0, 0, 70), ( 0, 60,100), ( 0, 80,100), ( 0, 0,230),
                            (119, 11, 32)])

BOX_COLOR_LUT = np.array([(220, 20, 60), (255, 0, 0), ( 0, 0,142), ( 0, 0, 70), ( 0, 60,100), ( 0, 80,100), ( 0, 0,230),
                          (119, 11, 32), (250,170, 30), (220,220, 0)])

cityscapes_label_LUT = np.array([7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33],
                                dtype=np.uint8)

instance_dict = {
    0: 'Pedestrian',
    1: 'Cyclist',
    2: 'Car',
}

ALPHA = 0.4

def _image_metadata(metadata):
    key = metadata['key'].numpy().astype(str)[0]
    width = metadata['original_width'].numpy()
    height = metadata['original_height'].numpy()
    return key, width, height

def _labels(labels):
    return labels.numpy()[0, :, :]

def _image(image):
    image = image.numpy()[0, :, :, :]
    image = image * 255.0
    image = np.flip(image, -1) # BGR to RGB
    return image

def write_flow_img(flow, metadata, path, forward):
    key, width, height = _image_metadata(metadata)
    flow = _labels(flow)

    if forward:
        directory = 'fw_flow'
    else:
        directory = 'bw_flow'

    image_path = os.path.join(path, directory, key + '.png')
    ensure_path(image_path)
    flow_x = flow[:, :, 0]
    flow_y = flow[:, :, 1]
    flow_angle = (np.arctan2(flow_x, flow_y) + np.pi) * 128.0 / np.pi
    flow_val = np.sqrt(np.square(flow_x) + np.square(flow_y))
    flow_val = flow_val * 255.0 / np.max(flow_val)
    flow_img = np.stack([flow_angle, np.ones_like(flow_val) * 255.0, flow_val], axis=2)
    flow_img = flow_img.astype(np.uint8)
    image = PIL.Image.fromarray(flow_img, 'HSV')
    image = image.convert('RGB')
    image = image.resize((width, height), PIL.Image.NEAREST)
    image.save(image_path)

def write_label_img(labels, metadata, path):
    key, width, height = _image_metadata(metadata)
    labels = _labels(labels)

    image_path = os.path.join(path, 'label', key + '.png')
    ensure_path(image_path)
    image = PIL.Image.fromarray(cityscapes_label_LUT[labels], 'L')
    image = image.resize((width, height), PIL.Image.NEAREST)
    image.save(image_path)

def write_debug_label_img(labels, image, metadata, path):
    key, width, height = _image_metadata(metadata)
    labels = _labels(labels)
    image = _image(image)

    image_path = os.path.join(path, 'label_debug', key + '.png')
    ensure_path(image_path)
    data = (1.0 - ALPHA) * LABEL_COLOR_LUT[labels] + ALPHA * image
    image = PIL.Image.fromarray(data.astype(np.uint8), 'RGB')
    image = image.resize((width, height), PIL.Image.BILINEAR)
    image.save(image_path)

def write_boxes_txt(boxes, image, metadata, path):
    key, width, height = _image_metadata(metadata)
    image = _image(image)

    txt_path = os.path.join(path, 'boxes_txt', key + '.txt')
    ensure_path(txt_path)
    inference_height = np.shape(image)[0]
    inference_width = np.shape(image)[1]

    # Use the KITTI format to write out bounding boxes
    with open(txt_path, 'w') as fh:
        for i in range(len(boxes)):
            box = boxes[i].box
            score = boxes[i].score
            if not box.cls in instance_dict:
                continue
            cls = instance_dict[box.cls]
            x1 = float(box.x1) * width / inference_width
            y1 = float(box.y1) * height / inference_height
            x2 = float(box.x2) * width / inference_width
            y2 = float(box.y2) * height / inference_height
            fh.write('%s 0.0 0.0 0.0 %d %d %d %d 0.0 0.0 0.0 0.0 0.0 0.0 0.0 %f\n' % (cls, x1, y1, x2, y2, score))

def write_boxes_json(boxes, image, metadata, path):
    key, width, height = _image_metadata(metadata)
    image = _image(image)

    json_path = os.path.join(path, 'boxes_json', key + '.json')
    ensure_path(json_path)
    inference_height = np.shape(image)[0]
    inference_width = np.shape(image)[1]
    width_factor = float(width / inference_width)
    height_factor = float(height / inference_height)

    data = {}
    data['boxes'] = []
    for i in range(len(boxes)):
        boxdata = {}
        boxdata['score'] = boxes[i].score
        boxdata['embedding'] = boxes[i].embedding
        box = boxes[i].box
        boxdata['cls'] = box.cls
        box = boxes[i].box
        boxdata['x1'] = float(box.x1) * width_factor
        boxdata['y1'] = float(box.y1) * height_factor
        boxdata['x2'] = float(box.x2) * width_factor
        boxdata['y2'] = float(box.y2) * height_factor
        boxdata['dxc'] = float(box.dxc) * width_factor
        boxdata['dyc'] = float(box.dyc) * height_factor
        boxdata['dw'] = float(box.dw) * width_factor
        boxdata['dh'] = float(box.dh) * height_factor
        data['boxes'].append(boxdata)

    with open(json_path, 'w') as fh:
        json.dump(data, fh)

def write_debug_boundingbox_img(boxes, image, metadata, path):
    key, width, height = _image_metadata(metadata)
    image = _image(image)

    image_path = os.path.join(path, 'boxes_debug', key + '.png')
    ensure_path(image_path)
    inference_height = np.shape(image)[0]
    inference_width = np.shape(image)[1]

    for box in boxes:
        cls = box.box.cls
        x1 = np.clip(box.box.x1, 0, inference_width - 1)
        y1 = np.clip(box.box.y1, 0, inference_height - 1)
        x2 = np.clip(box.box.x2, 0, inference_width - 1)
        y2 = np.clip(box.box.y2, 0, inference_height - 1)
        for coords in [[int(y1), int(x1), int(y1), int(x2)],
                       [int(y1), int(x2), int(y2), int(x2)],
                       [int(y2), int(x2), int(y2), int(x1)],
                       [int(y2), int(x1), int(y1), int(x1)]]:
            rr, cc = skimage.draw.line(*coords)
            image[rr, cc, :] = BOX_COLOR_LUT[cls]
            if coords[0] == coords[2]:
                if coords[0] > 0:
                    image[rr - 1, cc, :] = BOX_COLOR_LUT[cls]
                if coords[0] < inference_height - 1:
                    image[rr + 1, cc, :] = BOX_COLOR_LUT[cls]
            if coords[1] == coords[3]:
                if coords[1] > 0:
                    image[rr, cc - 1, :] = BOX_COLOR_LUT[cls]
                if coords[1] < inference_width - 1:
                    image[rr, cc + 1, :] = BOX_COLOR_LUT[cls]
    image = PIL.Image.fromarray(image.astype(np.uint8), 'RGB')
    image = image.resize((width, height), PIL.Image.BILINEAR)
    image.save(image_path)
