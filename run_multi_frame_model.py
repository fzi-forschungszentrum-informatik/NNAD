#!/usr/bin/env python3

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

import sys
sys.path.append('.')
import os
import time

import tensorflow as tf
import tensorflow_addons as tfa
tfa.register_all()

from data import *
from helpers.configreader import *
from helpers.helpers import *
from helpers.writers import *
from evaluation.boundingboxevaluator import *

# Argument handling
config, config_path = get_config()

# Create dataset for inference 
ds = Dataset(settings_path=config_path, mode='test')

# Clean output directory
out_dir = config['out_dir']
clean_directory(out_dir)

# Evaluators
bb_eval = BoundingBoxEvaluator(config['num_bb_classes'])

# Load the saved model
saved_model_dir = os.path.join(config['state_dir'], 'saved_model')
model = tf.saved_model.load(saved_model_dir)
infer_fn = model.signatures['infer']

if config['train_boundingboxes']:
    bbutils = BBUtils(config['eval_image_width'], config['eval_image_height'])

duration = 0.0
imgs = 0
while True:
    inp, gt, metadata = ds.get_batched_data(1)
    # Check if we reached the end
    if inp['left'].ndim != 4:
        break

    start_time = time.time()

    # Run inference
    output = infer_fn(current_img=inp['left'], prev_img=inp['prev_left'])

    # The first run comes with a lot of overhead...
    if imgs > 0:
        duration += time.time() - start_time

    bw_flow_output = output['bw_flow']
    write_flow_img(bw_flow_output, metadata, out_dir, False)

    if config['train_labels']:
        label_output = output['pixelwise_labels']

        write_label_img(label_output, metadata, out_dir)
        write_debug_label_img(label_output, inp['left'], metadata, out_dir)

    if config['train_boundingboxes']:
        bb_targets_offset = output['bb_targets_offset'].numpy()
        bb_targets_cls = output['bb_targets_cls'].numpy()
        bb_targets_objectness = output['bb_targets_objectness'].numpy()
        bb_targets_embedding = output['bb_targets_embedding'].numpy()
        bb_targets_delta = output['bb_targets_delta'].numpy()

        boxes = bbutils.bbListFromTargetsBuffer(bb_targets_objectness, bb_targets_cls, bb_targets_offset,
                                                bb_targets_delta, bb_targets_embedding, 0.5)
        bb_eval.add(gt['bb_list'].numpy(), boxes)

        write_boxes_txt(boxes, inp['left'], metadata, out_dir)
        write_boxes_json(boxes, inp['left'], metadata, out_dir)
        write_debug_boundingbox_img(boxes, inp['left'], metadata, out_dir)

    imgs += 1

print('Average time per image: %3f' % (duration / (imgs - 1)))
bb_eval.print_results()
