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

import tensorflow as tf

from model.Heads import *
from model.Resnet import *
from helpers.configreader import *
from helpers.helpers import *

# Argument handling
config, config_path = get_config()

# Define the inference model
class Infer(tf.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.backbone = ResnetBackbone('backbone')
        self.heads = Heads('heads', config)

    @tf.function(input_signature=[
            tf.TensorSpec([1, config['eval_image_height'], config['eval_image_width'], 3], tf.float32, 'current_img')])
    def infer(self, image):
        feature_map = self.backbone(image, False)
        results = self.heads(feature_map, False)
        return_vals = {}

        if self.config['train_labels']:
            labels = tf.argmax(results['pixelwise_labels'], -1)
            return_vals['pixelwise_labels'] = labels

        if self.config['train_boundingboxes']:
            bb_targets_offset = tf.reshape(results['bb_targets_offset'], [-1])
            bb_targets_cls = tf.reshape(results['bb_targets_cls'], [-1, self.config['num_bb_classes']])
            bb_targets_cls = tf.argmax(bb_targets_cls, -1)
            bb_targets_cls = tf.reshape(bb_targets_cls, [-1])
            bb_targets_objectness = tf.reshape(results['bb_targets_objectness'], [-1, 2])
            bb_targets_objectness = tf.slice(tf.nn.softmax(bb_targets_objectness, -1), [0, 1], [-1, 1])
            bb_targets_objectness = tf.reshape(bb_targets_objectness, [-1])
            bb_targets_embedding = tf.reshape(results['bb_targets_embedding'], [-1, self.config['box_embedding_len']])
            return_vals['bb_targets_offset'] = bb_targets_offset
            return_vals['bb_targets_cls'] = bb_targets_cls
            return_vals['bb_targets_objectness'] = bb_targets_objectness
            return_vals['bb_targets_embedding'] = bb_targets_embedding

        return return_vals

# Create model
infer = Infer(config)

# Load the latest checkpoint
checkpoint = tf.train.Checkpoint(backbone=infer.backbone, heads=infer.heads)
checkpoint_manager = tf.train.CheckpointManager(checkpoint, os.path.join(config['state_dir'], 'checkpoints'), 25)
checkpoint.restore(checkpoint_manager.latest_checkpoint)

# Save the trained model
signature_dict = {'infer': infer.infer}
saved_model_dir = os.path.join(config['state_dir'], 'saved_model')
clean_directory(saved_model_dir)
tf.saved_model.save(infer, saved_model_dir, signature_dict)
