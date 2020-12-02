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
from model.Flow import *
from model.EfficientNet import *
from model.BiFPN import *
from helpers.configreader import *
from helpers.helpers import *

# Argument handling
config, config_path = get_config()

# Define the inference model
class Infer(tf.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.backbone = EfficientNet('backbone', BACKBONE_ARGS)
        self.fpn1 = BiFPN('bifpn1', BIFPN_NUM_FEATURES, int(BIFPN_NUM_BLOCKS / 2), True)
        self.fpn2 = BiFPN('bifpn2', BIFPN_NUM_FEATURES, BIFPN_NUM_BLOCKS - int(BIFPN_NUM_BLOCKS / 2), False)
        self.flow = Flow('flow')
        self.flow_warp = FlowWarp('flow_warp')
        self.heads = Heads('heads', config, box_delta_regression=True)

    @tf.function(input_signature=[
            tf.TensorSpec([1, config['eval_image_height'], config['eval_image_width'], 3], tf.float32, 'img')])
    def inferBackbone(self, image):
        features = self.backbone(image, False)
        features = self.fpn1(features, False)
        return features

    @tf.function(input_signature=[
            [tf.TensorSpec([1,
                           int(config['eval_image_height'] / 2**(i+2)),
                           int(config['eval_image_width'] / 2**(i+2)),
                           BIFPN_NUM_FEATURES],
             tf.float32, 'current_features_{}'.format(i)) for i in range(6)],
            [tf.TensorSpec([1,
                           int(config['eval_image_height'] / 2**(i+2)),
                           int(config['eval_image_width'] / 2**(i+2)),
                           BIFPN_NUM_FEATURES],
             tf.float32, 'prev_features_{}'.format(i)) for i in range(6)]])
    def inferHeads(self, current_feature_map, prev_feature_map):
        return_vals = {}

        bw_flow = self.flow([prev_feature_map, current_feature_map], False)
        feature_map = self.flow_warp([current_feature_map,
                                      prev_feature_map,
                                      bw_flow], False)
        feature_map = self.fpn2(feature_map, False)

        return_vals['bw_flow'] = bw_flow['flow_0']

        results = self.heads(feature_map, False)

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
            bb_targets_delta = tf.reshape(results['bb_targets_delta'], [-1])
            return_vals['bb_targets_offset'] = bb_targets_offset
            return_vals['bb_targets_cls'] = bb_targets_cls
            return_vals['bb_targets_objectness'] = bb_targets_objectness
            return_vals['bb_targets_embedding'] = bb_targets_embedding
            return_vals['bb_targets_delta'] = bb_targets_delta

        return return_vals

    @tf.function(input_signature=[
            tf.TensorSpec([1, config['eval_image_height'], config['eval_image_width'], 3], tf.float32, 'current_img'),
            tf.TensorSpec([1, config['eval_image_height'], config['eval_image_width'], 3], tf.float32, 'prev_img')])
    def infer(self, image, prev_image):
        current_feature_map = self.inferBackbone(image)
        prev_feature_map = self.inferBackbone(prev_image)
        return self.inferHeads(current_feature_map, prev_feature_map)

# Create model
infer = Infer(config)

# Load the latest checkpoint
checkpoint = tf.train.Checkpoint(backbone=infer.backbone, fpn1=infer.fpn1, fpn2=infer.fpn2, flow=infer.flow,
                                 flow_warp=infer.flow_warp, heads=infer.heads)
checkpoint_manager = tf.train.CheckpointManager(checkpoint, os.path.join(config['state_dir'], 'checkpoints'), 25)
checkpoint.restore(checkpoint_manager.latest_checkpoint)

# Save the trained model
signature_dict = {'infer': infer.infer, 'inferBackbone': infer.inferBackbone, 'inferHeads': infer.inferHeads}
saved_model_dir = os.path.join(config['state_dir'], 'saved_model')
clean_directory(saved_model_dir)
save_options = tf.saved_model.SaveOptions(namespace_whitelist=['Addons'])
tf.saved_model.save(infer, saved_model_dir, signature_dict, save_options)
