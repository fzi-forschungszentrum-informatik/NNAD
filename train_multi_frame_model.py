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
from datetime import datetime
import os
import time
import random

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from model.constants import *
from model.Heads import *
from model.EfficientNet import *
from model.BiFPN import *
from model.Flow import *
from model.loss.BackwardFlowLoss import *
from model.loss.LabelLoss import *
from model.loss.BoxLoss import *
from model.loss.EmbeddingLoss import *
from data import *
from helpers.configreader import *

# Argument handling
config, config_path = get_config()

# Random seed
random.seed()

# Create a summary writer
train_summary_writer = tf.summary.create_file_writer(os.path.join(config['state_dir'], 'summaries_multi'))

# Create the dataset and the global step variable
ds = Dataset(settings_path=config_path, mode='train')
with tf.device('/cpu:0'):
    global_step = tf.Variable(0, 'global_multi_step')

# Define the learning rate schedule
learning_rate_fn, weight_decay_fn, max_train_steps = get_learning_rate_fn(config['multi_frame'], global_step, 1.0e-7)

# Create an optimizer, the network and the loss class
opt = tfa.optimizers.AdamW(learning_rate=learning_rate_fn, weight_decay=weight_decay_fn)

# Models
backbone = EfficientNet('backbone', BACKBONE_ARGS)
fpn1 = BiFPN('bifpn1', BIFPN_NUM_FEATURES, int(BIFPN_NUM_BLOCKS / 2), True)
fpn2 = BiFPN('bifpn2', BIFPN_NUM_FEATURES, BIFPN_NUM_BLOCKS - int(BIFPN_NUM_BLOCKS / 2), False)
flow = Flow('flow')
flow_warp = FlowWarp('flow_warp')
heads = Heads('heads', config, box_delta_regression=True)

flow_loss = BackwardFlowLoss('backward_flow_loss')
label_loss = LabelLoss('label_loss', config)
box_loss = BoxLoss('box_loss', config, box_delta_regression=True)
embedding_loss = EmbeddingLoss('embedding_loss', config)

@tf.function
def train_step():
    images, ground_truth, metadata = ds.get_batched_data(config['multi_frame']['batch_size_per_gpu'])

    current_feature_map = backbone(images['left'], False)
    current_feature_map = fpn1(current_feature_map, False)
    prev_feature_map = backbone(images['prev_left'], False)
    prev_feature_map = fpn1(prev_feature_map, False)
    bw_flow = flow([prev_feature_map, current_feature_map], False)
    with tf.GradientTape(persistent=True) as tape:
        feature_map = flow_warp([current_feature_map,
                                 prev_feature_map,
                                 bw_flow],
                                True)
        feature_map = fpn2(feature_map, True)
        results = heads(feature_map, True)

        losses = []
        if config['train_labels']:
            losses += [label_loss([results, ground_truth], tf.cast(global_step, tf.int64))]
        if config['train_boundingboxes']:
            losses += box_loss([results, ground_truth], tf.cast(global_step, tf.int64))
            losses += [embedding_loss([results, ground_truth], tf.cast(global_step, tf.int64))]

        ## Sum up all losses
        summed_losses = tf.add_n(losses)
        tf.summary.scalar('summed_losses', summed_losses, tf.cast(global_step, tf.int64))

        ## Regularization terms
        vs = flow_warp.trainable_variables + fpn2.trainable_variables + heads.trainable_variables
        if config['train_labels']:
            vs += label_loss.trainable_variables
        if config['train_boundingboxes']:
            vs += box_loss.trainable_variables + embedding_loss.trainable_variables
        total_loss = summed_losses

    gs = tape.gradient(total_loss, vs)
    opt.apply_gradients(zip(gs, vs))
    return total_loss

@tf.function
def combined_train_step():
    images, ground_truth, metadata = ds.get_batched_data(config['multi_frame']['batch_size_per_gpu'])

    with tf.GradientTape(persistent=True) as tape:
        with tf.device('/gpu:0'):
            current_feature_map = backbone(images['left'], True)
            current_feature_map = fpn1(current_feature_map, True)
            prev_feature_map = backbone(images['prev_left'], True)
            prev_feature_map = fpn1(prev_feature_map, True)
        with tf.device('/gpu:1'):
            bw_flow = flow([prev_feature_map, current_feature_map], True)
            feature_map = flow_warp([current_feature_map,
                                     prev_feature_map,
                                     bw_flow],
                                    True)
            feature_map = fpn2(feature_map, True)
            results = heads(feature_map, True)

            flow_l = flow_loss([bw_flow, images, ground_truth], tf.cast(global_step, tf.int64))
            losses = [flow_l]
            if config['train_labels']:
                losses += [label_loss([results, ground_truth], tf.cast(global_step, tf.int64))]
            if config['train_boundingboxes']:
                losses += box_loss([results, ground_truth], tf.cast(global_step, tf.int64))
                losses += [embedding_loss([results, ground_truth], tf.cast(global_step, tf.int64))]

        ## Sum up all losses
        summed_losses = tf.add_n(losses)
        tf.summary.scalar('summed_losses', summed_losses, tf.cast(global_step, tf.int64))

        ## Regularization terms
        vs = backbone.trainable_variables + fpn1.trainable_variables + flow.trainable_variables + \
             flow_warp.trainable_variables + fpn2.trainable_variables + heads.trainable_variables + \
             flow_loss.trainable_variables
        if config['train_labels']:
            vs += label_loss.trainable_variables
        if config['train_boundingboxes']:
            vs += box_loss.trainable_variables + embedding_loss.trainable_variables
        total_loss = summed_losses

    gs = tape.gradient(total_loss, vs)
    opt.apply_gradients(zip(gs, vs))
    return total_loss

# Load checkpoints
checkpoint = tf.train.Checkpoint(backbone=backbone, fpn1=fpn1, fpn2=fpn2, flow=flow, flow_warp=flow_warp,
                                 heads=heads, label_loss=label_loss, box_loss=box_loss,
                                 embedding_loss=embedding_loss, optimizer=opt, global_multi_step=global_step)
checkpoint_manager = tf.train.CheckpointManager(checkpoint, os.path.join(config['state_dir'], 'checkpoints'), 25)
checkpoint_status = checkpoint.restore(checkpoint_manager.latest_checkpoint)

# Training loop
step = global_step.numpy()
while step < max_train_steps:
    # Run training step
    with train_summary_writer.as_default():
        start_time = time.time()
        total_loss = combined_train_step()
        duration = time.time() - start_time

    assert not np.isnan(total_loss.numpy()), 'Model diverged with loss = NaN'

    # Write training progress to console
    if step % 10 == 0:
        examples_per_step = config['multi_frame']['batch_size_per_gpu']
        sec_per_batch = float(duration)
        examples_per_sec = examples_per_step / duration

        print('%s: step %d, lr = %e, loss = %.6f (%.1f examples/sec: %.3f sec/batch)' %
              (datetime.now(), step, learning_rate_fn().numpy(), total_loss.numpy(), examples_per_sec,
               sec_per_batch))

    # Save checkpoints
    if step > 0 and (step % 1000 == 0 or (step + 1) == max_train_steps):
        checkpoint_manager.save(global_step)

    # Increase step counter
    global_step.assign(global_step + 1)
    step = global_step.numpy()
