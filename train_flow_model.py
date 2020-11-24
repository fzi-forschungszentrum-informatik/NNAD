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
from model.loss.FlowLoss import *
from data import *
from helpers.configreader import *

# Argument handling
config, config_path = get_config()

# Random seed
random.seed()

# Create a summary writer
train_summary_writer = tf.summary.create_file_writer(os.path.join(config['state_dir'], 'summaries_flow'))

# Create the dataset and the global step variable
flow_ds = Dataset(settings_path=config_path, mode='flow')
with tf.device('/cpu:0'):
    global_step = tf.Variable(0, 'global_flow_step')

# Define the learning rate schedule
learning_rate_fn, weight_decay_fn, max_train_steps = get_learning_rate_fn(config['single_frame'], global_step, 1.0e-7)

# Create an optimizer, the network and the loss class
opt = tfa.optimizers.AdamW(learning_rate=learning_rate_fn, weight_decay=weight_decay_fn)

# Models
backbone = EfficientNet('backbone', BACKBONE_ARGS)
fpn1 = BiFPN('bifpn1', BIFPN_NUM_FEATURES, int(BIFPN_NUM_BLOCKS / 2), True)
flow = Flow('flow')

flow_loss = FlowLoss('flow_loss')

# Define a training step function for single images
@tf.function
def train_step():
    flow_images, flow_ground_truth, flow_metadata = flow_ds.get_batched_data(config['flow']['batch_size_per_gpu'])

    ## Optical flow loss
    with tf.GradientTape(persistent=True) as tape:
        current_flow_feature_map = backbone(flow_images['left'], True) #False)
        current_flow_feature_map = fpn1(current_flow_feature_map, True) #False)
        previous_flow_feature_map = backbone(flow_images['prev_left'], True) #False)
        previous_flow_feature_map = fpn1(previous_flow_feature_map, True) #False)
    #current_flow_feature_map = [tf.stop_gradient(x) for x in current_flow_feature_map]
    #previous_flow_feature_map = [tf.stop_gradient(x) for x in previous_flow_feature_map]
    #with tf.GradientTape(persistent=True) as tape:
        flow_results = flow([current_flow_feature_map, previous_flow_feature_map], True)
        flow_l = flow_loss([flow_results, flow_images, flow_ground_truth], tf.cast(global_step, tf.int64))
        vs = flow.trainable_variables + flow_loss.trainable_variables
        total_loss = flow_l
    gs = tape.gradient(total_loss, vs)
    opt.apply_gradients(zip(gs, vs))
    return total_loss

# Load checkpoints
checkpoint = tf.train.Checkpoint(backbone=backbone, fpn1=fpn1, flow=flow,
                                 flow_loss=flow_loss, optimizer=opt, global_flow_step=global_step)
checkpoint_manager = tf.train.CheckpointManager(checkpoint, os.path.join(config['state_dir'], 'checkpoints'), 25)
checkpoint_status = checkpoint.restore(checkpoint_manager.latest_checkpoint)

# Training loop
step = global_step.numpy()
summary_step = step + 1
while step < max_train_steps:
    # Run training step
    with train_summary_writer.as_default():
        start_time = time.time()
        total_loss = train_step()
        duration = time.time() - start_time

    assert not np.isnan(total_loss.numpy()), 'Model diverged with loss = NaN'

    # Write training progress to console
    if step % 10 == 0:
        examples_per_step = config['flow']['batch_size_per_gpu']
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
