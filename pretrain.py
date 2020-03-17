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

from model.PretrainHead import *
from model.Resnet import *
from model.loss.PretrainLoss import *
from data import *
from helpers.configreader import *

# Argument handling
config, config_path = get_config()

# Random seed
random.seed()

# Create a summary writer
train_summary_writer = tf.summary.create_file_writer(os.path.join(config['state_dir'], 'pretrain_summaries'))

# Create the dataset and the global step variable
ds = Dataset(settings_path=config_path, mode='pretrain')
with tf.device('/cpu:0'):
    global_step = tf.Variable(0, 'global_pretrain_step')

# Define a learning rate schedule
max_pretrain_steps = 2500000
pretrain_base_learning_rate = 1e-3

def learning_rate_fn():
    return pretrain_base_learning_rate * (1.0 - tf.pow(global_step / max_pretrain_steps, 0.9))

# Create an optimizer, the network and the loss class
opt = tfa.optimizers.LAMB(learning_rate_fn)

# Models
backbone = ResnetBackbone('backbone', pretrain=True)
pretrain_head = PretrainHead('pretrain_head', config)
pretrain_loss = PretrainLoss('pretrain_loss', config)

# Define a training step function
@tf.function
def train_step():
    images, ground_truth, metadata = ds.get_batched_data(config['pretrain_batch_size_per_gpu'])
    with tf.GradientTape() as tape:
        feature_map = backbone(images['left'], config['pretrain_batch_norm'])
        result = pretrain_head(feature_map, config['pretrain_batch_norm'])
        loss = pretrain_loss([result, ground_truth], tf.cast(global_step, tf.int64))

        loss = loss + tf.add_n(backbone.losses + pretrain_head.losses + pretrain_loss.losses)

    vs = backbone.trainable_variables + pretrain_head.trainable_variables + pretrain_loss.trainable_variables
    gs = tape.gradient(loss, vs)
    opt.apply_gradients(zip(gs, vs))
    return loss

# Load checkpoints
checkpoint = tf.train.Checkpoint(backbone=backbone, pretrain_head=pretrain_head, pretrain_loss=pretrain_loss,
                                    optimizer=opt, pretrain_global_step=global_step)
checkpoint_manager = tf.train.CheckpointManager(checkpoint, os.path.join(config['state_dir'], 'checkpoints'), 25)
checkpoint.restore(checkpoint_manager.latest_checkpoint)

# Training loop
step = global_step.numpy()
summary_step = step + 20
while step < max_pretrain_steps:
    # Enable trace
    if step == summary_step:
        tf.summary.trace_on()

    # Run training step
    with train_summary_writer.as_default():
        start_time = time.time()
        total_loss = train_step()
        duration = time.time() - start_time

    assert not np.isnan(total_loss.numpy()), 'Model diverged with loss = NaN'

    # Write training progress to console
    if step % 10 == 0:
        examples_per_step = config['pretrain_batch_size_per_gpu']
        sec_per_batch = float(duration)
        examples_per_sec = examples_per_step / duration

        print('%s: step %d, lr = %e, loss = %.6f (%.1f examples/sec: %.3f sec/batch)' %
                (datetime.now(), step, learning_rate_fn().numpy(), total_loss.numpy(), examples_per_sec,
                sec_per_batch))

    # Save trace
    if step == summary_step:
        tf.summary.trace_export("Trace", step)

    # Save checkpoints
    if step > 0 and (step % 1000 == 0 or (step + 1) == config['max_steps']):
        checkpoint_manager.save(global_step)

    # Increase step counter
    global_step.assign(global_step + 1)
    step = global_step.numpy()
