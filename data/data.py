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

import tensorflow as tf
import random
import os

current_dir = os.path.dirname(os.path.realpath(__file__))
dataset_module = tf.load_op_library(os.path.join(current_dir, 'build', 'libdataset_op.so'))
random.seed()

class Dataset(object):
    def __init__(self, **kwargs):
        self.id_ = random.randint(0, 2**30)
        self.kwargs_ = kwargs
        self.input_keys = ['left', 'prev_left']
        self.gt_keys = ['flow', 'flow_mask', 'cls', 'pixelwise_labels',
                        'bb_targets_objectness', 'bb_targets_cls', 'bb_targets_id', 'bb_targets_prev_id',
                        'bb_targets_offset', 'bb_targets_delta_valid', 'bb_targets_delta', 'bb_list']
        self.metadata_keys = ['key', 'original_width', 'original_height']

    def get_data(self):
        with tf.device('/cpu:0'):
            data = dataset_module.Dataset(id=self.id_, **self.kwargs_)
            inputs = {}
            gt = {}
            metadata = {}
            idx = 0
            for key in self.input_keys:
                inputs[key] = data[idx]
                idx += 1
            for key in self.gt_keys:
                gt[key] = data[idx]
                idx += 1
            for key in self.metadata_keys:
                metadata[key] = data[idx]
                idx += 1

        return inputs, gt, metadata

    def get_batched_data(self, batch_size):
        with tf.device('/cpu:0'):
            inputs = {}
            gt = {}
            metadata = {}

            lists = [ [] for i in range(len(self.input_keys) + len(self.gt_keys) + len(self.metadata_keys)) ]

            for i in range(batch_size):
                data = dataset_module.Dataset(id=self.id_, **self.kwargs_)
                idx = 0
                for i in range(len(lists)):
                    lists[i] += [data[idx]]
                    idx += 1

            for i in range(len(self.input_keys)):
                inputs[self.input_keys[i]] = tf.stack(lists[i + 0])
            for i in range(len(self.gt_keys)):
                if self.gt_keys[i] == 'bb_list':
                    gt[self.gt_keys[i]] = tf.ragged.stack(lists[i + len(self.input_keys)])
                else:
                    gt[self.gt_keys[i]] = tf.stack(lists[i + len(self.input_keys)])
            for i in range(len(self.metadata_keys)):
                metadata[self.metadata_keys[i]] = tf.stack(lists[i + len(self.input_keys) + len(self.gt_keys)])

        return inputs, gt, metadata

