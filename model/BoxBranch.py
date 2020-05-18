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
import numpy as np
from .constants import *
from .Resnet import *

class SingleDecoder(tf.keras.Model):
    def __init__(self, name, num_output_channels, channels):
        super().__init__(name=name)

        self.num_output_channels = num_output_channels

        self.rm1 = ResnetModule('resnet_module_1', [channels, channels, 2 * channels])
        self.rm2 = ResnetModule('resnet_module_2', [channels, channels, 2 * channels])
        self.last_conv = tf.keras.layers.Conv2D(num_output_channels,
                (1, 1),
                kernel_initializer=tf.keras.initializers.he_normal(),
                name='last_conv')

    def call(self, x, train_batch_norm=False):
        n = x.get_shape().as_list()[0]
        # This is an ugly hack for saved_model. Without it fails to determine the batch size
        if n is None:
            n = 1

        x = self.rm1(x, train_batch_norm=train_batch_norm)
        x = self.rm2(x, train_batch_norm=train_batch_norm)
        x = self.last_conv(x)
        x = tf.reshape(x, [n, -1, self.num_output_channels])
        return x

class BoxDecoder(tf.keras.Model):
    def __init__(self, name, config, box_delta_regression=False):
        super().__init__(name=name)

        boxes_per_pos = config['boxes_per_pos']
        self.config = config

        self.box_decoder = SingleDecoder('box_decoder', 4 * boxes_per_pos, 64)
        self.cls_decoder = SingleDecoder('cls_decoder', config['num_bb_classes'] * boxes_per_pos, 64)
        self.obj_decoder = SingleDecoder('obj_decoder', 2 * boxes_per_pos, 64)
        self.embedding_decoder = SingleDecoder('embedding_decoder',
                                               config['box_embedding_len'] * boxes_per_pos, 128)

        self.delta_decoder = None
        if box_delta_regression:
            self.delta_decoder = SingleDecoder('delta_decoder', 4 * boxes_per_pos, 64)

    def call(self, x, train_batch_norm=False):
        n = x.get_shape().as_list()[0]
        # This is an ugly hack for saved_model. Without it fails to determine the batch size
        if n is None:
            n = 1

        box = self.box_decoder(x, train_batch_norm=train_batch_norm)
        cls = self.cls_decoder(x, train_batch_norm=train_batch_norm)
        obj = self.obj_decoder(x, train_batch_norm=train_batch_norm)
        embedding = self.embedding_decoder(x, train_batch_norm=train_batch_norm)
        embedding = tf.reshape(embedding, [n, -1, self.config['box_embedding_len']])
        embeding = tf.math.l2_normalize(embedding, axis=-1)
        embedding = tf.reshape(embedding, [n, -1, self.config['box_embedding_len'] * self.config['boxes_per_pos']])

        results = [box, cls, obj, embedding]

        if self.delta_decoder:
            delta = self.delta_decoder(x, train_batch_norm=train_batch_norm)
            results += [delta]

        return results

class BoxBranch(tf.keras.Model):
    def __init__(self, name, config, box_delta_regression=False):
        super().__init__(name=name)
        self.core_branch = ResnetBranch('box_branch')

        self.downsample1 = ResnetModule('downsample_1', [512, 512, 1024], stride=2)
        self.downsample2 = ResnetModule('downsample_2', [512, 512, 1024], stride=2)
        self.downsample3 = ResnetModule('downsample_3', [512, 512, 1024], stride=2)
        self.downsample4 = ResnetModule('downsample_4', [512, 512, 1024], stride=2)

        self.decoder = BoxDecoder('decoder', config, box_delta_regression)

        self.box_delta_regression = box_delta_regression

    def call(self, x, train_batch_norm=False):
        x = self.core_branch(x, train_batch_norm=train_batch_norm)

        p3 = x
        p4 = self.downsample1(p3, train_batch_norm=train_batch_norm)
        p5 = self.downsample2(p4, train_batch_norm=train_batch_norm)
        p6 = self.downsample3(p5, train_batch_norm=train_batch_norm)
        p7 = self.downsample4(p6, train_batch_norm=train_batch_norm)

        res = []
        res += [self.decoder(p3, train_batch_norm=train_batch_norm)]
        res += [self.decoder(p4, train_batch_norm=train_batch_norm)]
        res += [self.decoder(p5, train_batch_norm=train_batch_norm)]
        res += [self.decoder(p6, train_batch_norm=train_batch_norm)]
        res += [self.decoder(p7, train_batch_norm=train_batch_norm)]

        box = tf.concat([entry[0] for entry in res], axis=1)
        cls = tf.concat([entry[1] for entry in res], axis=1)
        obj = tf.concat([entry[2] for entry in res], axis=1)
        embedding = tf.concat([entry[3] for entry in res], axis=1)

        results = [box, cls, obj, embedding]

        if self.box_delta_regression:
            delta_regression = tf.concat([entry[4] for entry in res], axis=1)
            results += [delta_regression]

        return results
