##########################################################################
# NNAD (Neural Networks for Automated Driving) training scripts          #
# Copyright (C) 2020 FZI Research Center for Information Technology      #
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
#                                                                        #
# This file incorporates work covered by the following copyright and     #
# permission notice:                                                     #
#                                                                        #
#   Copyright 2019 The TensorFlow Authors, Pavel Yakubovskiy,            #
#   Björn Barz. All Rights Reserved.                                     #
#                                                                        #
#   Licensed under the Apache License, Version 2.0 (the "License");      #
#   you may not use this file except in compliance with the License.     #
#   You may obtain a copy of the License at                              #
#                                                                        #
#       http://www.apache.org/licenses/LICENSE-2.0                       #
#                                                                        #
#   Unless required by applicable law or agreed to in writing, software  #
#   distributed under the License is distributed on an "AS IS" BASIS,    #
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or      #
#   implied. See the License for the specific language governing         #
#   permissions and limitations under the License.                       #
##########################################################################

import tensorflow as tf
from .Common import *
from .constants import *

import collections
import math

"""
Mobile Inverted Residual Bottleneck layer for EfficientNet
"""
class MbConvBlock(tf.keras.Model):
    def __init__(self, name, block_args, droprate=None):
        super().__init__(name=name)

        num_filters = block_args.input_filters * block_args.expand_ratio

        self.has_skip = block_args.id_skip and all(s == 1 for s in block_args.strides) and block_args.input_filters == block_args.output_filters

        self.has_expansion = block_args.expand_ratio != 1
        if self.has_expansion:
            self.expand_conv = tf.keras.layers.Conv2D(num_filters, 1,
                                                      padding='same',
                                                      use_bias=False,
                                                      kernel_initializer=KERNEL_INITIALIZER,
                                                      kernel_regularizer=tf.keras.regularizers.l2(L2_REGULARIZER_WEIGHT),
                                                      name='expand_conv')
            self.expand_norm = Normalization()
            self.expand_activation = tf.keras.layers.Activation('swish')

        self.conv1 = tf.keras.layers.DepthwiseConv2D(block_args.kernel_size,
                                                     strides=block_args.strides,
                                                     padding='same',
                                                     use_bias=False,
                                                     depthwise_initializer=KERNEL_INITIALIZER,
                                                     kernel_regularizer=tf.keras.regularizers.l2(L2_REGULARIZER_WEIGHT),
                                                     name='conv1')
        self.norm1 = Normalization()
        self.activation1 = tf.keras.layers.Activation('swish')

        self.has_se = (block_args.se_ratio is not None) and (0 < block_args.se_ratio <= 1)
        if self.has_se:
            self.se_pooling = tf.keras.layers.GlobalAveragePooling2D()
            self.se_reshape = tf.keras.layers.Reshape((1, 1, num_filters))
            num_reduced_filters = max(1, int(block_args.input_filters * block_args.se_ratio))
            self.se_conv1 = tf.keras.layers.Conv2D(num_reduced_filters,
                                                   1,
                                                   activation='swish',
                                                   padding='same',
                                                   use_bias=True,
                                                   kernel_initializer=KERNEL_INITIALIZER,
                                                   kernel_regularizer=tf.keras.regularizers.l2(L2_REGULARIZER_WEIGHT),
                                                   name='se_conv1')
            self.se_conv2 = tf.keras.layers.Conv2D(num_filters,
                                                   1,
                                                   activation='sigmoid',
                                                   padding='same',
                                                   use_bias=True,
                                                   kernel_initializer=KERNEL_INITIALIZER,
                                                   kernel_regularizer=tf.keras.regularizers.l2(L2_REGULARIZER_WEIGHT),
                                                   name='se_conv2')

        self.conv2 = tf.keras.layers.Conv2D(block_args.output_filters,
                                            1,
                                            padding='same',
                                            use_bias=False,
                                            kernel_initializer=KERNEL_INITIALIZER,
                                            kernel_regularizer=tf.keras.regularizers.l2(L2_REGULARIZER_WEIGHT),
                                            name='conv2')
        self.norm2 = Normalization()

        self.dropout = None
        if self.has_skip and droprate and droprate > 0:
            self.dropout = tf.keras.layers.Dropout(droprate, noise_shape=(None, 1, 1, 1))


    def call(self, x, training=False):
        xi = x
        if self.has_expansion:
            x = self.expand_conv(x)
            x = self.expand_norm(x, training=training)
            x = self.expand_activation(x)

        x = self.conv1(x)
        x = self.norm1(x, training=training)
        x = self.activation1(x)

        if self.has_se:
            x_se = self.se_pooling(x)
            x_se = self.se_reshape(x_se)
            x_se = self.se_conv1(x_se)
            x_se = self.se_conv2(x_se)
            x = x * x_se

        x = self.conv2(x)
        x = self.norm2(x, training=training)

        if self.dropout is not None:
            x = self.dropout(x)

        if self.has_skip:
            x = x + xi

        return x


"""
EfficientNet backbone
"""
class EfficientNet(tf.keras.Model):
    def __init__(self, name, backbone_args):
        super().__init__(name=name)

        KERNEL_INITIALIZER = tf.keras.initializers.VarianceScaling(2.0, mode='fan_out', distribution='normal')

        self.conv1 = tf.keras.layers.Conv2D(self._round_filters(32, backbone_args.width_coefficient, backbone_args.depth_divisor),
                                            3,
                                            strides=(2, 2),
                                            padding='same',
                                            use_bias=False,
                                            kernel_initializer=KERNEL_INITIALIZER,
                                            name='conv1')
        self.norm1 = Normalization()
        self.activation1 = tf.keras.layers.Activation('swish')

        self.blocks = []
        self.output_idx = []

        # Build blocks
        num_blocks_total = sum(block_args.num_repeat for block_args in BLOCKS_ARGS)
        block_num = 0
        level = 1
        for idx, block_args in enumerate(BLOCKS_ARGS):
            assert block_args.num_repeat > 0
            # Update block input and output filters based on depth multiplier.
            block_args = block_args._replace(
                input_filters=self._round_filters(block_args.input_filters,
                                                  backbone_args.width_coefficient,
                                                  backbone_args.depth_divisor),
                output_filters=self._round_filters(block_args.output_filters,
                                                   backbone_args.width_coefficient,
                                                   backbone_args.depth_divisor),
                num_repeat=self._round_repeats(block_args.num_repeat,
                                               backbone_args.depth_coefficient))

            # The first block needs to take care of stride and filter size increase.
            drop_rate = backbone_args.drop_connect_rate * float(block_num) / num_blocks_total
            block_name = 'block{}a'.format(idx + 1)
            if not all(s == 1 for s in block_args.strides):
                if level >= 3:
                    self.output_idx += [block_num - 1]
                level += 1
            self.blocks += [ MbConvBlock(block_name, block_args, drop_rate) ]
            block_num += 1
            if block_args.num_repeat > 1:
                block_args = block_args._replace(input_filters=block_args.output_filters, strides=[1, 1])
                for bidx in range(block_args.num_repeat - 1):
                    drop_rate = backbone_args.drop_connect_rate * float(block_num) / num_blocks_total
                    block_name = 'block{}{}'.format(idx + 1, bidx + 1)
                    self.blocks += [ MbConvBlock(block_name, block_args, drop_rate) ]
                    block_num += 1
        self.output_idx += [block_num - 1]
        assert len(self.output_idx) == 3

    def _round_filters(self, filters, width_coefficient, depth_divisor):
        """Round number of filters based on width multiplier."""

        filters *= width_coefficient
        new_filters = int(filters + depth_divisor / 2) // depth_divisor * depth_divisor
        new_filters = max(depth_divisor, new_filters)
        # Make sure that round down does not go down by more than 10%.
        if new_filters < 0.9 * filters:
            new_filters += depth_divisor
        return int(new_filters)


    def _round_repeats(self, repeats, depth_coefficient):
        """Round number of repeats based on depth multiplier."""

        return int(math.ceil(depth_coefficient * repeats))


    def call(self, x, training=False):
        x = self.conv1(x)
        x = self.norm1(x, training=training)
        x = self.activation1(x)

        outputs = []
        for i in range(len(self.blocks)):
            x = self.blocks[i](x, training=training)
            if i in self.output_idx:
                outputs += [ x ]
        return outputs
