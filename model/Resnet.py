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
from .constants import *

class ResnetModule(tf.keras.Model):
    def __init__(self, name, num_filters, stride=1, dilation_rate=1):
        super().__init__(name=name)
        assert stride <= 2
        self.num_filters = num_filters
        self.stride = stride
        self.dilation_rate = dilation_rate

    def build(self, input_shapes):
        self.has_skip_conv = input_shapes[-1] != self.num_filters[-1] or self.stride > 1

        self.bn1 = tf.keras.layers.BatchNormalization()
        self.bn2 = tf.keras.layers.BatchNormalization()

        self.conv1 = tf.keras.layers.SeparableConv2D(
            self.num_filters[0],
            (3, 3),
            padding='same',
            strides=(self.stride, self.stride),
            dilation_rate=(self.dilation_rate, self.dilation_rate),
            kernel_initializer=tf.keras.initializers.he_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(L2_REGULARIZER_WEIGHT),
            name='conv1')

        self.conv2 = tf.keras.layers.SeparableConv2D(
            self.num_filters[1],
            (3, 3),
            padding='same',
            dilation_rate=(self.dilation_rate, self.dilation_rate),
            kernel_initializer=tf.keras.initializers.he_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(L2_REGULARIZER_WEIGHT),
            name='conv2')

        if self.has_skip_conv:
            self.bn_skip = tf.keras.layers.BatchNormalization()

            self.conv_skip = tf.keras.layers.Conv2D(self.num_filters[1],
                (1, 1),
                strides=(self.stride, self.stride),
                kernel_initializer=tf.keras.initializers.he_normal(),
                kernel_regularizer=tf.keras.regularizers.l2(L2_REGULARIZER_WEIGHT),
                name='conv_skip')

    def call(self, x, train_batch_norm=False):
        x = self.bn1(x, training=train_batch_norm)
        x = tf.keras.activations.relu(x)
        skip = x
        if self.has_skip_conv:
            skip = self.conv_skip(skip)
        x = self.conv1(x)
        x = self.bn2(x, training=train_batch_norm)
        x = tf.keras.activations.relu(x)
        x = self.conv2(x)
        x = x + skip
        return x

class ResnetBackbone(tf.keras.Model):
    def __init__(self, name, pretrain=False):
        super().__init__(name=name)

        self.first_conv = tf.keras.layers.Conv2D(64,
            (7, 7),
            padding='same',
            strides=(2, 2),
            kernel_initializer=tf.keras.initializers.he_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(L2_REGULARIZER_WEIGHT),
            name='first_conv')

        self.bn1 = tf.keras.layers.BatchNormalization()
        self.maxpool1 = tf.keras.layers.MaxPooling2D(2)
        self.maxpool2 = tf.keras.layers.MaxPooling2D(2)

        self.module_1a = ResnetModule('resnet_module_1a', [64, 64])
        self.module_1b = ResnetModule('resnet_module_1b', [64, 64])
        self.module_1c = ResnetModule('resnet_module_1c', [64, 64])

        self.module_2a = ResnetModule('resnet_module_2a', [128, 128])
        self.module_2b = ResnetModule('resnet_module_2b', [128, 128])
        self.module_2c = ResnetModule('resnet_module_2c', [128, 128])
        self.module_2d = ResnetModule('resnet_module_2d', [128, 128])

        stride = 1
        dilation_rate = 1
        if pretrain:
            stride = 2
        else:
            dilation_rate = 2
        self.module_3a = ResnetModule('resnet_module_3a', [256, 256], stride=stride,
                                      dilation_rate=dilation_rate)
        self.module_3b = ResnetModule('resnet_module_3b', [256, 256],
                                      dilation_rate=dilation_rate)
        self.module_3c = ResnetModule('resnet_module_3c', [256, 256],
                                      dilation_rate=dilation_rate)

        if not pretrain:
            dilation_rate = 4
        self.module_3d = ResnetModule('resnet_module_3d', [512, 512], stride=stride,
                                      dilation_rate=dilation_rate)
        self.module_3e = ResnetModule('resnet_module_3e', [512, 512],
                                      dilation_rate=2*dilation_rate)
        self.module_3f = ResnetModule('resnet_module_3f', [512, 512],
                                      dilation_rate=dilation_rate)

    def call(self, x, train_batch_norm=False):
        x = self.first_conv(x)
        x = tf.keras.activations.relu(x)
        x = self.bn1(x, training=train_batch_norm)

        x = self.module_1a(x, train_batch_norm=train_batch_norm)
        x = self.module_1b(x, train_batch_norm=train_batch_norm)

        x = self.maxpool1(x)

        x = self.module_1c(x, train_batch_norm=train_batch_norm)
        x = self.module_2a(x, train_batch_norm=train_batch_norm)

        x = self.maxpool2(x)

        x = self.module_2b(x, train_batch_norm=train_batch_norm)
        x = self.module_2c(x, train_batch_norm=train_batch_norm)
        x = self.module_2d(x, train_batch_norm=train_batch_norm)

        x = self.module_3a(x, train_batch_norm=train_batch_norm)
        x = self.module_3b(x, train_batch_norm=train_batch_norm)
        x = self.module_3c(x, train_batch_norm=train_batch_norm)

        x = self.module_3d(x, train_batch_norm=train_batch_norm)
        x = self.module_3e(x, train_batch_norm=train_batch_norm)
        x = self.module_3f(x, train_batch_norm=train_batch_norm)
        return x


class ResnetBranch(tf.keras.Model):
    def __init__(self, name):
        super().__init__(name=name)

        self.module_4a = ResnetModule('resnet_module_4a', [512, 512])
        self.module_4b = ResnetModule('resnet_module_4b', [512, 512])
        self.module_4c = ResnetModule('resnet_module_4c', [512, 512])

    def call(self, x, train_batch_norm=False):
        x = self.module_4a(x, train_batch_norm=train_batch_norm)
        x = self.module_4b(x, train_batch_norm=train_batch_norm)
        x = self.module_4c(x, train_batch_norm=train_batch_norm)
        return x
