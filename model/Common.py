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
##########################################################################

import tensorflow as tf
from .constants import *

class Downsample(tf.keras.Model):
    def __init__(self, name, num_output_channels):
        super().__init__(name=name)

        self.bn = Normalization()

        self.conv = tf.keras.layers.Conv2D(num_output_channels,
                                           (3, 3),
                                           padding='same',
                                           strides=(2, 2),
                                           use_bias=False,
                                           kernel_initializer=KERNEL_INITIALIZER,
                                           kernel_regularizer=tf.keras.regularizers.l2(L2_REGULARIZER_WEIGHT),
                                           name='downsample')

    def call(self, x, training=False):
        x = self.conv(x)
        x = self.bn(x, training=training)
        return x

class Resize(tf.keras.Model):
    def __init__(self, name, num_features):
        super().__init__(name=name)
        self.antialiasing_conv = tf.keras.layers.SeparableConv2D(num_features,
                                                                 kernel_size=3,
                                                                 padding='same',
                                                                 use_bias=False,
                                                                 kernel_initializer=KERNEL_INITIALIZER,
                                                                 kernel_regularizer=tf.keras.regularizers.l2(L2_REGULARIZER_WEIGHT),
                                                                 name='antialiasing_conv')
        self.norm = Normalization()

    def call(self, inputs, training=False):
        images, target_dim = inputs
        h, w = target_dim[1], target_dim[2]
        x = tf.image.resize(images, [h, w], method='nearest')
        x = self.antialiasing_conv(x)
        x = self.norm(x, training=training)
        return x

class SeparableConv(tf.keras.Model):
    def __init__(self, name, num_features):
        super().__init__(name=name)
        self.conv = tf.keras.layers.SeparableConv2D(num_features,
                                                    kernel_size=3,
                                                    padding='same',
                                                    use_bias=False,
                                                    kernel_initializer=KERNEL_INITIALIZER,
                                                    kernel_regularizer=tf.keras.regularizers.l2(L2_REGULARIZER_WEIGHT),
                                                    name='conv')
        self.norm = Normalization()
        self.activation = tf.keras.layers.Activation('swish')

    def call(self, x, training=False):
        x = self.conv(x)
        x = self.norm(x, training=training)
        x = self.activation(x)
        return x

class SeparableConvBlock(tf.keras.Model):
    def __init__(self, name, depth, num_features):
        super().__init__(name=name)

        self.convs = [SeparableConv('conv_{}'.format(i), num_features) for i in range(depth)]

    def call(self, x, training=False):
        for conv in self.convs:
            x = conv(x, training=training)
        return x
