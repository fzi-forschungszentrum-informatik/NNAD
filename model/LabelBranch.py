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
from .Resnet import *

class Upsample(tf.keras.Model):
    def __init__(self, name, factor, num_output_channels):
        super().__init__(name=name)

        self.bn = Normalization()

        kernel_size = 2 * factor - factor % 2
        self.transposed_conv = tf.keras.layers.Conv2DTranspose(num_output_channels,
            (kernel_size, kernel_size),
            (factor, factor),
            padding='same',
            kernel_initializer=tf.keras.initializers.he_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(L2_REGULARIZER_WEIGHT),
            name='transposed_conv')

    def call(self, x, train_batch_norm=False):
        x = self.bn(x, training=train_batch_norm)
        x = self.transposed_conv(x)
        x = tf.nn.relu(x)
        return x

class LabelBranch(tf.keras.Model):
    def __init__(self, name, config):
        super().__init__(name=name)

        self.core_branch = ResnetBranch('label_branch')
        self.upsample1 = Upsample('upsample1', 2, 128)
        self.upsample2 = Upsample('upsample2', 2, 64)
        self.upsample3 = Upsample('upsample2', 2, 32)
        self.final_conv = tf.keras.layers.Conv2D(config['num_label_classes'],
            (1, 1),
            kernel_initializer=tf.keras.initializers.he_normal(),
            name='final_conv')

    def call(self, x, train_batch_norm=False):
        x = self.core_branch(x, train_batch_norm=train_batch_norm)
        x = self.upsample1(x, train_batch_norm=train_batch_norm)
        x = self.upsample2(x, train_batch_norm=train_batch_norm)
        x = self.upsample3(x, train_batch_norm=train_batch_norm)
        x = self.final_conv(x)
        return x
