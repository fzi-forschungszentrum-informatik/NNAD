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
from .Common import *

class Upsample(tf.keras.Model):
    def __init__(self, name, factor, num_output_channels, bn_and_activation=True):
        super().__init__(name=name)

        self.bn = None
        if bn_and_activation:
            self.bn = Normalization()
            self.activation = tf.keras.layers.Activation('swish')

        kernel_size = 2 * factor - factor % 2
        self.transposed_conv = tf.keras.layers.Conv2DTranspose(num_output_channels,
            (kernel_size, kernel_size),
            (factor, factor),
            padding='same',
            use_bias=not bn_and_activation,
            kernel_initializer=KERNEL_INITIALIZER,
            name='transposed_conv')

    def call(self, x, training=False):
        x = self.transposed_conv(x)
        if self.bn:
            x = self.bn(x, training=training)
            x = self.activation(x)
        return x

class LabelBranch(tf.keras.Model):
    def __init__(self, name, config):
        super().__init__(name=name)

        self.feature_extractor = SeparableConvBlock('feature_extractor', HEADS_NUM_BLOCKS, BIFPN_NUM_FEATURES)
        self.upsample2 = Upsample('upsample', 4, config['num_label_classes'], False)

    def call(self, x, training=False):
        x, _, _, _, _, _ = x # We only care about feature level P2

        x = self.feature_extractor(x, training=training)
        x = self.upsample2(x, training=training)
        return x
