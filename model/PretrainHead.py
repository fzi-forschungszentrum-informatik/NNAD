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

class PretrainHead(tf.keras.Model):
    def __init__(self, name, config):
        super().__init__(name=name)

        self.pre_conv = tf.keras.layers.Conv2D(config['pretrain']['num_classes'],
                                               1,
                                               padding='same',
                                               use_bias=False,
                                               kernel_initializer=KERNEL_INITIALIZER,
                                               kernel_regularizer=tf.keras.regularizers.l2(L2_REGULARIZER_WEIGHT),
                                               name='pre_conv')
        self.norm = Normalization()
        self.activation = tf.keras.layers.Activation('swish')
        self.avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.final_conv = tf.keras.layers.Dense(config['pretrain']['num_classes'],
            kernel_initializer=KERNEL_INITIALIZER,
            name='final_conv')

    def call(self, inputs, training=False):
        p3, p4, p5 = inputs
        x = p5
        x = self.pre_conv(x)
        x = self.norm(x, training=training)
        x = self.activation(x)
        x = self.avg_pool(x)
        x = self.final_conv(x)
        return x
