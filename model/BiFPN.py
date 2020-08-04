##########################################################################
# NNAD (Neural Networks for Automated Driving) training scripts          #
# Copyright (C) 2020 FZI Research Center for Information Technology      #
# Copyright (C) 2020 Guillem Orellana Trullols                           #
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

class FastFusion(tf.keras.Model):
    def __init__(self, name, size, num_features):
        super().__init__(name = name)

        self.EPSILON = 1e-5

        self.size = size
        self.w = self.add_weight(name='w',
                                 shape=(size,),
                                 initializer=tf.initializers.Ones(),
                                 trainable=True)
        self.relu = tf.keras.layers.Activation('relu')

        self.conv = tf.keras.layers.SeparableConv2D(num_features,
                                                    kernel_size=3,
                                                    padding='same',
                                                    use_bias=False,
                                                    kernel_initializer=KERNEL_INITIALIZER,
                                                    name='conv')
        self.norm = Normalization()
        self.activation = tf.keras.layers.Activation('swish')
        self.resize = Resize('resize', num_features)

    def call(self, inputs, training=False):

        resampled_feature = self.resize([inputs[-1], tf.shape(inputs[0])], training=training)
        resampled_features = inputs[:-1] + [resampled_feature]

        # wi has to be larger than 0 -> ReLU
        w = self.relu(self.w)
        w_sum = self.EPSILON + tf.reduce_sum(w, axis=0)

        weighted_inputs = [(w[i] * resampled_features[i]) / w_sum for i in range(self.size)]
        weighted_sum = tf.add_n(weighted_inputs)
        x = self.conv(weighted_sum, training=training)
        x = self.norm(x, training=training)
        x = self.activation(x)
        return x


class BiFPNBlock(tf.keras.Model):
    def __init__(self, name, num_features):
        super().__init__(name = name)

        # Feature fusion for intermediate level
        self.ff_6_td = FastFusion('ff_6_td', 2, num_features)
        self.ff_5_td = FastFusion('ff_5_td', 2, num_features)
        self.ff_4_td = FastFusion('ff_4_td', 2, num_features)
        self.ff_3_td = FastFusion('ff_3_td', 2, num_features)

        # Feature fusion for output
        self.ff_7_out = FastFusion('ff_7_out', 2, num_features)
        self.ff_6_out = FastFusion('ff_6_out', 3, num_features)
        self.ff_5_out = FastFusion('ff_5_out', 3, num_features)
        self.ff_4_out = FastFusion('ff_4_out', 3, num_features)
        self.ff_3_out = FastFusion('ff_3_out', 3, num_features)
        self.ff_2_out = FastFusion('ff_2_out', 2, num_features)

    def call(self, inputs, training=False):
        P2, P3, P4, P5, P6, P7 = inputs

        # Compute the intermediate state
        # Note that P2 and P7 have no intermediate state
        P6_td = self.ff_6_td([P6, P7], training=training)
        P5_td = self.ff_5_td([P5, P6_td], training=training)
        P4_td = self.ff_4_td([P4, P5_td], training=training)
        P3_td = self.ff_3_td([P3, P4_td], training=training)

        # Compute output features maps
        P2_out = self.ff_2_out([P2, P3_td], training=training)
        P3_out = self.ff_3_out([P3, P3_td, P2_out], training=training)
        P4_out = self.ff_4_out([P4, P4_td, P3_out], training=training)
        P5_out = self.ff_5_out([P5, P5_td, P4_out], training=training)
        P6_out = self.ff_6_out([P6, P6_td, P5_out], training=training)
        P7_out = self.ff_7_out([P7, P6_td], training=training)

        return [P2_out, P3_out, P4_out, P5_out, P6_out, P7_out]

class BiFPNPreproc(tf.keras.Model):
    def __init__(self, name, num_features):
        super().__init__(name = name)

        self.conv_p2 = tf.keras.layers.Conv2D(num_features,
                                              kernel_size=1,
                                              padding='same',
                                              kernel_initializer=KERNEL_INITIALIZER,
                                              name='conv_p2')
        self.conv_p3 = tf.keras.layers.Conv2D(num_features,
                                              kernel_size=1,
                                              padding='same',
                                              kernel_initializer=KERNEL_INITIALIZER,
                                              name='conv_p3')
        self.conv_p4 = tf.keras.layers.Conv2D(num_features,
                                              kernel_size=1,
                                              padding='same',
                                              kernel_initializer=KERNEL_INITIALIZER,
                                              name='conv_p4')
        self.conv_p5 = tf.keras.layers.Conv2D(num_features,
                                              kernel_size=1,
                                              padding='same',
                                              kernel_initializer=KERNEL_INITIALIZER,
                                              name='conv_p5')

        self.downsample1 = Downsample('downsample1', num_features)
        self.activation1 = tf.keras.layers.Activation('swish')
        self.downsample2 = Downsample('downsample2', num_features)

    def call(self, inputs, training=False):
        P2, P3, P4, P5 = inputs

        P2_out = self.conv_p2(P2)
        P3_out = self.conv_p3(P3)
        P4_out = self.conv_p4(P4)
        P5_out = self.conv_p5(P5)
        P6_out = self.downsample1(P5)
        P7_out = self.downsample2(self.activation1(P6_out))

        return [P2_out, P3_out, P4_out, P5_out, P6_out, P7_out]

class BiFPN(tf.keras.Model):
    def __init__(self, name, num_features, num_blocks, needs_preproc):
        super().__init__(name = name)

        self.preproc = None
        if needs_preproc:
            self.preproc = BiFPNPreproc('bifpn_preproc', num_features)

        self.blocks = []
        for i in range(num_blocks):
            self.blocks += [ BiFPNBlock('bifpn_block', num_features) ]

    def call(self, x, training = False):
        if self.preproc is not None:
            x = self.preproc(x, training=training)
        for block in self.blocks:
            x = block(x, training=training)
        return x
