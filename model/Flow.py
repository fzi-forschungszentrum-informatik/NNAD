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
import tensorflow_addons as tfa
from .constants import *
from .Common import *

class FlowUpsample(tf.keras.Model):
    def __init__(self, name):
        super().__init__(name=name)

        self.transposed_conv = tf.keras.layers.Conv2DTranspose(2,
            (4, 4),
            (2, 2),
            padding='same',
            kernel_initializer=KERNEL_INITIALIZER,
            name='transposed_conv')

    def call(self, x):
        x = self.transposed_conv(x)
        # If we upsample the flow image, we also have to scale the flow vectors:
        x *= tf.constant(2.0)
        return x

class FlowEstimator(tf.keras.Model):
    def __init__(self, name, num_features):
        super().__init__(name=name)

        self.conv_block = SeparableConvBlock('conv_block', FLOW_NUM_BLOCKS, num_features)
        self.conv = tf.keras.layers.SeparableConv2D(2,
            (3, 3),
            padding='same',
            kernel_initializer=KERNEL_INITIALIZER,
            name='conv')

    def call(self, x, train_batch_norm=False):
        x = self.conv_block(x, training=train_batch_norm)
        x = self.conv(x)
        return x

def _warp(features, flow):
    # We need to use the negative flow here because of how tfa.image.dense_image_warp works
    # What we really want is to look up the coordinates at coord_old + flow and map them back to coorld_old
    neg_flow = -flow
    warped = tfa.image.dense_image_warp(features, neg_flow)
    warped.set_shape(features.get_shape())
    return warped

'''
This class estimates the flow between the current and previous images.
'''
class Flow(tf.keras.Model):
    def __init__(self, name, num_features):
        super().__init__(name=name)

        self.fe0 = FlowEstimator('fe0', num_features)
        self.fe1 = FlowEstimator('fe1', num_features)
        self.fe2 = FlowEstimator('fe2', num_features)
        self.fe3 = FlowEstimator('fe3', num_features)
        self.fe4 = FlowEstimator('fe4', num_features)
        self.fe5 = FlowEstimator('fe5', num_features)

        self.flow_upsample1 = FlowUpsample('flow_upsample1')
        self.flow_upsample2 = FlowUpsample('flow_upsample2')
        self.flow_upsample3 = FlowUpsample('flow_upsample3')
        self.flow_upsample4 = FlowUpsample('flow_upsample4')
        self.flow_upsample5 = FlowUpsample('flow_upsample5')

        self.correlate = tfa.layers.optical_flow.CorrelationCost(1, 4, 1, 1, 4, 'channels_last')

    # Calculates _forward_ flow. For backward flow exchange inputs.
    def call(self, inputs, train_batch_norm=False):
        current, prev = inputs

        # This corresponds to the features levels P3 to P7
        current_0, current_1, current_2, current_3, current_4 = current
        prev_0, prev_1, prev_2, prev_3, prev_4 = prev

        # Correlation on level 4
        correlation_4 = self.correlate([prev_4, current_4])
        x = tf.concat([correlation_4, current_4, prev_4], axis=-1)
        flow_4 = self.fe4(x, train_batch_norm=train_batch_norm)
        flow_4_up = self.flow_upsample4(flow_4)

        # Correlation and flow on level 3
        warped_current_3 = _warp(current_3, flow_4_up)
        correlation_3 = self.correlate([prev_3, warped_current_3])
        x = tf.concat([correlation_3, warped_current_3, prev_3, flow_4_up], axis=-1)
        flow_3 = self.fe3(x, train_batch_norm=train_batch_norm)
        flow_3_up = self.flow_upsample3(flow_3)

        # Correlation and flow on level 2
        warped_current_2 = _warp(current_2, flow_3_up)
        correlation_2 = self.correlate([prev_2, warped_current_2])
        x = tf.concat([correlation_2, warped_current_2, prev_2, flow_3_up], axis=-1)
        flow_2 = self.fe2(x, train_batch_norm=train_batch_norm)
        flow_2_up = self.flow_upsample2(flow_2)

        # Correlation and flow on level 1
        warped_current_1 = _warp(current_1, flow_2_up)
        correlation_1 = self.correlate([prev_1, warped_current_1])
        x = tf.concat([correlation_1, warped_current_1, prev_1, flow_2_up], axis=-1)
        flow_1 = self.fe1(x, train_batch_norm=train_batch_norm)
        flow_1_up = self.flow_upsample1(flow_1)

        # Correlation and flow on level 0
        warped_current_0 = _warp(current_0, flow_1_up)
        correlation_0 = self.correlate([prev_0, warped_current_0])
        x = tf.concat([correlation_0, warped_current_0, prev_0, flow_1_up], axis=-1)
        flow_0 = self.fe0(x, train_batch_norm=train_batch_norm)

        results = {}
        results['flow_0'] = flow_0
        results['flow_1'] = flow_1
        results['flow_2'] = flow_2
        results['flow_3'] = flow_3
        results['flow_4'] = flow_4
        results['flow_5'] = flow_5
        results['flow_1_up'] = flow_1_up
        results['flow_2_up'] = flow_2_up
        results['flow_3_up'] = flow_3_up
        results['flow_5_up'] = flow_5_up

        return results

class FlowWarp(tf.keras.Model):
    def __init__(self, name, num_features):
        super().__init__(name=name)
        self.channel_reduce = []
        for i in range(6):
            self.channel_reduce += [
                tf.keras.layers.SeparableConv2D(num_features,
                    (1, 1),
                    kernel_initializer=KERNEL_INITIALIZER,
                    name='conv_reduce_{}'.format(i)) ]

    def call(self, inputs, train_batch_norm=False):
        x_current, x_prev, bw_flow_dict = inputs

        # This corresponds to the feature levels P3 to P7
        bw_flows = bw_flow_dict['flow_0'], bw_flow_dict['flow_1'], bw_flow_dict['flow_2'], bw_flow_dict['flow_3'], bw_flow_dict['flow_4'], bw_flow_dict['flow_5']

        outputs = []
        for i in range(len(self.channel_reduce)):
            x_prev_warped = _warp(x_prev[i], bw_flows[i])
            x = tf.concat([x_current[i], x_prev_warped, bw_flows[i]], axis=-1)
            x = self.channel_reduce[i](x)
            outputs += [x]
        return outputs
