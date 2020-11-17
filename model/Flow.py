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

def _warp(features, flow):
    # We need to use the negative flow here because of how tfa.image.dense_image_warp works
    # What we really want is to look up the coordinates at coord_old + flow and map them back to coorld_old
    neg_flow = -flow
    warped = tfa.image.dense_image_warp(features, neg_flow)
    warped.set_shape(features.get_shape())
    return warped

class FlowModule(tf.keras.Model):
    def __init__(self, name):
        super().__init__(name=name, is_lowest_level):

        self.is_lowest_level = is_lowest_level
        if not is_lowest_level:
            self.upsample_flow = FlowUpsample('flow_upsample')
            self.upsample_features = tf.keras.layers.Conv2DTranspose(2,
                (4, 4),
                (2, 2),
                padding='same',
                kernel_initializer=KERNEL_INITIALIZER,
                name='feature_upsample')

        self.correlate = tfa.layers.optical_flow.CorrelationCost(1, 4, 1, 1, 4, 'channels_last')

        self.conv_block = SeparableConvBlock('conv_block', FLOW_NUM_BLOCKS, BIFPN_NUM_FEATURES)
        self.conv_final_flow = tf.keras.layers.SeparableConv2D(2,
            (3, 3),
            padding='same',
            kernel_initializer=KERNEL_INITIALIZER,
            name='final_flow')
        self.conv_final_mask = tf.keras.layers.SeparableConv2D(1,
            (3, 3),
            padding='same',
            kernel_initializer=KERNEL_INITIALIZER,
            name='final_flow')

    # Calculates _forward_ flow. For backward flow exchange inputs.
    def call(self, inputs, training=False):
        to_concat = []
        if not self.is_lowest_level:
            small_features, small_flow, small_mask, prev, current = inputs
            upsampled_features = self.upsample_features(small_features)
            upsampled_flow = self.upsample_flow(small_flow)
            _, h, w, _ = upsampled_flow.get_shape().as_list()
            upsampled_mask = tf.image.resize(upsampled_mask, [h, w])
            to_concat += [upsampled_features, upsampled_flow]
            current = _warp(current, upsampled_flow) * tf.nn.sigmoid(upsampled_mask)
        else:
            prev, current = inputs
            upsampled_features = None
            upsampled_flow = None
        corr = self.correlate([prev, current])
        to_concat += [corr, prev]

        out_features = self.conv_block(tf.concat(to_concat, axis=-1), training=training)
        out_flow = self.conv_final_flow(out_features)
        out_mask = self.conv_final_mask(out_features)

        return out_features, out_flow, out_mask, upsampled_flow

'''
This class estimates the flow between the current and previous images.
'''
class Flow(tf.keras.Model):
    def __init__(self, name):
        super().__init__(name=name)

        self.fe0 = FlowModule('fe0', False)
        self.fe1 = FlowModule('fe1', False)
        self.fe2 = FlowModule('fe2', False)
        self.fe3 = FlowModule('fe3', False)
        self.fe4 = FlowModule('fe4', False)
        self.fe5 = FlowModule('fe5', True)

    # Calculates _forward_ flow. For backward flow exchange inputs.
    def call(self, inputs, training=False):
        current, prev = inputs

        # This corresponds to the features levels P2 to P7
        current_0, current_1, current_2, current_3, current_4, current_5 = current
        prev_0, prev_1, prev_2, prev_3, prev_4, prev_5 = prev

        features_5, flow_5, mask_5, _, _ = fe5([prev5, current_5], training=training)
        features_4, flow_4, mask_4, flow_5_up = fe4([features_5, flow_5, mask_5, prev4, current_4], training=training)
        features_3, flow_3, mask_3, flow_4_up = fe3([features_4, flow_4, mask_4, prev3, current_3], training=training)
        features_2, flow_2, mask_2, flow_3_up = fe2([features_3, flow_3, mask_3, prev2, current_2], training=training)
        features_1, flow_1, mask_1, flow_2_up = fe1([features_2, flow_2, mask_2, prev1, current_1], training=training)
        features_0, flow_0, mask_0, flow_1_up = fe0([features_1, flow_1, mask_1, prev0, current_0], training=training)

        results = {}
        results['flow_0'] = flow_0
        results['flow_1'] = flow_1
        results['flow_2'] = flow_2
        results['flow_3'] = flow_3
        results['flow_4'] = flow_4
        results['flow_5'] = flow_5
        results['mask_0'] = mask_0
        results['mask_1'] = mask_1
        results['mask_2'] = mask_2
        results['mask_3'] = mask_3
        results['mask_4'] = mask_4
        results['mask_5'] = mask_5
        results['flow_1_up'] = flow_1_up
        results['flow_2_up'] = flow_2_up
        results['flow_3_up'] = flow_3_up
        results['flow_5_up'] = flow_5_up

        return results

class FlowWarp(tf.keras.Model):
    def __init__(self, name):
        super().__init__(name=name)
        self.channel_reduce = []
        for i in range(6):
            self.channel_reduce += [
                tf.keras.layers.SeparableConv2D(BIFPN_NUM_FEATURES,
                    (1, 1),
                    kernel_initializer=KERNEL_INITIALIZER,
                    name='conv_reduce_{}'.format(i)) ]

    def call(self, inputs, train_batch_norm=False):
        x_current, x_prev, bw_flow_dict = inputs

        # This corresponds to the feature levels P3 to P7
        bw_flows = bw_flow_dict['flow_0'], bw_flow_dict['flow_1'], bw_flow_dict['flow_2'], bw_flow_dict['flow_3'], bw_flow_dict['flow_4'], bw_flow_dict['flow_5']
        bw_masks = bw_flow_dict['mask_0'], bw_flow_dict['mask_1'], bw_flow_dict['mask_2'], bw_flow_dict['mask_3'], bw_flow_dict['mask_4'], bw_flow_dict['mask_5']
        outputs = []
        for i in range(len(self.channel_reduce)):
            mask = tf.nn.sigmoid(bw_masks[i])
            flow = bw_flows[i]
            mask = tf.stop_gradient(mask)
            flow = tf.stop_gradient(flow)
            x_prev_warped = _warp(x_prev[i], flow) * mask
            x = tf.concat([x_current[i], x_prev_warped, flow, mask], axis=-1)
            x = self.channel_reduce[i](x)
            outputs += [x]
        return outputs
