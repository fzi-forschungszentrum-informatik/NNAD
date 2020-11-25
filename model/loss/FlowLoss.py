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

from .losses import *
from model.constants import *
from .WeightKendall import *

class FlowLoss(tf.keras.Model):
    def __init__(self, name):
        super().__init__(name=name)

        self.weight_supervised = WeightKendall('weight_supervised')
        self.weight_photo = WeightKendall('weight_foto')
        self.weight_mask = WeightKendall('weight_mask')

    def call(self, inputs, step):
        result, images, ground_truth = inputs

        flows = [result['flow_0'], result['flow_1'], result['flow_2'],
                 result['flow_3'], result['flow_4'], result['flow_5']]

        flows_up = [result['flow_1_up'], result['flow_2_up'], result['flow_3_up'],
                    result['flow_4_up'], result['flow_5_up']]

        masks = [result['fmask_0'], result['fmask_1'], result['fmask_2'],
                 result['fmask_3'], result['fmask_4'], result['fmask_5']]

        current_img=images['left']
        h = tf.shape(current_img)[1]
        w = tf.shape(current_img)[2]
        prev_img=images['prev_left']

        gt_flow = ground_truth['flow']
        gt_flow *= FLOW_FACTOR # We train larger flow for regression.
        gt_flow = tf.stop_gradient(gt_flow)

        gt_mask = ground_truth['flow_mask']
        gt_mask = tf.stop_gradient(gt_mask)

        photo_loss = tf.constant(0.0, tf.float32)
        supervised_loss = tf.constant(0.0, tf.float32)
        factor = 1.0
        for i in range(len(flows)):
            flow = flows[i]
            mask = masks[i]

            smask = tf.nn.sigmoid(mask)
            h_flow = tf.shape(flow)[1]
            flow = tf.image.resize(flow, [h, w]) * tf.cast(h, tf.float32) / tf.cast(h_flow, tf.float32)
            mask = tf.image.resize(mask, [h, w])
            smask = tf.image.resize(smask, [h, w])
            supervised_loss += factor * tf.reduce_sum(smask * smooth_l1_diff(flow - gt_flow, delta=1.0))

            warped_img = tfa.image.dense_image_warp(current_img, -flow / FLOW_FACTOR)
            warped_img.set_shape(prev_img.get_shape())
            photo_loss += factor * tf.reduce_sum(smask * photometric_loss(prev_img, warped_img))

            gt_mask=None # hack
            mloss = factor * tf.reduce_sum(mask_regularization_loss(mask, gt_mask))
            supervised_loss += mloss
            photo_loss += mloss
            factor *= 0.25

        factor = 0.5
        for i in range(len(flows_up)):
            flow = flows_up[i]
            mask = masks[i + 1]

            smask = tf.nn.sigmoid(mask)
            h_flow = tf.shape(flow)[1]
            flow = tf.image.resize(flow, [h, w]) * tf.cast(h, tf.float32) / tf.cast(h_flow, tf.float32)
            mask = tf.image.resize(mask, [h, w])
            smask = tf.image.resize(smask, [h, w])
            supervised_loss += factor * tf.reduce_sum(smask * smooth_l1_diff(flow - gt_flow, delta=1.0))

            warped_img = tfa.image.dense_image_warp(current_img, -flow / FLOW_FACTOR)
            warped_img.set_shape(prev_img.get_shape())
            photo_loss += factor * tf.reduce_sum(smask * photometric_loss(prev_img, warped_img))

            gt_mask=None # hack
            mloss = factor * tf.reduce_sum(mask_regularization_loss(mask, gt_mask))
            supervised_loss += mloss
            photo_loss += mloss
            factor *= 0.25

        # Some sensible scaling
        supervised_loss *= 1e-6
        photo_loss *= 1e-5

        supervised_loss = self.weight_supervised(supervised_loss, step)
        photo_loss = self.weight_photo(photo_loss, step)

        tf.summary.scalar('flow_loss_supervised', supervised_loss, step)
        tf.summary.scalar('flow_loss_photometric', photo_loss, step)
        return supervised_loss + photo_loss
