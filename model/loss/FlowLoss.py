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

from .losses import *
from .WeightKendall import *

class FlowLoss(tf.keras.Model):
    def __init__(self, name):
        super().__init__(name=name)

    def call(self, inputs, step):
        result, ground_truth = inputs

        flow_0 = result['flow_0']
        flow_0 = tf.reshape(flow_0, [-1])
        flow_1 = result['flow_1']
        flow_1 = tf.reshape(flow_1, [-1])
        flow_2 = result['flow_2']
        flow_2 = tf.reshape(flow_2, [-1])
        flow_3 = result['flow_3']
        flow_3 = tf.reshape(flow_3, [-1])
        flow_4 = result['flow_4']
        flow_4 = tf.reshape(flow_4, [-1])
        flow_1_up = result['flow_1_up']
        flow_1_up = tf.reshape(flow_1_up, [-1])
        flow_2_up = result['flow_2_up']
        flow_2_up = tf.reshape(flow_2_up, [-1])
        flow_3_up = result['flow_3_up']
        flow_3_up = tf.reshape(flow_3_up, [-1])
        flow_4_up = result['flow_4_up']
        flow_4_up = tf.reshape(flow_4_up, [-1])

        gt_flow_0 = ground_truth['flow_0']
        gt_flow_0 = tf.reshape(gt_flow_0, [-1])
        gt_flow_0 = tf.stop_gradient(gt_flow_0)
        gt_flow_1 = ground_truth['flow_1']
        gt_flow_1 = tf.reshape(gt_flow_1, [-1])
        gt_flow_1 = tf.stop_gradient(gt_flow_1)
        gt_flow_2 = ground_truth['flow_2']
        gt_flow_2 = tf.reshape(gt_flow_2, [-1])
        gt_flow_2 = tf.stop_gradient(gt_flow_2)
        gt_flow_3 = ground_truth['flow_3']
        gt_flow_3 = tf.reshape(gt_flow_3, [-1])
        gt_flow_3 = tf.stop_gradient(gt_flow_3)
        gt_flow_4 = ground_truth['flow_4']
        gt_flow_4 = tf.reshape(gt_flow_4, [-1])
        gt_flow_4 = tf.stop_gradient(gt_flow_4)

        loss_0 = tf.norm(flow_0 - gt_flow_0) + tf.norm(flow_1_up - gt_flow_0)
        loss_1 = tf.norm(flow_1 - gt_flow_1) + tf.norm(flow_2_up - gt_flow_1)
        loss_2 = tf.norm(flow_2 - gt_flow_2) + tf.norm(flow_3_up - gt_flow_2)
        loss_3 = tf.norm(flow_3 - gt_flow_3) + tf.norm(flow_4_up - gt_flow_3)
        loss_4 = tf.norm(flow_4 - gt_flow_4)

        loss = 0.05 * loss_0 + 0.1 * loss_1 + 0.2 * loss_2 + 0.8 * loss_3 + 3.2 * loss_4

        tf.summary.scalar('flow_loss', loss, step)
        return loss
