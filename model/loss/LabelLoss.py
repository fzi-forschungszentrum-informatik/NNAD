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
from .WeightKendall import *

class LabelLoss(tf.keras.Model):
    def __init__(self, name, config):
        super().__init__(name=name)

        self.num_label_classes = config['num_label_classes']

        self.weight = WeightKendall('weight_label')

    def call(self, inputs, step):
        results, ground_truth = inputs
        labels = results['pixelwise_labels']
        gt_labels = ground_truth['pixelwise_labels']

        shape = tf.shape(labels)
        labels = tf.reshape(labels, [-1, self.num_label_classes])
        gt_labels = tf.cast(gt_labels, tf.int32)
        gt_labels = tf.reshape(gt_labels, [-1])
        label_mask = tf.not_equal(gt_labels, tf.constant([-1])) # -1 is the ignore label
        masked_labels = tf.boolean_mask(labels, label_mask)
        masked_gt_labels = tf.boolean_mask(gt_labels, label_mask)

        masked_gt_labels = tf.stop_gradient(masked_gt_labels)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=masked_labels, labels=masked_gt_labels)
        loss = tf.reduce_sum(loss) / (tf.cast(shape[1], tf.float32) * tf.cast(shape[2], tf.float32))

        correct_prediction = tf.equal(tf.cast(tf.argmax(masked_labels, 1), tf.int32), masked_gt_labels)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # Apply some sensible scaling before loss weighting
        loss *= 50.0

        loss = self.weight(loss, step)

        tf.summary.scalar('loss', loss, step)
        tf.summary.scalar('accuracy', accuracy, step)

        return loss
