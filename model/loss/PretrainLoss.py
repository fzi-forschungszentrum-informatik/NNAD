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

class PretrainLoss(tf.keras.Model):
    def __init__(self, name, config):
        super().__init__(name=name)

        self.num_label_classes = config['num_pretrain_classes']

    def call(self, inputs, step):
        result, ground_truth = inputs
        labels = result
        gt_labels = ground_truth['cls']

        shape = tf.shape(labels)
        labels = tf.reshape(labels, [-1, self.num_label_classes])
        gt_labels = tf.cast(gt_labels, tf.int32)
        gt_labels = tf.reshape(gt_labels, [-1])

        gt_labels = tf.stop_gradient(gt_labels)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=labels, labels=gt_labels)
        loss = tf.reduce_sum(loss)

        correct_prediction = tf.equal(tf.cast(tf.argmax(labels, 1), tf.int32), gt_labels)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        tf.summary.scalar('loss', loss, step)
        tf.summary.scalar('accuracy', accuracy, step)

        return loss
