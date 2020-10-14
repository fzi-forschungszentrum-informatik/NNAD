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
from .losses import *
from .WeightKendall import *

class EmbeddingLoss(tf.keras.Model):
    def __init__(self, name, config):
        super().__init__(name=name)
        self.box_embedding_len = config['box_embedding_len']

        self.weight = WeightKendall('weight_embedding')

    def _embedding_loss(self, embeddings, ids):
        MAX_NUM_SAMPLES = 5000.0
        EPS = 1e-5

        assignment_losses = []
        n = len(ids)
        for i in range(n):
            batch_box_ids = tf.reshape(ids[i], [-1])
            batch_embeddings = tf.reshape(embeddings[i], [-1, self.box_embedding_len])

            mask = tf.math.greater(batch_box_ids, 0)
            num_mask = tf.reduce_sum(tf.cast(mask, tf.float32))
            samples = tf.random.normal(tf.shape(mask), num_mask / MAX_NUM_SAMPLES + EPS)
            mask = tf.logical_and(mask, tf.less(samples, 1.0))
            masked_embeddings = tf.boolean_mask(batch_embeddings, mask)
            masked_box_ids = tf.boolean_mask(batch_box_ids, mask)
            batch_assignment_loss = metric_loss(masked_box_ids, masked_embeddings)
            batch_assignment_loss = tf.reduce_sum(batch_assignment_loss) / MAX_NUM_SAMPLES
            assignment_losses += [batch_assignment_loss]
        assignment_loss = tf.stack(assignment_losses)
        return tf.reduce_mean(assignment_loss)

    def call(self, inputs, step):
        results, ground_truth = inputs
        embeddings = results['bb_targets_embedding']
        embeddings = tf.unstack(embeddings, axis=0)
        ids = tf.unstack(ground_truth['bb_targets_id'], axis=0)

        embedding_loss = self._embedding_loss(embeddings, ids)

        # Apply some sensible scaling before loss weighting
        embedding_loss *= 100.0

        embedding_loss = self.weight(embedding_loss, step)

        tf.summary.scalar('embedding_loss', embedding_loss, step)

        return embedding_loss
