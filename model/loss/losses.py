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

# Losses for bounding boxes
def focal_loss(logits, labels, gamma=1.5):
    with tf.name_scope("focal_loss") as scope:
        logits = tf.nn.softmax(logits)
        focal_weight = tf.where(tf.equal(labels, 1), 1. - logits, logits)
        focal_weight = focal_weight**gamma
        losses = focal_weight * tf.keras.backend.binary_crossentropy(labels, logits)
        return tf.reduce_sum(losses)

def sparse_focal_loss(logits, labels, gamma=1.5):
    depth = logits.get_shape().as_list()[1]
    labels = tf.one_hot(labels, depth)
    return focal_loss(logits, labels, gamma=gamma)

def smooth_l1_loss(logits, labels, delta):
    diff = tf.abs(logits - labels)
    loss = tf.where(diff < delta, 0.5 * diff * diff, delta * diff - 0.5 * delta * delta)
    return loss

# Metric learning loss
def smooth_l1_diff(diff, delta):
    diff = tf.abs(diff)
    loss = tf.where(diff < delta, 0.5 * diff * diff, delta * diff - 0.5 * delta * delta)
    return loss

def get_pairwise_distances(features):
    pairwise_distances_squared = tf.math.reduce_sum(tf.math.square(features), axis=1, keepdims=True) \
        + tf.math.reduce_sum(tf.math.square(tf.transpose(features)), axis=0, keepdims=True) \
        - 2.0 * tf.matmul(features, tf.transpose(features))

    # Set small negatives (because of numeric instabilities) to zero
    pairwise_distances_squared = tf.math.maximum(pairwise_distances_squared, 0.0)

    # Explicitly set diagonals to zero
    num_entries = tf.shape(features)[0]
    mask = tf.ones_like(pairwise_distances_squared) - tf.linalg.diag(tf.ones([num_entries]))
    pairwise_distances_squared = tf.math.multiply(pairwise_distances_squared, mask)
    pairwise_distances_squared = tf.where(tf.math.greater(pairwise_distances_squared, 0.0), pairwise_distances_squared, 0.0)

    pairwise_distances = tf.math.sqrt(pairwise_distances_squared)

    return pairwise_distances

# This is Margin Loss and not Contrastive Loss as used in the ICPRAM 2020 paper.
# But that paper did not use the embedding anyways.
def metric_loss(labels, embeddings, alpha = 0.2, beta=1.2):
    with tf.device('/cpu:0'):
        distances = get_pairwise_distances(embeddings)

        labels = tf.reshape(labels, [-1, 1])
        labels = tf.tile(labels, [1, tf.shape(labels)[0]])
        adjacency = tf.math.equal(labels, tf.transpose(labels))
        adjacency_not = tf.math.logical_not(adjacency)
        mask_positives = tf.cast(adjacency, dtype=tf.dtypes.float32)
        mask_negatives = tf.cast(adjacency_not, dtype=tf.dtypes.float32)

        num_entries = 100.0 # HACK #tf.reduce_sum(mask_positives) + tf.reduce_sum(mask_negatives)

        pos_dist = distances - (beta - alpha)
        pos_dist = tf.math.maximum(pos_dist, 0.0)
        loss_positives = pos_dist * mask_positives
        loss_positives = tf.where(tf.math.greater(num_entries, 0.0),
                                  tf.reduce_sum(loss_positives) / num_entries,
                                  0.0)

        neg_dist = (beta + alpha) - distances
        neg_dist = tf.math.maximum(neg_dist, 0.0)
        loss_negatives = neg_dist * mask_negatives
        loss_negatives = tf.where(tf.math.greater(num_entries, 0.0),
                                  tf.reduce_sum(loss_negatives) / num_entries,
                                  0.0)

        return loss_positives + loss_negatives
