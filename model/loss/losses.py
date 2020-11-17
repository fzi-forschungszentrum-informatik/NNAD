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

# This is our Weighted Margin Loss
def metric_loss(labels, embeddings, alpha = 0.2, beta=1.2):
    distances = get_pairwise_distances(embeddings)

    labels = tf.reshape(labels, [-1, 1])
    labels = tf.tile(labels, [1, tf.shape(labels)[0]])
    adjacency = tf.math.equal(labels, tf.transpose(labels))
    adjacency_not = tf.math.logical_not(adjacency)
    mask_positives = tf.cast(adjacency, dtype=tf.dtypes.float32)
    mask_negatives = tf.cast(adjacency_not, dtype=tf.dtypes.float32)

    pos_dist = distances - (beta - alpha)
    pos_dist = tf.math.maximum(pos_dist, 0.0)
    loss_positives = pos_dist * mask_positives
    loss_positives = tf.reduce_sum(loss_positives)

    neg_dist = (beta + alpha) - distances
    L = 0.2
    N = tf.cast(tf.shape(embeddings)[-1], tf.float32)
    neg_dist = neg_dist * L / (L + tf.math.exp(-N * tf.math.pow(distances - tf.math.sqrt(2.0), 2.0)))
    neg_dist = tf.math.maximum(neg_dist, 0.0)
    loss_negatives = neg_dist * mask_negatives
    loss_negatives = tf.reduce_sum(loss_negatives)

    return loss_positives + loss_negatives

## MonoDepth2 losses

def ssim(x, y):
    # constants
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # padding of images
    x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]] 'reflect')
    y = tf.pad(y, [[0, 0], [1, 1], [1, 1], [0, 0]] 'reflect')

    # local mean and variance
    mu_x = tf.nn.avg_pool2d(x, 3, 1, 'valid')
    mu_y = tf.nn.avg_pool2d(y, 3, 1, 'valid')

    avg_xx = tf.nn.avg_pool2d(x * x, 3, 1, 'valid')
    avg_xy = tf.nn.avg_pool2d(x * y, 3, 1, 'valid')
    avg_yy = tf.nn.avg_pool2d(y * y, 3, 1, 'valid')

    sigma_x  = avg_xx - mu_x * mu_x
    sigma_y  = avg_yy - mu_y * mu_y
    sigma_xy = avg_xy - mu_x * mu_y

    # loss
    SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x * mu_x + mu_y * mu_y + C1) * (sigma_x + sigma_y + C2)

    return tf.clip_by_value((1.0 - SSIM_n / SSIM_d) / 2.0, 0.0, 1.0)

def photometric_loss(x, y):
    return 0.85 * ssim(x, y) + 0.15 * tf.math.abs(x - y)

def smoothness_loss(img, disp):
    EPS = 1e-7

    # normalize disparity
    mean_disp = tf.math.reduce_mean(disp, [1, 2])
    norm_disp = disp / (mean_disp + EPS)

    # calculate gradients of disparity and image
    grad_disp_x = tf.math.abs(norm_disp[:, :, :-1, :] - norm_disp[:, :, 1:, :])
    grad_disp_y = tf.math.abs(norm_disp[:, :-1, :, :] - norm_disp[:, 1:, :, :])

    grad_img_x = tf.math.abs(img[:, :, :-1, :] - img[:, :, 1:, :])
    grad_img_y = tf.math.abs(img[:, :-1, :, :] - img[:, 1:, :, :])

    # Weight disparity gradients
    grad_disp_x *= tf.math.exp(-tf.math.reduce_mean(grad_img_x, -1))
    grad_disp_y *= tf.math.exp(-tf.math.reduce_mean(grad_img_y, -1))

    loss = grad_disp_x + grad_disp_y
    loss *= 1e-3
    return grad_disp_x + grad_disp_y

# The "mask" input must be the logits (before sigmoid)!
def mask_regularization_loss(mask):
    return 0.2 * itf.nn.sigmoid_cross_entropy_with_logits(tf.ones_like(mask), mask)

