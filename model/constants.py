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
import collections
from thirdparty.online_norm import *

L2_REGULARIZER_WEIGHT=1e-4

def Normalization():
    return OnlineNorm(alpha_fwd=0.999, alpha_bkw=0.99)

KERNEL_INITIALIZER = tf.keras.initializers.VarianceScaling(2.0, mode='fan_out', distribution='normal')

# This line has to match the C++ code!
BOXES_PER_POS = 20

##### The following is for EfficientDet #####

BlockArgs = collections.namedtuple('BlockArgs', [
    'kernel_size', 'num_repeat', 'input_filters', 'output_filters',
    'expand_ratio', 'id_skip', 'strides', 'se_ratio'
])

BLOCKS_ARGS = [
    BlockArgs(kernel_size=3, num_repeat=1, input_filters=32, output_filters=16,
              expand_ratio=1, id_skip=True, strides=[1, 1], se_ratio=0.25),
    BlockArgs(kernel_size=3, num_repeat=2, input_filters=16, output_filters=24,
              expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
    BlockArgs(kernel_size=5, num_repeat=2, input_filters=24, output_filters=40,
              expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
    BlockArgs(kernel_size=3, num_repeat=3, input_filters=40, output_filters=80,
              expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
    BlockArgs(kernel_size=5, num_repeat=3, input_filters=80, output_filters=112,
              expand_ratio=6, id_skip=True, strides=[1, 1], se_ratio=0.25),
    BlockArgs(kernel_size=5, num_repeat=4, input_filters=112, output_filters=192,
              expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
    BlockArgs(kernel_size=3, num_repeat=1, input_filters=192, output_filters=320,
              expand_ratio=6, id_skip=True, strides=[1, 1], se_ratio=0.25)
]

BackboneArgs = collections.namedtuple('BackboneArgs', [
    'width_coefficient', 'depth_coefficient', 'dropout_rate', 'drop_connect_rate', 'depth_divisor'
])


WIDTH_COEFFICIENT = 1.2 # For D = 3
DEPTH_COEFFICIENT = 1.4 # For D = 3
DROPOUT_RATE = 0.3 # For D = 3
DROP_CONNECT_RATE = 0.2
DEPTH_DIVISOR = 8

BACKBONE_ARGS = BackboneArgs(width_coefficient=WIDTH_COEFFICIENT,
                             depth_coefficient=DEPTH_COEFFICIENT,
                             dropout_rate=DROPOUT_RATE,
                             drop_connect_rate=DROP_CONNECT_RATE,
                             depth_divisor=DEPTH_DIVISOR)

# BiFPN parameters
D = 3
BIFPN_NUM_FEATURES = int(64 * 1.35 ** D)
BIFPN_NUM_BLOCKS = int(3 + D)

HEADS_NUM_BLOCKS = int(3 + D / 3.0)
FLOW_NUM_BLOCKS = int(2 + D / 3.0)
