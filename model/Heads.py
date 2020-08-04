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
from .LabelBranch import *
from .BoxBranch import *

class Heads(tf.keras.Model):
    def __init__(self, name, config, box_delta_regression=False):
        super().__init__(name=name)

        if config['train_labels']:
            self.label_branch = LabelBranch('label_branch', config)
        else:
            self.label_branch = None

        if config['train_boundingboxes']:
            self.box_branch = BoxBranch('box_branch', config, box_delta_regression)
        else:
            self.box_branch = None

        self.box_delta_regression = box_delta_regression

    def call(self, x, train_batch_norm=False):
        results = {}

        if self.label_branch:
            labels = self.label_branch(x, training=train_batch_norm)
            results['pixelwise_labels'] = labels

        if self.box_branch:
            box_results = self.box_branch(x, train_batch_norm=train_batch_norm)
            if self.box_delta_regression:
                bb_targets, cls_targets, obj_targets, embedding_targets, delta_targets = box_results
            else:
                bb_targets, cls_targets, obj_targets, embedding_targets = box_results
            results['bb_targets_offset'] = bb_targets
            results['bb_targets_cls'] = cls_targets
            results['bb_targets_objectness'] = obj_targets
            results['bb_targets_embedding'] = embedding_targets
            if self.box_delta_regression:
                results['bb_targets_delta'] = delta_targets

        return results
