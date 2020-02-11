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

class WeightKendall(tf.keras.layers.Layer):
    def __init__(self, name):
        super().__init__(name=name)
        self.factor = self.add_weight("s", [], initializer=tf.ones_initializer)

    def call(self, loss, step):
        tf.summary.scalar('s', self.factor, step)
        # This does not match our ICPRAM 2020 paper but the original paper. It seems that there was an error
        # in the ICPRAM 2020 paper.
        return loss / (2.0 * self.factor * self.factor) + tf.math.log(self.factor)

