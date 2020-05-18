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

import argparse
import sys
import yaml

import tensorflow as tf

def get_config():
    argv = sys.argv
    arg_parser = argparse.ArgumentParser(description='NNAD (Neural Networks for Automated Driving)')
    arg_parser.add_argument('config', help='Path to the config file')
    del argv[0]
    args = arg_parser.parse_args(argv)
    config_path = args.config

    # Config handling
    with open(config_path, 'r') as stream:
        config = yaml.safe_load(stream)
    return config, config_path

def get_learning_rate_fn(config, global_step):
    lr_steps = config['lr_steps']
    lr_values = config['lr_values']
    last_step = 0
    for step in lr_steps:
        assert step > last_step
        last_step = step
    assert len(lr_steps) == len(lr_values)
    lr_steps = [tf.constant(v) for v in lr_steps]
    lr_values = [tf.constant(v) for v in lr_values]

    @tf.function
    def learning_rate_fn():
        lr = tf.constant(0.0)
        for i in range(len(lr_steps)):
            if global_step < lr_steps and lr == tf.constant(0.0):
                lr = lr_values[i]
        return lr

    return learning_rate_fn, lr_steps[-1]

