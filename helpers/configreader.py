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
