# NNAD - Neural Networks for Automated Driving

NNAD is a collection of scripts to train neural networks for automated driving.
The trained network accepts camera images and outputs a semantic segmentation and detected objects.
It is used on the research vehicles of ISPE-MPS at FZI Research Center for Information Technology.

These scripts are licenced under the GPL-3.

## Requirements
The following packages are required to use this software:

- CMake
- Boost
- OpenCV 3 or later
- JsonCpp
- yaml-cpp
- Python 3
- Tensorflow 2
- Tensorflow Addons
- Numpy
- Scipy
- Pillow
- PyYAML

## Usage
Please run `data/setup.sh` to compile the preprocessing and data loading code.
Then you can create a configuration file for the training based on `example_config.yaml`

The training and inference itself works as follows:

- Optionally run `pretrain.py` to pretrain the neural network.
- Run `train_single_frame_model.py` to train the neural network.
- Run `write_single_frame_model.py` to export the model as a "saved_model".
- Run `run_single_frame_model.py` for inference. It imports a "saved_model",
  so make sure to run the previous step before.

If you want to also train the multi-frame model, then first train the optical flow estimation
with `train_flow_model.py` and then the multi-frame heads with `train_multi_frame_model.py`.
This model can be used with `write_multi_frame_model.py` and `run_multi_frame_model`.

The hyperparameters are tuned for a training with 8 GPUs.
If you use a different number of GPUs or a different batch size you might have to adjust them.
