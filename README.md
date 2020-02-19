# FeatureNMS: Non-Maximum Suppression by Learning Feature Embeddings

This is the code for the paper "FeatureNMS: Non-Maximum Suppression by Learning
Feature Embeddings" (https://arxiv.org/abs/2002.07662).

*PLEASE NOTE*: This code is far from production-quality. It just serves as a
reference to reproduce the results from the paper.

Follow these steps to train the object detector:

- Download the CrowdHuman dataset from https://www.crowdhuman.org/
- Run `crowdhuman_to_tfrecord.py` to convert the dataset to a TFRecords dataset
- Clone the training scripts
  - `git clone https://github.com/tensorflow/models/`
  - `cd models`
  - `git checkout 558bab5deafb2306ff14ce794e84629e0d433173`
  - `git am ../0001-Fixes-to-the-base-model.patch ../0002-Implement-code-for-paper.patch`
- Create a config file `myconfig.yaml`:

```
type: 'retinanet'
model_dir: '/path/to/checkpoints/'

train:
  train_file_pattern: '/path/to/train.tfrecords'
  checkpoint:
    path: '/path/to/pretrain_checkpoint/'

eval:
  eval_file_pattern: '/path/to/val.tfrecords'
```

- Run the training:
  `python official/vision/detection/ main.py --config_file myconfig.yaml`
- Run the evaluation on the CPU (GPU support seems broken for the moment):
  `CUDA_VISIBLE_DEVICES="" python official/vision/detection/ main.py --config_file myconfig.yaml --mode eval`
