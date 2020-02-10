#!/usr/bin/env python3

import json
import os
import tensorflow as tf
from PIL import Image

def _bytes_feature(value):
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy()
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def _float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _convert(in_filename, out_filename):
  with tf.io.TFRecordWriter(out_filename) as writer:
    with open(in_filename) as fp:
      line = fp.readline()
      while line:
        gt = json.loads(line)
        id = gt['ID']
        if id == '273275,513f0000dc988092': # skip broken file
          line = fp.readline()
          continue
        image_file = os.path.join('Images', id + '.jpg')
        with open(image_file, 'rb') as imgfp:
          image = imgfp.read()
        pil_im=Image.open(image_file)
        width, height = pil_im.size
        xmins = []
        xmaxs = []
        ymins = []
        ymaxs = []
        labels = []
        areas = []
        crowds = []
        for entry in gt['gtboxes']:
          if entry['tag'] != 'person':
            continue
          b = entry['vbox']
          xmin = b[0]
          ymin = b[1]
          xmax = xmin + b[2]
          ymax = ymin + b[3]
          xmins += [float(xmin) / width]
          xmaxs += [float(xmax) / width]
          ymins += [float(ymin) / height]
          ymaxs += [float(ymax) / height]
          labels += [1]
          areas += [(xmax - xmin) * (ymax - ymin)]
          crowds += [False]
        example = tf.train.Example(features = tf.train.Features(feature = {
                    'image/encoded': _bytes_feature([image]),
                    'image/source_id': _bytes_feature([str.encode(id)]),
                    'image/height': _int64_feature([height]),
                    'image/width': _int64_feature([width]),
                    'image/object/bbox/xmin': _float_feature(xmins),
                    'image/object/bbox/xmax': _float_feature(xmaxs),
                    'image/object/bbox/ymin': _float_feature(ymins),
                    'image/object/bbox/ymax': _float_feature(ymaxs),
                    'image/object/class/label': _int64_feature(labels),
                    'image/object/area': _float_feature(areas),
                    'image/object/is_crowd': _int64_feature(crowds)
        }))
        writer.write(example.SerializeToString())
        line = fp.readline()

_convert('annotation_train.odgt', 'train.tfrecords')
_convert('annotation_val.odgt', 'val.tfrecords')

