# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Provides flags that are common to scripts.

Common flags from train/eval/vis/export_model.py are collected in this script.
"""
import collections
import copy
import json

import tensorflow as tf

flags = tf.app.flags

# Flags for input preprocessing.

flags.DEFINE_integer('min_resize_value', None,
                     'Desired size of the smaller image side.')

flags.DEFINE_integer('max_resize_value', None,
                     'Maximum allowed size of the larger image side.')

flags.DEFINE_integer('resize_factor', None,
                     'Resized dimensions are multiple of factor plus one.')

# Model dependent flags.

flags.DEFINE_integer('logits_kernel_size', 1,
                     'The kernel size for the convolutional kernel that '
                     'generates logits.')

flags.DEFINE_multi_float('image_pyramid', None,
                         'Input scales for multi-scale feature extraction.')

flags.DEFINE_boolean('add_image_level_feature', True,
                     'Add image level feature.')

flags.DEFINE_boolean('aspp_with_batch_norm', True,
                     'Use batch norm parameters for ASPP or not.')

flags.DEFINE_boolean('aspp_with_separable_conv', False,
                     'Use separable convolution for ASPP or not.')

flags.DEFINE_multi_integer('multi_grid', None,
                           'Employ a hierarchy of atrous rates for ResNet.')

flags.DEFINE_float('depth_multiplier', 1.0,
                   'Multiplier for the depth (number of channels) for all '
                   'convolution ops used in MobileNet.')

flags.DEFINE_integer('decoder_output_stride', 1,
                     'The ratio of input to output spatial resolution when '
                     'employing decoder to refine segmentation results.')

flags.DEFINE_boolean('mat_decoder_use_separable_conv', False,
                     'Employ separable convolution for decoder or not.')
flags.DEFINE_boolean('seg_decoder_use_separable_conv', False,
                     'Employ separable convolution for decoder or not.')

flags.DEFINE_enum('merge_method', 'max', ['max', 'avg'],
                  'Scheme to merge multi scale features.')

FLAGS = flags.FLAGS

# Constants

# Perform semantic segmentation predictions.
OUTPUT_TYPE = 'semantic'
OUTPUT_TYPE_PYRAMID = 'semantic_final'

# Semantic segmentation item names.
LABELS_CLASS = 'labels_class'
COMP_IMAGE = 'comp_image'
COMP_HEIGHT = 'com_height'
COMP_WIDTH = 'comp_width'
COMP_NAME = 'comp_name'
COMP_ORIGINAL_IMAGE = 'comp_original_image'
COMP_ORIGINAL_TRIMAP = 'comp_original_trimap'
COMP_TRIMAP = 'comp_trimap'
COMP_ALPHA = 'comp_alpha'
COMP_SEG = 'comp_seg'
COMP_PRED_MAT = 'comp_pred_mat'
COMP_PRED_SEG = 'comp_pred_seg'

PATCH_IMAGE = 'patch_image'
PATCH_LOC = 'patch_location'
PATCH_HEIGHT = 'patch_height'
PATCH_WIDTH = 'patch_width'
PATCH_NAME = 'patch_name'
PATCH_ORIGINAL_IMAGE = 'patch_original_image'
PATCH_ORIGINAL_TRIMAP = 'patch_original_trimap'
PATCH_TRIMAP = 'patch_trimap'
PATCH_ALPHA = 'patch_alpha'
PATCH_SEG = 'patch_seg'
PATCH_FG = 'patch_fg'
PATCH_BG = 'patch_bg'
PATCH_PRED_MAT = 'patch_pred_mat'
PATCH_PRED_SEG = 'patch_pred_seg'


# Test set name.
TEST_SET = 'test'

class ModelOptions(
    collections.namedtuple('ModelOptions', [
        'num_classes',
        'crop_size',
        'atrous_rates',
        'output_stride',
        'merge_method',
        'add_image_level_feature',
        'aspp_with_batch_norm',
        'aspp_with_separable_conv',
        'multi_grid',
        'decoder_output_stride',
        'mat_decoder_use_separable_conv',
        'seg_decoder_use_separable_conv',
        'logits_kernel_size',
        'model_variant'
    ])):
  """Immutable class to hold model options."""

  __slots__ = ()


  def __new__(cls,
              num_classes=1,
              crop_size=None,
              atrous_rates=None,
              output_stride=8,
              model_variant='mobilenet_v2'):
    """Constructor to set default values.

    Args:
      outputs_to_num_classes: A dictionary from output type to the number of
        classes. For example, for the task of semantic segmentation with 21
        semantic classes, we would have outputs_to_num_classes['semantic'] = 21.
      crop_size: A tuple [crop_height, crop_width].
      atrous_rates: A list of atrous convolution rates for ASPP.
      output_stride: The ratio of input to output spatial resolution.

    Returns:
      A new ModelOptions instance.
    """
    return super(ModelOptions, cls).__new__(
        cls, num_classes, crop_size, atrous_rates, output_stride,
        FLAGS.merge_method, FLAGS.add_image_level_feature,
        FLAGS.aspp_with_batch_norm, FLAGS.aspp_with_separable_conv,
        FLAGS.multi_grid, FLAGS.decoder_output_stride,
        FLAGS.mat_decoder_use_separable_conv, 
        FLAGS.seg_decoder_use_separable_conv, 
        FLAGS.logits_kernel_size,
        model_variant)

  def __deepcopy__(self, memo):
      """
      Returns a copy of the model

      Args:
          self: (todo): write your description
          memo: (dict): write your description
      """
    return ModelOptions(copy.deepcopy(self.num_classes),
                        self.crop_size,
                        self.atrous_rates,
                        self.output_stride,
                        self.model_variant)
