import tensorflow as tf
from conmat.core import feature_extractor

slim = tf.contrib.slim
_LOGITS_SCOPE_NAME = 'logits'
_MERGED_LOGITS_SCOPE = 'merged_logits'
_IMAGE_POOLING_SCOPE = 'image_pooling'
_ASPP_SCOPE = 'aspp'
_CONCAT_PROJECTION_SCOPE = 'concat_projection'
_DECODER_SCOPE = 'decoder'


def get_extra_layer_scopes(last_layers_contain_logits_only=False):
  """Gets the scopes for extra layers.

  Args:
    last_layers_contain_logits_only: Boolean, True if only consider logits as
    the last layer (i.e., exclude ASPP module, decoder module and so on)

  Returns:
    A list of scopes for extra layers.
  """
  if last_layers_contain_logits_only:
    return [_LOGITS_SCOPE_NAME]
  else:
    return [
        _LOGITS_SCOPE_NAME,
        _IMAGE_POOLING_SCOPE,
        _ASPP_SCOPE,
        _CONCAT_PROJECTION_SCOPE,
        _DECODER_SCOPE,
    ]


def predict_labels_conmat(
        comp_images,
        patch_images,
        comp_options,
        patch_options,
        mode,
        color_refine_one_time=True,
        color_refine_layer_id=1,
        mat_refine_one_time=False,
        mat_refine_layer_id=0,
        add_trimap=False,
        decoder_depth=128,
        kernel_size=3,
        image_pyramid=None,
        model_parallelism=False,
        branch=(True, True, True, True)):
  """Predicts segmentation labels.

  Args:
    images: A tensor of size [batch, height, width, channels].
    model_options: A ModelOptions instance to configure models.
    image_pyramid: Input image scales for multi-scale feature extraction.

  Returns:
    A dictionary with keys specifying the output_type (e.g., semantic
      prediction) and values storing Tensors representing predictions (argmax
      over channels). Each prediction has size [batch, height, width].
  """
  patch_mat_scales, patch_color_scales= multi_scale_logits_conmat(
      comp_images,
      patch_images,
      comp_options,
      patch_options,
      mode,
      image_pyramid,
      color_refine_one_time=color_refine_one_time,
      color_refine_layer_id=color_refine_layer_id,
      mat_refine_one_time=mat_refine_one_time,
      mat_refine_layer_id=mat_refine_layer_id,
      add_trimap=add_trimap,
      decoder_depth=decoder_depth,
      kernel_size=kernel_size,
      is_training=False,
      fine_tune_batch_norm=False,
      model_parallelism=model_parallelism,
      branch=branch)

  patch_mat_pred = logits2prediction(patch_mat_scales, patch_images)
  patch_color_pred = logits2prediction(patch_color_scales, patch_images, sigmoid=False)
  return patch_mat_pred, patch_color_pred


def logits2prediction(outputs_to_scales_to_logits, images, method=None, sigmoid=True):
    """
    Convert predictions to predictions.

    Args:
        outputs_to_scales_to_logits: (todo): write your description
        images: (list): write your description
        method: (str): write your description
        sigmoid: (float): write your description
    """
  predictions = {}
  for i, output in enumerate(sorted(outputs_to_scales_to_logits)):
    scales_to_logits = outputs_to_scales_to_logits[output]
    if sigmoid:
        scales_to_logits[_MERGED_LOGITS_SCOPE] = tf.nn.sigmoid(scales_to_logits[_MERGED_LOGITS_SCOPE])
    logits = tf.image.resize_bilinear(
        scales_to_logits[_MERGED_LOGITS_SCOPE],
        tf.shape(images)[1:3],
        align_corners=True)
    predictions[output] = logits
    if method == 'argmax':
        predictions[output] = tf.argmax(logits, 3)

    if i == len(sorted(outputs_to_scales_to_logits))-1:
      predictions['semantic_final'] = predictions[output]

  return predictions


def scale_dimension(dim, scale):
  """Scales the input dimension.

  Args:
    dim: Input dimension (a scalar or a scalar Tensor).
    scale: The amount of scaling applied to the input.

  Returns:
    Scaled dimension.
  """
  if isinstance(dim, tf.Tensor):
    return tf.cast((tf.to_float(dim) - 1.0) * scale + 1.0, dtype=tf.int32)
  else:
    return int((float(dim) - 1.0) * scale + 1.0)


def multi_scale_logits_conmat(comp_images,
                              patch_images,
                              comp_options,
                              patch_options,
                              mode,
                              image_pyramid,
                              color_refine_one_time=True,
                              color_refine_layer_id=1,
                              mat_refine_one_time=False,
                              mat_refine_layer_id=1,
                              add_trimap=False,
                              decoder_depth=128,
                              weight_decay=0.0001,
                              kernel_size=3,
                              is_training=False,
                              fine_tune_batch_norm=False,
                              fine_tune_batch_norm_decoder=True,
                              model_parallelism=False,
                              branch=(True, True)):
  """
  Gets the logits for multi-scale inputs.
  """
  # Setup default values.
  if not image_pyramid:
    image_pyramid = [1.0]

  # Compute the logits for each scale in the image pyramid.
  patch_mat_scales = {}
  patch_seg_scales = {}

  for count, image_scale in enumerate(image_pyramid):
    if image_scale != 1.0:
        raise ValueError(
            "we don't support multi scale yet.")

    patch_mat_logits, patch_color_logits  = _get_logits_conmat(
        comp_images,
        patch_images,
        comp_options,
        patch_options,
        mode,
        color_refine_one_time=color_refine_one_time,
        color_refine_layer_id=color_refine_layer_id,
        mat_refine_one_time=mat_refine_one_time,
        mat_refine_layer_id=mat_refine_layer_id,
        add_trimap=add_trimap,
        kernel_size=kernel_size,
        decoder_depth=decoder_depth,
        weight_decay=weight_decay,
        reuse=True if count else None,
        is_training=is_training,
        fine_tune_batch_norm=fine_tune_batch_norm,
        fine_tune_batch_norm_decoder=fine_tune_batch_norm_decoder,
        model_parallelism=model_parallelism,
        branch=branch)

    if patch_mat_logits is not None:
        for output in sorted(patch_mat_logits):
            patch_mat_scales[output] = {}
            patch_mat_scales[output][_MERGED_LOGITS_SCOPE] = patch_mat_logits[output]
    if patch_color_logits is not None:
        for output in sorted(patch_color_logits):
            patch_seg_scales[output] = {}
            patch_seg_scales[output][_MERGED_LOGITS_SCOPE] = patch_color_logits[output]
    return patch_mat_scales, patch_seg_scales


def _add_aspp(features,
              model_options,
              weight_decay=0.0001,
              reuse=None,
              is_training=False,
              fine_tune_batch_norm=False):
    """
    Add image aspp model.

    Args:
        features: (array): write your description
        model_options: (todo): write your description
        weight_decay: (todo): write your description
        reuse: (todo): write your description
        is_training: (bool): write your description
        fine_tune_batch_norm: (bool): write your description
    """
    if not model_options.aspp_with_batch_norm:
        return features
    else:
        batch_norm_params = {
            'is_training': is_training and fine_tune_batch_norm,
            'decay': 0.9997,
            'epsilon': 1e-5,
            'scale': True,
        }

        with slim.arg_scope(
                [slim.conv2d, slim.separable_conv2d],
                weights_regularizer=slim.l2_regularizer(weight_decay),
                activation_fn=tf.nn.relu,
                normalizer_fn=slim.batch_norm,
                padding='SAME',
                stride=1,
                reuse=reuse):
            with slim.arg_scope([slim.batch_norm], **batch_norm_params):
                depth = 256
                branch_logits = []

                if model_options.add_image_level_feature:
                    pool_height = scale_dimension(model_options.crop_size[0],
                                                  1. / model_options.output_stride)
                    pool_width = scale_dimension(model_options.crop_size[1],
                                                 1. / model_options.output_stride)
                    image_feature = slim.avg_pool2d(
                        features, [pool_height, pool_width], [pool_height, pool_width],
                        padding='VALID')
                    image_feature = slim.conv2d(
                        image_feature, depth, 1, scope=_IMAGE_POOLING_SCOPE)
                    image_feature = tf.image.resize_bilinear(
                        image_feature, [pool_height, pool_width], align_corners=True)
                    image_feature.set_shape([None, pool_height, pool_width, depth])
                    branch_logits.append(image_feature)

                # Employ a 1x1 convolution.
                branch_logits.append(slim.conv2d(features, depth, 1,
                                                 scope=_ASPP_SCOPE + str(0)))

                if model_options.atrous_rates:
                    # Employ 3x3 convolutions with different atrous rates.
                    for i, rate in enumerate(model_options.atrous_rates, 1):
                        scope = _ASPP_SCOPE + str(i)
                        if model_options.aspp_with_separable_conv:
                            aspp_features = _split_separable_conv2d(
                                features,
                                filters=depth,
                                rate=rate,
                                weight_decay=weight_decay,
                                scope=scope)
                        else:
                            aspp_features = slim.conv2d(
                                features, depth, 3, rate=rate, scope=scope)
                        branch_logits.append(aspp_features)

                # Merge branch logits.

                concat_logits = tf.concat(branch_logits, 3)
                concat_logits = slim.conv2d(
                    concat_logits, depth, 1, scope=_CONCAT_PROJECTION_SCOPE)
                concat_logits = slim.dropout(
                    concat_logits,
                    keep_prob=0.9,
                    is_training=is_training,
                    scope=_CONCAT_PROJECTION_SCOPE + '_dropout')

                return concat_logits


def _extract_features(images,
                      model_options,
                      weight_decay=0.0001,
                      reuse=None,
                      is_training=False,
                      fine_tune_batch_norm=False,
                      model_parallelism=False,
                      device_id=0):
  """Extracts features by the particular model_variant.

  Args:
    images: A tensor of size [batch, height, width, channels].
    model_options: A ModelOptions instance to configure models.
    weight_decay: The weight decay for model variables.
    reuse: Reuse the model variables or not.
    is_training: Is training or not.
    fine_tune_batch_norm: Fine-tune the batch norm parameters or not.

  Returns:
    concat_logits: A tensor of size [batch, feature_height, feature_width,
      feature_channels], where feature_height/feature_width are determined by
      the images height/width and output_stride.
    end_points: A dictionary from components of the network to the corresponding
      activation.
  """
  if model_parallelism:
      with tf.device('/gpu:%d'%device_id):
          features, end_points = feature_extractor.extract_features(
              images,
              output_stride=model_options.output_stride,
              multi_grid=model_options.multi_grid,
              model_variant=model_options.model_variant,
              weight_decay=weight_decay,
              reuse=reuse,
              is_training=is_training,
              fine_tune_batch_norm=fine_tune_batch_norm)
  else:
      features, end_points = feature_extractor.extract_features(
          images,
          output_stride=model_options.output_stride,
          multi_grid=model_options.multi_grid,
          model_variant=model_options.model_variant,
          weight_decay=weight_decay,
          reuse=reuse,
          is_training=is_training,
          fine_tune_batch_norm=fine_tune_batch_norm)

  if model_parallelism:
      small_model = ['mobilenet_v2']
      device = '/gpu:%d'%device_id if model_options.model_variant in small_model else '/cpu:0'
      with tf.device(device):
          features = tf.identity(features)
          features = _add_aspp(
              features,
              model_options=model_options,
              weight_decay=weight_decay,
              reuse=reuse,
              is_training=is_training,
              fine_tune_batch_norm=fine_tune_batch_norm
          )
  else:
      features = _add_aspp(
          features,
          model_options=model_options,
          weight_decay=weight_decay,
          reuse=reuse,
          is_training=is_training,
          fine_tune_batch_norm=fine_tune_batch_norm
      )

  return features, end_points


def _add_features_endpoints(scope, model_variant, images, endpoints):
    """
    Add endpoints to model endpoints.

    Args:
        scope: (todo): write your description
        model_variant: (todo): write your description
        images: (todo): write your description
        endpoints: (todo): write your description
    """
    if 'mobilenet' in model_variant:
        scope_name = ''
        model_variant_name = ''
    elif 'resnet' in model_variant:
        scope_name = list(endpoints.keys())[0]
        scope_name = scope_name.split('/')[:2]
        scope_name = '/'.join(scope_name) + '/'
        model_variant_name = model_variant + '/'
    else:
        scope_name = list(endpoints.keys())[0]
        scope_name = scope_name.split('/')[:2]
        scope_name = '/'.join(scope_name) + '/'
        # scope_name = scope + '/'
        model_variant_name = model_variant + '/'

    images_shape = tf.shape(images)
    endpoints[scope_name + model_variant_name + 'images'] = tf.identity((2.0 / 255) * images - 1.0)
    end_name = scope_name + feature_extractor.networks_aux_feature_maps[model_variant][feature_extractor.DECODER_END_POINTS][0]
    endpoints[scope_name + model_variant_name + 'conv1'] = tf.image.resize_bilinear(
          endpoints[end_name], images_shape[1:3], align_corners=True)

    return endpoints


def build_branch(
        images,
        model_options,
        extra_feature = None,
        color_refine_one_time=False,
        color_refine_layer_id = 0,
        mat_refine_one_time=False,
        mat_refine_layer_id = 0,
        weight_decay=0.0001,
        add_trimap=False,
        decoder_depth=128,
        kernel_size=3,
        reuse=None,
        is_training=False,
        fine_tune_batch_norm=False,
        fine_tune_batch_norm_decoder=None,
        model_parallelism=False,
        mat_branch=True,
        seg_branch=True):
    """
    Builds the model for training.

    Args:
        images: (list): write your description
        model_options: (todo): write your description
        extra_feature: (str): write your description
        color_refine_one_time: (str): write your description
        color_refine_layer_id: (str): write your description
        mat_refine_one_time: (str): write your description
        mat_refine_layer_id: (str): write your description
        weight_decay: (str): write your description
        add_trimap: (str): write your description
        decoder_depth: (str): write your description
        kernel_size: (int): write your description
        reuse: (todo): write your description
        is_training: (bool): write your description
        fine_tune_batch_norm: (bool): write your description
        fine_tune_batch_norm_decoder: (bool): write your description
        model_parallelism: (bool): write your description
        mat_branch: (str): write your description
        seg_branch: (todo): write your description
    """

  if fine_tune_batch_norm_decoder is None:
     fine_tune_batch_norm_decoder = fine_tune_batch_norm

  with tf.variable_scope('encoder'):
      input = tf.identity(images, name='input')
      if model_parallelism:
          with slim.arg_scope([slim.model_variable, slim.variable], device='/cpu:0'):
              features, end_points = _extract_features(
                  input,
                  model_options,
                  weight_decay=weight_decay,
                  reuse=reuse,
                  is_training=is_training,
                  fine_tune_batch_norm=fine_tune_batch_norm,
                  model_parallelism=model_parallelism,
                  device_id=0)
      else:
          features, end_points = _extract_features(
              input,
              model_options,
              weight_decay=weight_decay,
              reuse=reuse,
              is_training=is_training,
              fine_tune_batch_norm=fine_tune_batch_norm,
              model_parallelism=model_parallelism,
              device_id=0)

  if model_options.decoder_output_stride == 1:
      end_points = _add_features_endpoints('encoder', model_options.model_variant, images, end_points)

  if extra_feature is not None:
      features = tf.concat([features, extra_feature], axis=-1)


  mat_decoder_features = None
  mat_decoder_endpoints = None
  if mat_branch:
      with tf.variable_scope('mat_decoder'):
          if model_parallelism:
              # with tf.device('/gpu:0'):
              with slim.arg_scope([slim.model_variable, slim.variable], device='/cpu:0'):
                      mat_decoder_features, mat_decoder_endpoints = refine_by_decoder_conmat(
                          features,
                          end_points,
                          refine_one_time=mat_refine_one_time,
                          refine_layer_id=mat_refine_layer_id,
                          add_trimap=add_trimap,
                          decoder_depth=decoder_depth,
                          kernel_size=kernel_size,
                          decoder_use_separable_conv=model_options.mat_decoder_use_separable_conv,
                          model_variant=model_options.model_variant,
                          weight_decay=weight_decay,
                          reuse=reuse,
                          is_training=is_training,
                          fine_tune_batch_norm=fine_tune_batch_norm_decoder)
          else:
              mat_decoder_features, mat_decoder_endpoints = refine_by_decoder_conmat(
                  features,
                  end_points,
                  refine_one_time=mat_refine_one_time,
                  refine_layer_id=mat_refine_layer_id,
                  add_trimap=add_trimap,
                  decoder_depth=decoder_depth,
                  kernel_size=kernel_size,
                  decoder_use_separable_conv=model_options.mat_decoder_use_separable_conv,
                  model_variant=model_options.model_variant,
                  weight_decay=weight_decay,
                  reuse=reuse,
                  is_training=is_training,
                  fine_tune_batch_norm=fine_tune_batch_norm_decoder)

  color_decoder_features = None
  color_decoder_endpoints = None
  if seg_branch:
      with tf.variable_scope('seg_decoder') as scope:
          if model_parallelism:
              with slim.arg_scope([slim.model_variable, slim.variable], device='/cpu:0'):
                  # with tf.device('/cpu:0'):
                      color_decoder_features, color_decoder_endpoints = refine_by_decoder_conmat(
                          features,
                          end_points,
                          refine_one_time=color_refine_one_time,
                          refine_layer_id=color_refine_layer_id,
                          add_trimap=add_trimap,
                          decoder_depth=decoder_depth,
                          kernel_size=kernel_size,
                          decoder_use_separable_conv=model_options.seg_decoder_use_separable_conv,
                          model_variant=model_options.model_variant,
                          weight_decay=weight_decay,
                          reuse=reuse,
                          is_training=is_training,
                          fine_tune_batch_norm=fine_tune_batch_norm_decoder)
          else:
              color_decoder_features, color_decoder_endpoints = refine_by_decoder_conmat(
                  features,
                  end_points,
                  refine_one_time=color_refine_one_time,
                  refine_layer_id=color_refine_layer_id,
                  add_trimap=add_trimap,
                  decoder_depth=decoder_depth,
                  kernel_size=kernel_size,
                  decoder_use_separable_conv=model_options.seg_decoder_use_separable_conv,
                  model_variant=model_options.model_variant,
                  weight_decay=weight_decay,
                  reuse=reuse,
                  is_training=is_training,
                  fine_tune_batch_norm=fine_tune_batch_norm_decoder)
  return features, end_points, mat_decoder_features, mat_decoder_endpoints, color_decoder_features, color_decoder_endpoints


def _build_logits(
        endpoints,
        num_class,
        atrous_rates,
        aspp_with_batch_norm,
        logits_kernel_size,
        weight_decay,
        reuse,
        scope_suffix,):
    """
    : param endpoints matrix. logits.

    Args:
        endpoints: (todo): write your description
        num_class: (int): write your description
        atrous_rates: (float): write your description
        aspp_with_batch_norm: (todo): write your description
        logits_kernel_size: (int): write your description
        weight_decay: (str): write your description
        reuse: (todo): write your description
        scope_suffix: (str): write your description
    """
  outputs_to_logits = {}
  for i, features in enumerate(endpoints):
    outputs_to_logits[i] = _get_branch_logits(
      features,
      num_class,
      atrous_rates,
      aspp_with_batch_norm=aspp_with_batch_norm,
      kernel_size=logits_kernel_size,
      weight_decay=weight_decay,
      reuse=reuse,
      scope_suffix=scope_suffix + '_' + str(i))
  return outputs_to_logits



def _get_logits_conmat(comp_images,
                       patch_images,
                       comp_model_options,
                       patch_model_options,
                       mode,
                       color_refine_one_time=False,
                       color_refine_layer_id = 1,
                       mat_refine_one_time=False,
                       mat_refine_layer_id = 1,
                       weight_decay=0.0001,
                       add_trimap=False,
                       decoder_depth=128,
                       kernel_size=3,
                       reuse=None,
                       is_training=False,
                       fine_tune_batch_norm=False,
                       fine_tune_batch_norm_decoder=None,
                       model_parallelism=False,
                       branch=(True, True)):
  """Gets the logits by atrous/image spatial pyramid pooling.

  Args:
    images: A tensor of size [batch, height, width, channels].
    model_options: A ModelOptions instance to configure models.
    weight_decay: The weight decay for model variables.
    reuse: Reuse the model variables or not.
    is_training: Is training or not.
    fine_tune_batch_norm: Fine-tune the batch norm parameters or not.

  Returns:
    outputs_to_logits: A map from output_type to logits.
  """

  # set v10 params
  comp_decoder_depth = decoder_depth
  comp_refine_layer_id = color_refine_layer_id
  if mode['model'] in ['v10']:
      if comp_model_options.model_variant == 'xception_65':
          comp_refine_layer_id = 1
          comp_model_options = comp_model_options._replace(seg_decoder_use_separable_conv=True)

  with tf.variable_scope('comp') as scope:
    (comp_features, comp_end_points, comp_mat_decoder_features, comp_mat_decoder_endpoints,
     comp_color_decoder_features, comp_color_decoder_endpoints) = build_branch(
        images=comp_images,
        model_options=comp_model_options,
        extra_feature=None,
        color_refine_one_time=color_refine_one_time,
        color_refine_layer_id=comp_refine_layer_id,
        mat_refine_one_time=mat_refine_one_time,
        mat_refine_layer_id=mat_refine_layer_id,
        weight_decay=weight_decay,
        add_trimap=add_trimap,
        decoder_depth=comp_decoder_depth,
        kernel_size=kernel_size,
        reuse=reuse,
        is_training=is_training,
        fine_tune_batch_norm=fine_tune_batch_norm,
        fine_tune_batch_norm_decoder=fine_tune_batch_norm_decoder,
        model_parallelism=model_parallelism,
        mat_branch=False,
        seg_branch=True,
    )

  if mode['model'] == 'v10':
      patch_input = patch_images
      comp_proj_features =  slim.repeat(
         comp_color_decoder_features,
         mode['num_convs'],
         slim.conv2d,
         mode['proj_depth'],
         kernel_size,
         scope='comp_proj')
      pool_height = scale_dimension(patch_model_options.crop_size[0],
                                    1. / patch_model_options.output_stride)
      pool_width = scale_dimension(patch_model_options.crop_size[1],
                                   1. / patch_model_options.output_stride)
      crop_size = [pool_height, pool_width]
      extra_feature = tf.image.resize_bilinear(comp_proj_features, crop_size, align_corners=True)
  else:
      raise ValueError('unsupported model version')


  with tf.variable_scope('patch') as scope:
    (patch_features, patch_end_points, patch_mat_decoder_features, patch_mat_decoder_endpoints,
     patch_color_decoder_features, patch_color_decoder_endpoints) = build_branch(
        images=patch_input,
        model_options=patch_model_options,
        extra_feature=extra_feature,
        color_refine_one_time=False,
        color_refine_layer_id=color_refine_layer_id,
        mat_refine_one_time=False,
        mat_refine_layer_id=mat_refine_layer_id,
        weight_decay=weight_decay,
        add_trimap=add_trimap,
        decoder_depth=decoder_depth,
        kernel_size=kernel_size,
        reuse=reuse,
        is_training=is_training,
        fine_tune_batch_norm=fine_tune_batch_norm,
        fine_tune_batch_norm_decoder=fine_tune_batch_norm_decoder,
        model_parallelism=model_parallelism,
        mat_branch=branch[0],
        seg_branch=branch[1],
    )


  with tf.variable_scope('logits') as scope:
      patch_mat_logits = None
      if patch_mat_decoder_endpoints is not None:
          patch_mat_logits = _build_logits(
              endpoints=patch_mat_decoder_endpoints,
              num_class=1,
              atrous_rates=patch_model_options.atrous_rates,
              aspp_with_batch_norm=patch_model_options.aspp_with_batch_norm,
              logits_kernel_size=patch_model_options.logits_kernel_size,
              weight_decay=weight_decay,
              reuse=reuse,
              scope_suffix='patch_mat_logits'
          )

      patch_color_logits = None
      if patch_color_decoder_endpoints is not None:
          patch_color_logits = _build_logits(
              endpoints=patch_color_decoder_endpoints,
              num_class=3, # we change the patch seg to patch color
              atrous_rates=patch_model_options.atrous_rates,
              aspp_with_batch_norm=patch_model_options.aspp_with_batch_norm,
              logits_kernel_size=patch_model_options.logits_kernel_size,
              weight_decay=weight_decay,
              reuse=reuse,
              scope_suffix='patch_seg_logits'
          )

  return patch_mat_logits, patch_color_logits


def get_feature_name(name, model_variant, scope):
    """
    Returns the name of a model name.

    Args:
        name: (str): write your description
        model_variant: (todo): write your description
        scope: (todo): write your description
    """
    if model_variant == 'mobilenet_v2':
        feature_name = '{}'.format(name)
    elif 'xception' in model_variant:
        name_scope = tf.get_variable_scope().name.split('/')[0]
        feature_name = '{}/{}/{}'.format(name_scope, scope, name)
        # feature_name = '{}/{}'.format(scope, name)
    elif 'resnet' in model_variant:
        name_scope = tf.get_variable_scope().name.split('/')[0]
        feature_name = '{}/{}/{}'.format(name_scope, scope, name)
    else:
        ValueError('please use valid feature')
    return feature_name


def refine_by_decoder_conmat(features,
                             end_points,
                             refine_one_time=False,
                             refine_layer_id = 0,
                             encoder_scope='encoder',
                             trimaps=None,
                             kernel_size=3,
                             add_trimap=False,
                             decoder_depth=128,
                             decoder_use_separable_conv=False,
                             model_variant=None,
                             weight_decay=0.0001,
                             reuse=None,
                             is_training=False,
                             fine_tune_batch_norm=False):
    """
    Refine decoder.

    Args:
        features: (todo): write your description
        end_points: (bool): write your description
        refine_one_time: (bool): write your description
        refine_layer_id: (str): write your description
        encoder_scope: (str): write your description
        trimaps: (todo): write your description
        kernel_size: (int): write your description
        add_trimap: (todo): write your description
        decoder_depth: (str): write your description
        decoder_use_separable_conv: (bool): write your description
        model_variant: (todo): write your description
        weight_decay: (todo): write your description
        reuse: (todo): write your description
        is_training: (bool): write your description
        fine_tune_batch_norm: (bool): write your description
    """
  batch_norm_params = {
      'is_training': is_training and fine_tune_batch_norm,
      'decay': 0.9997,
      'epsilon': 1e-5,
      'scale': True,
  }

  with slim.arg_scope(
      [slim.conv2d, slim.separable_conv2d],
      weights_regularizer=slim.l2_regularizer(weight_decay),
      activation_fn=tf.nn.relu,
      normalizer_fn=slim.batch_norm,
      padding='SAME',
      stride=1,
      reuse=reuse):
    with slim.arg_scope([slim.batch_norm], **batch_norm_params):
      with tf.variable_scope(_DECODER_SCOPE, _DECODER_SCOPE, [features, ]):
        feature_list = feature_extractor.networks_to_feature_maps[
            model_variant][feature_extractor.DECODER_END_POINTS][:]

        if refine_one_time:
            feature_list = [feature_list[refine_layer_id], ]

        decoder_features = features
        decoder_endpoints = [decoder_features]
        for i, name, in enumerate(feature_list, ):
          feature_name = get_feature_name(name, model_variant, encoder_scope)

          encoder_features =  slim.conv2d(
                  end_points[feature_name],
                  48,
                  1,
                  scope='feature_projection' + str(i))

          encoder_shape = tf.shape(encoder_features)
          decoder_features = tf.image.resize_bilinear(
              decoder_features, encoder_shape[1:3], align_corners=True)

          if add_trimap:
              resized_trimaps = tf.image.resize_bilinear(
                  trimaps, encoder_shape[1:3], align_corners=True
              )
              decoder_features = tf.concat([resized_trimaps, encoder_features, decoder_features], 3)
          else:
              decoder_features = tf.concat([encoder_features, decoder_features], 3)


          if decoder_use_separable_conv:
            decoder_features = _split_separable_conv2d(
                decoder_features,
                filters=decoder_depth,
                kernel_size = kernel_size,
                rate=1,
                weight_decay=weight_decay,
                scope='decoder_conv0_proj' + str(i))
            decoder_features = _split_separable_conv2d(
                decoder_features,
                filters=decoder_depth,
                kernel_size = kernel_size,
                rate=1,
                weight_decay=weight_decay,
                scope='decoder_conv1_proj' + str(i))
          else:
            num_convs = 2
            decoder_features = slim.repeat(
                decoder_features,
                num_convs,
                slim.conv2d,
                decoder_depth,
                kernel_size,
                scope='decoder_conv' + str(i))
          decoder_endpoints.append(decoder_features)
        return decoder_features, decoder_endpoints


"""
Ref deeplab codes: https://github.com/tensorflow/models/tree/master/research/deeplab
"""
def _get_branch_logits(features,
                       num_classes,
                       atrous_rates=None,
                       aspp_with_batch_norm=False,
                       kernel_size=1,
                       weight_decay=0.0001,
                       reuse=None,
                       scope_suffix=''):
    """
    Get the logits.

    Args:
        features: (str): write your description
        num_classes: (int): write your description
        atrous_rates: (str): write your description
        aspp_with_batch_norm: (bool): write your description
        kernel_size: (int): write your description
        weight_decay: (str): write your description
        reuse: (todo): write your description
        scope_suffix: (str): write your description
    """
  if aspp_with_batch_norm or atrous_rates is None:
    if kernel_size != 1:
      raise ValueError('Kernel size must be 1 when atrous_rates is None or '
                       'using aspp_with_batch_norm. Gets %d.' % kernel_size)
    atrous_rates = [1]

  with slim.arg_scope(
      [slim.conv2d],
      weights_regularizer=slim.l2_regularizer(weight_decay),
      weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
      reuse=reuse):
    with tf.variable_scope(_LOGITS_SCOPE_NAME, _LOGITS_SCOPE_NAME, [features]):
      branch_logits = []
      for i, rate in enumerate(atrous_rates):
        scope = scope_suffix
        if i:
          scope += '_%d' % i

        branch_logits.append(
            slim.conv2d(
                features,
                num_classes,
                kernel_size=kernel_size,
                rate=rate,
                activation_fn=None,
                normalizer_fn=None,
                scope=scope))

      return tf.add_n(branch_logits)


"""
Ref deeplab codes: https://github.com/tensorflow/models/tree/master/research/deeplab
"""
def _split_separable_conv2d(inputs,
                            filters,
                            rate=1,
                            weight_decay=0.00004,
                            kernel_size=3,
                            depthwise_weights_initializer_stddev=0.33,
                            pointwise_weights_initializer_stddev=0.06,
                            scope=None):
  """Splits a separable conv2d into depthwise and pointwise conv2d.

  This operation differs from `tf.layers.separable_conv2d` as this operation
  applies activation function between depthwise and pointwise conv2d.

  Args:
    inputs: Input tensor with shape [batch, height, width, channels].
    filters: Number of filters in the 1x1 pointwise convolution.
    rate: Atrous convolution rate for the depthwise convolution.
    weight_decay: The weight decay to use for regularizing the model.
    depthwise_weights_initializer_stddev: The standard deviation of the
      truncated normal weight initializer for depthwise convolution.
    pointwise_weights_initializer_stddev: The standard deviation of the
      truncated normal weight initializer for pointwise convolution.
    scope: Optional scope for the operation.

  Returns:
    Computed features after split separable conv2d.
  """
  outputs = slim.separable_conv2d(
      inputs,
      None,
      kernel_size,
      depth_multiplier=1,
      rate=rate,
      weights_initializer=tf.truncated_normal_initializer(
          stddev=depthwise_weights_initializer_stddev),
      weights_regularizer=None,
      scope=scope + '_depthwise')
  return slim.conv2d(
      outputs,
      filters,
      1,
      weights_initializer=tf.truncated_normal_initializer(
          stddev=pointwise_weights_initializer_stddev),
      weights_regularizer=slim.l2_regularizer(weight_decay),
      scope=scope + '_pointwise')
