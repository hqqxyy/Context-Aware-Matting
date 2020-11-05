import sys, os
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.getcwd() + '/slim')

import numpy as np
from PIL import Image
import scipy.misc
import tensorflow as tf
from conmat import common
from conmat import model
from conmat.utils import save_annotation
import time

slim = tf.contrib.slim

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('master', '', 'BNS name of the tensorflow server')

# Settings for log directories.
flags.DEFINE_string('vis_logdir', None, 'Where to write the event logs.')
flags.DEFINE_string('checkpoint', None, 'checkpoint path')

# Settings for visualizing the model.
flags.DEFINE_integer('vis_batch_size', 1,
                     'The number of images in each batch during evaluation.')

flags.DEFINE_multi_integer('atrous_rates', [3, 6, 9],
                           'Atrous rates for atrous spatial pyramid pooling.')

flags.DEFINE_integer('comp_output_stride', 16,
                     'The ratio of input to output spatial resolution.')
flags.DEFINE_integer('patch_output_stride', 4,
                     'The ratio of input to output spatial resolution.')

# Dataset settings.
flags.DEFINE_string('vis_split', 'test',
                    'Which split of the dataset used for visualizing results')

flags.DEFINE_boolean('add_trimap', False, 'add trimap in decoder')
flags.DEFINE_string('last_features', 'images', 'features for the last layer')
flags.DEFINE_integer('decoder_depth', 64, 'decoder layer channel')
flags.DEFINE_integer('decoder_kernel_size', 3, 'decoder kernel size')
flags.DEFINE_boolean('model_parallelism', False, 'whether use cpu for the decoder.')
flags.DEFINE_multi_integer('branch_vis', [1, 0], 'show the result of branch')


# only release xception model
flags.DEFINE_enum('comp_model_variant', 'xception_65', ['xception_65',], 'context encoder model variant.')
flags.DEFINE_enum('patch_model_variant', 'xception_65', ['xception_65'], 'matting encoder model variant.')


flags.DEFINE_multi_integer('vis_comp_crop_size', [2010, 2101],
                           'composed image crop size [height, width]')
flags.DEFINE_multi_integer('vis_patch_crop_size', [2101, 2101],
                           'patch image crop size [height, width]')

# only release v10 model
flags.DEFINE_enum('mode', 'v10', ['v10'], 'model mode')
flags.DEFINE_integer('mode_num_convs', 1, 'conv number')
flags.DEFINE_integer('mode_depth', 8, 'projector depth')


flags.DEFINE_string('fgpath', None, 'fgpath')
flags.DEFINE_string('trimappath', None, 'trimappath')

# The format to save image.
_IMAGE_FORMAT = '%s_%06d_image'
_TRIMAP_FORMAT = '%s_%06d_trimap'


def _process_batch(sess, feed_dict, patch_images, patch_mat_preds, patch_color_preds, image_name, patch_size, save_dir):
    """
    Process a batch of images.

    Args:
        sess: (todo): write your description
        feed_dict: (dict): write your description
        patch_images: (todo): write your description
        patch_mat_preds: (str): write your description
        patch_color_preds: (str): write your description
        image_name: (str): write your description
        patch_size: (int): write your description
        save_dir: (str): write your description
    """
  (patch_mat_preds,
   patch_color_preds,) = sess.run([patch_mat_preds, patch_color_preds], feed_dict=feed_dict)

  patch_image = np.squeeze(feed_dict[patch_images])
  save_annotation.save_matting(
      patch_image[:patch_size[0],:patch_size[1],:3], save_dir, '%s_patch_image' % (image_name),)
  save_annotation.save_matting(
      patch_image[:patch_size[0],:patch_size[1],3], save_dir,  '%s_patch_trimap' % (image_name),)

  if isinstance(patch_mat_preds, np.ndarray):
      patch_mat_pred = np.squeeze(patch_mat_preds[0])

      patch_trimap = patch_image[...,3]
      patch_mat_pred[patch_trimap == 0] = 0
      patch_mat_pred[patch_trimap == 255] = 1

      save_annotation.save_matting(
          patch_mat_pred[:patch_size[0],:patch_size[1]], save_dir,
          '%s_mat_prediction' % (image_name), scale=True,)


  if isinstance(patch_color_preds, np.ndarray):
      patch_color_pred = np.squeeze(patch_color_preds[0]) * 255
      patch_color_pred = patch_color_pred.clip(0.0, 255.0)
      save_annotation.save_image(
          patch_color_pred[:patch_size[0],:patch_size[1]], save_dir,
          '%s_color_prediction' % (image_name))


def process_image(fgpath, trimappath=None, comp_size=(2101, 2101), patch_size=(2101, 2101)):
    """
    Process an image.

    Args:
        fgpath: (str): write your description
        trimappath: (str): write your description
        comp_size: (int): write your description
        patch_size: (int): write your description
    """
    fg = np.array(Image.open(fgpath).convert('RGB'))

    trimap = None
    if trimappath is not None:
        trimap = np.array(Image.open(trimappath))

    fgh, fgw = fg.shape[:2]
    comp = fg
    patch_image = np.ones((1, *patch_size, 4)) * 127.5
    patch_image[...,3] = 0
    patch_image[0, :fgh, :fgw, :3] = comp
    patch_image[0, :fgh, :fgw, 3] = trimap
    if comp_size != patch_size:
        comp_image = scipy.misc.imresize(patch_image[0, ...], comp_size)
        comp_image = comp_image[np.newaxis, ...]
    else:
        comp_image = patch_image
    comp_size = (int(fgh * comp_size[0] / patch_size[0]), int(fgw * comp_size[1] / patch_size[1]))
    patch_size = (fgh, fgw)
    return comp_image, patch_image, comp_size, patch_size


def main(unused_argv):
    """
    Main function.

    Args:
        unused_argv: (bool): write your description
    """
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.gfile.MakeDirs(FLAGS.vis_logdir)

  # Prepare for visualization.
  save_root = FLAGS.vis_logdir
  tf.gfile.MakeDirs(save_root)

  g = tf.Graph()
  with g.as_default():

    # set options for the network
    comp_options = common.ModelOptions(
        crop_size=FLAGS.vis_comp_crop_size,
        atrous_rates=FLAGS.atrous_rates,
        output_stride=FLAGS.comp_output_stride)
    comp_options = comp_options._replace(model_variant=FLAGS.comp_model_variant)

    patch_options = common.ModelOptions(
        crop_size=FLAGS.vis_patch_crop_size,
        atrous_rates=FLAGS.atrous_rates,
        output_stride=FLAGS.patch_output_stride)
    patch_options = patch_options._replace(model_variant=FLAGS.patch_model_variant)

    mode = {}
    mode['model']=FLAGS.mode
    mode['num_convs']=FLAGS.mode_num_convs
    mode['proj_depth']=FLAGS.mode_depth
    tf.logging.info('Performing single-scale test.')
    branch = [vis == 1 for vis in FLAGS.branch_vis]
    comp_image = tf.placeholder(tf.float32, shape=(1, *FLAGS.vis_comp_crop_size, 4))
    patch_image = tf.placeholder(tf.float32, shape=(1, *FLAGS.vis_patch_crop_size, 4))
    patch_mat_preds, patch_color_preds= model.predict_labels_conmat(
        comp_image,
        patch_image,
        comp_options=comp_options,
        patch_options=patch_options,
        mode=mode,
        add_trimap=FLAGS.add_trimap,
        decoder_depth=FLAGS.decoder_depth,
        kernel_size=FLAGS.decoder_kernel_size,
        image_pyramid=FLAGS.image_pyramid,
        model_parallelism=FLAGS.model_parallelism,
        branch=branch)

    if patch_mat_preds:
        patch_mat_preds = patch_mat_preds[common.OUTPUT_TYPE_PYRAMID]
    if patch_color_preds:
        patch_color_preds = patch_color_preds[common.OUTPUT_TYPE_PYRAMID]

    tf.train.get_or_create_global_step()
    saver = tf.train.Saver(slim.get_variables_to_restore())
    sv = tf.train.Supervisor(graph=g,
                             logdir=FLAGS.vis_logdir,
                             init_op=tf.global_variables_initializer(),
                             summary_op=None,
                             summary_writer=None,
                             global_step=None,
                             saver=saver)
    tf.logging.info(
        'Starting visualization at ' + time.strftime('%Y-%m-%d-%H:%M:%S',
                                                     time.gmtime()))
    tf.logging.info('Visualizing with model %s', FLAGS.checkpoint)

    with sv.managed_session(FLAGS.master,
                            start_standard_services=False) as sess:
      sv.saver.restore(sess, FLAGS.checkpoint)
      save_dir = save_root
      tf.gfile.MakeDirs(save_dir)

      fgpath = FLAGS.fgpath
      trimappath = FLAGS.trimappath
      tf.logging.info('Visualizing %s', fgpath)

      comp_np, patch_np, comp_size, patch_size = process_image(fgpath=fgpath,
                                                               trimappath=trimappath,
                                                               comp_size=FLAGS.vis_comp_crop_size,
                                                               patch_size=FLAGS.vis_patch_crop_size
                                                               )
      t1 = time.time()
      feed_dict = {comp_image: comp_np, patch_image:patch_np}
      _process_batch(sess=sess,
                     feed_dict=feed_dict,
                     patch_images=patch_image,
                     patch_mat_preds=patch_mat_preds,
                     patch_color_preds=patch_color_preds,
                     image_name=fgpath.split('/')[-1][:-4],
                     save_dir=save_dir,
                     patch_size=patch_size)
      t2 = time.time()
      tf.logging.info('time: %f', t2 - t1)

    tf.logging.info(
        'Finished visualization at ' + time.strftime('%Y-%m-%d-%H:%M:%S',
                                                     time.gmtime()))


if __name__ == '__main__':
  flags.mark_flag_as_required('checkpoint')
  flags.mark_flag_as_required('vis_logdir')
  tf.app.run()


