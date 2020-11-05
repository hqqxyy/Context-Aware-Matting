import numpy as np
import PIL.Image as img
import tensorflow as tf

def save_matting(label,
                 save_dir,
                 filename,
                 scale = False):
    """
    Save matlab matplotlib.

    Args:
        label: (array): write your description
        save_dir: (str): write your description
        filename: (str): write your description
        scale: (float): write your description
    """
  if scale:
      label *= 255
  pil_image = img.fromarray(label.astype(dtype=np.uint8))
  with tf.gfile.Open('%s/%s.png' % (save_dir, filename), mode='w') as f:
    pil_image.save(f, 'PNG')


def save_image(label,
               save_dir,
               filename):
    """
    Saves image as png file.

    Args:
        label: (array): write your description
        save_dir: (str): write your description
        filename: (str): write your description
    """
  pil_image = img.fromarray(label.astype(dtype=np.uint8))
  with tf.gfile.Open('%s/%s.png' % (save_dir, filename), mode='w') as f:
    pil_image.save(f, 'PNG')

