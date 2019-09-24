import numpy as np
import PIL.Image as img
import tensorflow as tf

def save_matting(label,
                 save_dir,
                 filename,
                 scale = False):
  if scale:
      label *= 255
  pil_image = img.fromarray(label.astype(dtype=np.uint8))
  with tf.gfile.Open('%s/%s.png' % (save_dir, filename), mode='w') as f:
    pil_image.save(f, 'PNG')


def save_image(label,
               save_dir,
               filename):
  pil_image = img.fromarray(label.astype(dtype=np.uint8))
  with tf.gfile.Open('%s/%s.png' % (save_dir, filename), mode='w') as f:
    pil_image.save(f, 'PNG')

