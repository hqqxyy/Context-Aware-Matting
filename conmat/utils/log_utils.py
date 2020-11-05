from __future__ import print_function
import os
import os.path as osp
import _pickle as pickle
from scipy import io
import datetime
import time
from contextlib import contextmanager


class ReDirectSTD(object):
  """Modified from Tong Xiao's `Logger` in open-reid.
  This class overwrites sys.stdout or sys.stderr, so that console logs can
  also be written to file.
  Args:
    fpath: file path
    console: one of ['stdout', 'stderr']
    immediately_visible: If `False`, the file is opened only once and closed
      after exiting. In this case, the message written to file may not be
      immediately visible (Because the file handle is occupied by the
      program?). If `True`, each writing operation of the console will
      open, write to, and close the file. If your program has tons of writing
      operations, the cost of opening and closing file may be obvious. (?)
  Usage example:
    `ReDirectSTD('stdout.txt', 'stdout', False)`
    `ReDirectSTD('stderr.txt', 'stderr', False)`
  NOTE: File will be deleted if already existing. Log dir and file is created
    lazily -- if no message is written, the dir and file will not be created.
  """

  def __init__(self, fpath=None, console='stdout', immediately_visible=False):
      """
      Initialize the console.

      Args:
          self: (todo): write your description
          fpath: (str): write your description
          console: (todo): write your description
          immediately_visible: (bool): write your description
      """
    import sys
    import os
    import os.path as osp

    assert console in ['stdout', 'stderr']
    self.console = sys.stdout if console == 'stdout' else sys.stderr
    self.file = fpath
    self.f = None
    self.immediately_visible = immediately_visible
    if fpath is not None:
      # Remove existing log file.
      if osp.exists(fpath):
        os.remove(fpath)

    # Overwrite
    if console == 'stdout':
      sys.stdout = self
    else:
      sys.stderr = self

  def __del__(self):
      """
      Closes the stream.

      Args:
          self: (todo): write your description
      """
    self.close()

  def __enter__(self):
      """
      Enter the callable

      Args:
          self: (todo): write your description
      """
    pass

  def __exit__(self, *args):
      """
      Exit the exit.

      Args:
          self: (todo): write your description
      """
    self.close()

  def write(self, msg):
      """
      Write a message to the console.

      Args:
          self: (todo): write your description
          msg: (str): write your description
      """
    self.console.write(msg)
    if self.file is not None:
      may_make_dir(os.path.dirname(osp.abspath(self.file)))
      if self.immediately_visible:
        with open(self.file, 'a') as f:
          f.write(msg)
      else:
        if self.f is None:
          self.f = open(self.file, 'w')
        self.f.write(msg)

  def flush(self):
      """
      Flush the console

      Args:
          self: (todo): write your description
      """
    self.console.flush()
    if self.f is not None:
      self.f.flush()
      import os
      os.fsync(self.f.fileno())

  def close(self):
      """
      Close the console.

      Args:
          self: (todo): write your description
      """
    self.console.close()
    if self.f is not None:
      self.f.close()


def may_make_dir(path):
  """
  Args:
    path: a dir, or result of `osp.dirname(osp.abspath(file_path))`
  Note:
    `osp.exists('')` returns `False`, while `osp.exists('.')` returns `True`!
  """
  # This clause has mistakes:
  # if path is None or '':

  if path in [None, '']:
    return
  if not osp.exists(path):
    os.makedirs(path)


def time_str(fmt=None):
    """
    Return a string representing the time of the string.

    Args:
        fmt: (str): write your description
    """
  if fmt is None:
    fmt = '%Y-%m-%d_%H:%M:%S'
  return datetime.datetime.today().strftime(fmt)


