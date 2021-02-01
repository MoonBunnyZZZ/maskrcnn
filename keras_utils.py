from tensorflow.python.keras import backend

try:
  from tensorflow.python.keras.engine import keras_tensor  # pylint: disable=g-import-not-at-top,unused-import
  keras_tensor.disable_keras_tensors()
except ImportError:
  keras_tensor = None


class NoOpContextManager(object):

  def __enter__(self):
    pass

  def __exit__(self, *args):
    pass


def maybe_enter_backend_graph():
  if (keras_tensor is not None) and keras_tensor.keras_tensors_enabled():
    return NoOpContextManager()
  else:
    return backend.get_graph().as_default()