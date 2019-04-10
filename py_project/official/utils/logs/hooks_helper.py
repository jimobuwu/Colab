from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

_TENSORS_TO_LOG = dict((x, x) for x in ['learning_rate',
                                        'cross_entropy',
                                        'train_accuracy'])


def get_logging_tensor_hook(every_n_iter=100, tensors_to_log=None, **kwargs):
    if tensors_to_log is None:
        tensors_to_log = _TENSORS_TO_LOG

    return tf.estimator.LoggingTensorHook(
        tensors=tensors_to_log,
        every_n_iter=every_n_iter)


def get_train_hooks(name_list, use_tpu=False, **kwargs):
    if not name_list:
        return []
    if use_tpu:
        tf.compat.v1.logging.warning('hooks_helper received name_list `{}`, but a '
                                     'TPU is specified. No hooks will be used.'
                                     .format(name_list))
        return []

    train_hooks = []
    for name in name_list:
        hook_name = HOOKS.get(name.strip().lower())
        if hook_name is None:
            raise ValueError('Unrecognized training hook requested: {}'.format(name))
        else:
            train_hooks.append(hook_name(**kwargs))

    return train_hooks


HOOKS = {
    'loggingtensorhook': get_logging_tensor_hook,
}