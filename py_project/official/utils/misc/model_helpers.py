from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numbers


def past_stop_threshold(stop_threshold, eval_metric):
    if stop_threshold is None:
        return False
    if not isinstance(stop_threshold, numbers.Number):
        raise ValueError("Threshold for checking stop conditions must be a number.")
    if not isinstance(eval_metric, numbers.Number):
        raise ValueError("Eval metric being checked against stop conditions "
                         "must be a number.")

    if eval_metric >= stop_threshold:
        tf.compat.v1.logging.info("Stop threshold of {} was passed with metric value {}.".format(
            stop_threshold, eval_metric))
        return True
    return False

def apply_clean(flags_obj):
    if flags_obj.clean and tf.gfile.Exists(flags_obj.model_dir):
        tf.logging.info("--clean flag set. Removing existing model dir:"
                        " {}".format(flags_obj.model_dir))
        tf.gfile.DeleteRecursively(flags_obj.model_dir)
