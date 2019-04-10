from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

# _COLLECTIVE_COMMUNICATION_OPTIONS = {
#     None: tf.distribute.experimental.CollectiveCommunication.AUTO,
#     "ring": tf.distribute.experimental.CollectiveCommunication.RING,
#     "nccl": tf.distribute.experimental.CollectiveCommunication.NCCL
# }

def get_distribution_strategy(distribution_strategy='default',
                              num_gpus=0,
                              num_workers=1,
                              all_reduce_alg=None):
    if num_gpus < 0:
        raise ValueError("'num_gpus' can not be negative.")

    distribution_strategy = distribution_strategy.lower()
    if distribution_strategy == "off":
        if num_gpus > 1 or num_workers > 1:
            raise ValueError(
                "when {} gpus and {} workers are distribution_strategy"
                "flag cannot be set to off.".format(num_gpus, num_workers))
        return None

    if distribution_strategy == 'multi_worker_mirrored' or num_workers > 1:
        # if all_reduce_alg not in _COLLECTIVE_COMMUNICATION_OPTIONS:
        #     raise ValueError(
        #         "When used with `multi_worker_mirrored`, valid values for "
        #         "all_reduce_alg are [`ring`, `nccl`].  Supplied value: {}".format(
        #             all_reduce_alg))
        # return tf.distribute.expermental.MultiWorkerMirroredStrategy(
        #     communication=_COLLECTIVE_COMMUNICATION_OPTIONS[all_reduce_alg]

        return tf.contrib.distribute.CollectiveAllReduceStrategy()
