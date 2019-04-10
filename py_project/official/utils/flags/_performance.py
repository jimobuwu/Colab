import multiprocessing
from absl import flags
import tensorflow as tf
from official.utils.flags._conventions import help_wrap

def define_performance(num_parallel_calls=True, inter_op=True, intra_op=True,
                       synthetic_data=True, max_train_steps=True, dtype=True,
                       all_reduce_alg=True, tf_gpu_thread_mode=False,
                       datasets_num_private_threads=False,
                       datasets_num_parallel_batches=False):
    key_flags = []
    if num_parallel_calls:
        flags.DEFINE_integer(
            name="num_parallel_calls", short_name="npc",
            default=multiprocessing.cpu_count(),
            help=help_wrap("The number of records that are  processed in parallel "
                           "during input processing. This can be optimized per "
                           "data set but for generally homogeneous data sets, "
                           "should be approximately the number of available CPU "
                           "cores. (default behavior)"))

    if inter_op:
        flags.DEFINE_integer(
            name="inter_op_parallelism_threads", short_name="inter", default=0,
            help=help_wrap("Number of inter_op_parallelism_threads to use for CPU. "
                           "See TensorFlow config.proto for details.")
        )

    if intra_op:
        flags.DEFINE_integer(
            name="intra_op_parallelism_threads", short_name="intra", default=0,
            help=help_wrap("Number of intra_op_parallelism_threads to use for CPU. "
                           "See TensorFlow config.proto for details."))

    if all_reduce_alg:
        flags.DEFINE_string(
            name="all_reduce_alg", short_name="ara", default=None,
            help=help_wrap("Defines the algorithm to use for performing all-reduce."
                           "When specified with MirroredStrategy for single "
                           "worker, this controls "
                           "tf.contrib.distribute.AllReduceCrossTowerOps.  When "
                           "specified with MultiWorkerMirroredStrategy, this "
                           "controls "
                           "tf.distribute.experimental.CollectiveCommunication; "
                           "valid options are `ring` and `nccl`."))

    return key_flags
