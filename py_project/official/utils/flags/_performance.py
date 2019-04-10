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

    return key_flags
