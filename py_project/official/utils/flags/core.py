from absl import flags

from official.utils.flags import _base
from official.utils.flags import _performance
from official.utils.flags import _misc


get_num_gpus = _base.get_num_gpus


def set_defaults(**kwargs):
    for key, value in kwargs.items():
        flags.FLAGS.set_default(name=key, value=value)


def register_key_flags_in_core(f):
    def core_fn(*args, **kwargs):
        key_flags = f(*args, **kwargs)
        [flags.declare_key_flag(fl) for fl in key_flags]
    return core_fn


define_base = register_key_flags_in_core(_base.define_base)
define_performance = register_key_flags_in_core(_performance.define_performance)
define_image = register_key_flags_in_core(_misc.define_image)