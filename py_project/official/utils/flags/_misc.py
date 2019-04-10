from absl import flags
from official.utils.flags._conventions import help_wrap


def define_image(data_format=True):
    key_flags = []
    if data_format:
        flags.DEFINE_enum(
            name="data_format",
            short_name="df", default=None,
            # 代表图像的通道维的位置
            enum_values=['channels_first', 'channel_last'],
            help=help_wrap(
                "A flag to override the data format used in the model. "
                "channels_first provides a performance boost on GPU but is not "
                "always compatible with CPU. If left unspecified, the data format "
                "will be chosen automatically based on whether TensorFlow was "
                "built for CPU or GPU."))
    key_flags.append("data_format")
    return key_flags
