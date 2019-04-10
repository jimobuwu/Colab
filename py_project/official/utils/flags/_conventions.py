from absl import flags

import functools
import codecs

_help_wrap = functools.partial(flags.text_wrap, length=80, indent="",
                               firstline_indent='\n')

try:
    # Looks up the codec info in the Python codec registry
    codecs.lookup("utf-8")
    help_wrap = _help_wrap
except LookupError: # 不存在utf-8
    def help_wrap(text, *args, **kwargs):
        # utf-16等编码中，\ufeff用于标识字节序
        # 去掉\ufeff
        return _help_wrap(text, *args, **kwargs).replace("\ufeff", "")

