import logging


def get_logger():
    wide_deep = logging.getLogger("wide_deep")
    formatter = logging.Formatter(fmt="[%(asctime)s] [%(levelname)s] %(message)s",
                                  datefmt="%m/%d/%Y %H:%M:%S %p")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    wide_deep.addHandler(stream_handler)
    wide_deep.setLevel("DEBUG")
    return wide_deep

logger = get_logger()