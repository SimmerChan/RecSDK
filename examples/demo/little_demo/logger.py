import logging


def get_logger():
    demo_logger = logging.getLogger("little_demo")
    formatter = logging.Formatter(fmt="[%(asctime)s] [%(levelname)s] %(message)s",
                                  datefmt="%m/%d/%Y %H:%M:%S %p")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    demo_logger.addHandler(stream_handler)
    demo_logger.setLevel("DEBUG")
    return demo_logger

logger = get_logger()