import logging

def get_logger():
    dlrm_logger = logging.getLogger("dlrm")
    formatter = logging.Formatter(fmt="[%(asctime)s] [%(levelname)s] %(message)s",
                                  datefmt="%m/%d/%Y %H:%M:%S %p")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    dlrm_logger.addHandler(stream_handler)
    dlrm_logger.setLevel("DEBUG")
    return dlrm_logger

logger = get_logger()