import logging
import os
import sys


logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    from . import embcache_pybind
    logging.debug("Successfully imported embcache_pybind from %s", current_dir)
except ImportError as e:
    logging.error("Error importing embcache_pybind from %s: %s", current_dir, e)
    logging.debug("Current sys.path: %s", sys.path)
    logging.debug("Contents of current directory: %s", os.listdir(current_dir))
    raise

from . import distributed
from . import sparse
from . import saver
