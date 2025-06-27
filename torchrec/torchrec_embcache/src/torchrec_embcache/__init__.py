import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    from embcache_pybind import *
except ImportError as e:
    print(f"Error importing embcache_pybind from {current_dir}: {e}")
    print(f"Current sys.path: {sys.path}")
    print(f"Contents of current directory: {os.listdir(current_dir)}")
    raise

from . import distributed
from . import sparse
from . import saver
