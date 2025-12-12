from collections import deque

import torch
import pyarrow.parquet as pq

from nanochat.common import get_dist_info