"""
Train model. Run as:

python base_train.py

or distributed as:

torchrun --nproc_per_node=8 base_train.py

If you are only on CPU/Macbook, you'll want to train a much much smaller LLM. Example:
python -m scripts.base_train --depth=4 --max_seq_len=512 --device_batch_size=1 --eval_tokens=512 --core_metric_every=-1 --total_batch_size=512 --num_iterations=20
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import time
from contextlib import nullcontext

import wandb
import torch

from nanochat.gpt import GPT, GPTConfig

# ---------------------------------------------------------------
# user settings
run = "dummy" # wandb run name defaults to "dummy" for no wandb logging
# Runtime
device_type = "" # cuda|mps|cpu, "" selects cuda if available in order: cuda > mps > cpu
# Model architecture
depth = 20 # the depth of the transformer and rest of the model is derived from this
max_seq_len = 2048 # max context length
# Training horizon. Only one of these 3 will be used, in this order of precedence.
num_iterations = -1 # explicit number of steps of the optimization (-1 = disable)
target_flops = -1.0 # calculate num_iterations to reach target_flops. Useful for scaling laws experiments (-1 = disable)
target_param_data_ratio = 20 # calculate num_iterations to maintain fixed data:param ratio (Chinchilla=20) (-1 = disable)
# Optimization hyperparameters
device_batch_size = 32 # per-device batch size (set to not OOM)
total_batch_size = 524288 # total batch size (across all devices) in tokens
embeddingZ_lr = 0.2 # learning reate for the embedding parameters (Adam)
unembedding_lr = 0.004 # learning rate for the unembedding parameters (Adam)
weight_decay = 0.0 # weight decay for AdamW
matrix_lr = 0.02 # learning rate for the weight matrices (SGD)
grad_clip = 1.0 # gradient clipping
warmup_ratio = 0.0 # fraction of total steps used for learning rate warmup
warmdown_ratio = 0.2 # ratio of total steps used for learning rate warmdown
final_lr_ratio = 0.0 # final learning rate as a ratio of initial learning rate
resume_from_step = -1 # resume training from checkpoint at this step (-1 = disable)
# Evaluation settings
