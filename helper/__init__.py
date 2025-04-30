import os
import pickle
import numpy as np
import torch
import random

from torch import nn
from torch.nn import functional
from torch.utils.data import TensorDataset, DataLoader
from transformers import CLIPProcessor, CLIPModel
from mmaction.apis import inference_recognizer, init_recognizer
from mmaction.engine.hooks import OutputHook