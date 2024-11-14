import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('/envs')
sys.path.append('/models')
import envs.viper as vpr
import models.models as models
import models.dqn as dqn

print(torch.cuda.is_available())