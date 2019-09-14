
import logging
import os
import pandas as pd
import numpy as np

from common_interface import model
from model_9.utils import SquadRunner

from sklearn.metrics import log_loss

import os 
dir_path = os.path.dirname(os.path.realpath(__file__))

print(dir_path)


from hltproject.score.score import compute_loss


compute_loss("stage1_swag_only_my_w.csv","gap-test.tsv")


