import warnings
from env.builder import *

try:
    from env.gym_env.gym_env import *
except:
    warnings.warn("gym is not installed!")

try:
    from env.starcraft2_env.starcraft2_env import *
except:
    warnings.warn("smac is not installed!")