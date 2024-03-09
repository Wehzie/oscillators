"""
This module defines constants used throughout the project.

Some constants need to be frequently changed during development and testing and would be better replaced by command line interface.
"""

from typing import Final
import numpy as np

# a good sampling rate to sample a signal with max frequency f
# is 2*f*OVERSAMPLING_FACTOR, where 2*f is the Nyquist rate
OVERSAMPLING_FACTOR: Final[int] = 10 
GLOBAL_SEED = 5
RNG: Final = np.random.default_rng(GLOBAL_SEED)