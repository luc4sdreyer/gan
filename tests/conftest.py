import random
import pytest

import numpy as np

@pytest.fixture
def reset_random_seed():
    random.seed(0)
    np.random.seed(0)
