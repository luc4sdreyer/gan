import random
import pytest

@pytest.fixture
def reset_random_seed():
    random.seed(0)
