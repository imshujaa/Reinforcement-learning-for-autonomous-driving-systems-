"""Test configuration for pytest."""

import pytest
import numpy as np


@pytest.fixture
def random_seed():
    """Set random seed for reproducibility."""
    np.random.seed(42)
    yield 42


@pytest.fixture
def sample_state():
    """Create sample state for testing."""
    return np.random.randn(10)


@pytest.fixture
def sample_action():
    """Create sample action for testing."""
    return 2
