"""Pytest configuration and fixtures."""

import pytest
import torch


@pytest.fixture
def sample_tensor():
    """Provide a sample tensor for testing."""
    return torch.randn(100)


@pytest.fixture
def sample_steering_vector():
    """Provide a sample SteeringVector for testing."""
    from steering_llm.core.steering_vector import SteeringVector
    
    return SteeringVector(
        tensor=torch.randn(3072),
        layer=15,
        layer_name="model.layers.15",
        model_name="meta-llama/Llama-3.2-3B",
        metadata={
            "description": "Test vector",
            "positive_samples": 50,
            "negative_samples": 50,
        },
    )


@pytest.fixture
def temp_vector_path(tmp_path):
    """Provide a temporary path for saving vectors."""
    return tmp_path / "test_vector"


# Configure pytest to show full diffs
def pytest_configure(config):
    """Configure pytest options."""
    config.option.verbose = 2
