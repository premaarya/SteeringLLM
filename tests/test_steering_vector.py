"""Unit tests for SteeringVector class."""

import json
import tempfile
from datetime import datetime
from pathlib import Path

import pytest
import torch

from steering_llm.core.steering_vector import SteeringVector


class TestSteeringVectorCreation:
    """Tests for SteeringVector creation and validation."""
    
    def test_create_basic_vector(self) -> None:
        """Test creating a basic steering vector."""
        tensor = torch.randn(3072)
        vector = SteeringVector(
            tensor=tensor,
            layer=15,
            layer_name="model.layers.15",
            model_name="meta-llama/Llama-3.2-3B",
        )
        
        assert vector.tensor.shape == (3072,)
        assert vector.layer == 15
        assert vector.layer_name == "model.layers.15"
        assert vector.model_name == "meta-llama/Llama-3.2-3B"
        assert vector.method == "mean_difference"
        assert vector.magnitude is not None
        assert vector.magnitude > 0
        assert vector.created_at is not None
    
    def test_magnitude_auto_computed(self) -> None:
        """Test that magnitude is automatically computed."""
        tensor = torch.randn(100)
        vector = SteeringVector(
            tensor=tensor,
            layer=0,
            layer_name="model.layers.0",
            model_name="test-model",
        )
        
        expected_magnitude = float(torch.norm(tensor).item())
        assert abs(vector.magnitude - expected_magnitude) < 1e-5
    
    def test_magnitude_can_be_provided(self) -> None:
        """Test providing magnitude explicitly."""
        tensor = torch.randn(100)
        expected_mag = 5.0
        
        vector = SteeringVector(
            tensor=tensor,
            layer=0,
            layer_name="model.layers.0",
            model_name="test-model",
            magnitude=expected_mag,
        )
        
        assert vector.magnitude == expected_mag
    
    def test_created_at_auto_set(self) -> None:
        """Test that created_at is automatically set."""
        from datetime import timezone
        
        tensor = torch.randn(100)
        before = datetime.now(timezone.utc)
        
        vector = SteeringVector(
            tensor=tensor,
            layer=0,
            layer_name="model.layers.0",
            model_name="test-model",
        )
        
        after = datetime.now(timezone.utc)
        
        # Parse timestamp
        created = datetime.fromisoformat(vector.created_at)
        assert before <= created <= after
    
    def test_metadata_stored(self) -> None:
        """Test that metadata is stored correctly."""
        tensor = torch.randn(100)
        metadata = {
            "description": "Test vector",
            "tags": ["test", "example"],
            "positive_samples": 50,
        }
        
        vector = SteeringVector(
            tensor=tensor,
            layer=0,
            layer_name="model.layers.0",
            model_name="test-model",
            metadata=metadata,
        )
        
        assert vector.metadata == metadata
        assert vector.metadata["description"] == "Test vector"
        assert "test" in vector.metadata["tags"]
    
    def test_invalid_tensor_type(self) -> None:
        """Test error when tensor is not a torch.Tensor."""
        with pytest.raises(TypeError, match="tensor must be torch.Tensor"):
            SteeringVector(
                tensor=[1, 2, 3],  # type: ignore
                layer=0,
                layer_name="model.layers.0",
                model_name="test-model",
            )
    
    def test_invalid_tensor_dimensions(self) -> None:
        """Test error when tensor is not 1-dimensional."""
        with pytest.raises(ValueError, match="tensor must be 1-dimensional"):
            SteeringVector(
                tensor=torch.randn(10, 10),
                layer=0,
                layer_name="model.layers.0",
                model_name="test-model",
            )
    
    def test_invalid_layer_negative(self) -> None:
        """Test error when layer is negative."""
        with pytest.raises(ValueError, match="layer must be non-negative integer"):
            SteeringVector(
                tensor=torch.randn(100),
                layer=-1,
                layer_name="model.layers.-1",
                model_name="test-model",
            )
    
    def test_invalid_layer_type(self) -> None:
        """Test error when layer is not an integer."""
        with pytest.raises(ValueError, match="layer must be non-negative integer"):
            SteeringVector(
                tensor=torch.randn(100),
                layer="15",  # type: ignore
                layer_name="model.layers.15",
                model_name="test-model",
            )


class TestSteeringVectorProperties:
    """Tests for SteeringVector properties."""
    
    def test_shape_property(self) -> None:
        """Test shape property."""
        tensor = torch.randn(3072)
        vector = SteeringVector(
            tensor=tensor,
            layer=0,
            layer_name="model.layers.0",
            model_name="test-model",
        )
        
        assert vector.shape == torch.Size([3072])
    
    def test_dtype_property(self) -> None:
        """Test dtype property."""
        tensor = torch.randn(100, dtype=torch.float32)
        vector = SteeringVector(
            tensor=tensor,
            layer=0,
            layer_name="model.layers.0",
            model_name="test-model",
        )
        
        assert vector.dtype == torch.float32
    
    def test_device_property(self) -> None:
        """Test device property."""
        tensor = torch.randn(100)
        vector = SteeringVector(
            tensor=tensor,
            layer=0,
            layer_name="model.layers.0",
            model_name="test-model",
        )
        
        assert vector.device == torch.device("cpu")


class TestSteeringVectorDeviceMovement:
    """Tests for moving vectors between devices."""
    
    def test_to_device_cpu(self) -> None:
        """Test moving vector to CPU."""
        tensor = torch.randn(100)
        vector = SteeringVector(
            tensor=tensor,
            layer=0,
            layer_name="model.layers.0",
            model_name="test-model",
        )
        
        vector_cpu = vector.to_device("cpu")
        assert vector_cpu.device == torch.device("cpu")
        assert torch.allclose(vector.tensor, vector_cpu.tensor)
    
    def test_to_device_preserves_metadata(self) -> None:
        """Test that to_device preserves metadata."""
        tensor = torch.randn(100)
        metadata = {"test": "value"}
        
        vector = SteeringVector(
            tensor=tensor,
            layer=15,
            layer_name="model.layers.15",
            model_name="test-model",
            metadata=metadata,
        )
        
        vector_moved = vector.to_device("cpu")
        
        assert vector_moved.layer == vector.layer
        assert vector_moved.layer_name == vector.layer_name
        assert vector_moved.model_name == vector.model_name
        assert vector_moved.method == vector.method
        assert vector_moved.magnitude == vector.magnitude
        assert vector_moved.metadata == metadata
        assert vector_moved.created_at == vector.created_at
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_to_device_cuda(self) -> None:
        """Test moving vector to CUDA."""
        tensor = torch.randn(100)
        vector = SteeringVector(
            tensor=tensor,
            layer=0,
            layer_name="model.layers.0",
            model_name="test-model",
        )
        
        vector_cuda = vector.to_device("cuda")
        assert vector_cuda.device.type == "cuda"


class TestSteeringVectorValidation:
    """Tests for vector validation."""
    
    def test_validate_success(self) -> None:
        """Test validation passes for valid vector."""
        tensor = torch.randn(100)
        vector = SteeringVector(
            tensor=tensor,
            layer=0,
            layer_name="model.layers.0",
            model_name="test-model",
        )
        
        vector.validate()  # Should not raise
    
    def test_validate_expected_dim(self) -> None:
        """Test validation with expected dimension."""
        tensor = torch.randn(3072)
        vector = SteeringVector(
            tensor=tensor,
            layer=0,
            layer_name="model.layers.0",
            model_name="test-model",
        )
        
        vector.validate(expected_dim=3072)  # Should not raise
    
    def test_validate_wrong_dim(self) -> None:
        """Test validation fails for wrong dimension."""
        tensor = torch.randn(100)
        vector = SteeringVector(
            tensor=tensor,
            layer=0,
            layer_name="model.layers.0",
            model_name="test-model",
        )
        
        with pytest.raises(ValueError, match="Dimension mismatch"):
            vector.validate(expected_dim=200)
    
    def test_validate_nan_values(self) -> None:
        """Test validation fails for NaN values."""
        tensor = torch.randn(100)
        tensor[0] = float("nan")
        
        vector = SteeringVector(
            tensor=tensor,
            layer=0,
            layer_name="model.layers.0",
            model_name="test-model",
        )
        
        with pytest.raises(ValueError, match="contains NaN values"):
            vector.validate()
    
    def test_validate_inf_values(self) -> None:
        """Test validation fails for infinite values."""
        tensor = torch.randn(100)
        tensor[0] = float("inf")
        
        vector = SteeringVector(
            tensor=tensor,
            layer=0,
            layer_name="model.layers.0",
            model_name="test-model",
        )
        
        with pytest.raises(ValueError, match="contains infinite values"):
            vector.validate()


class TestSteeringVectorSaveLoad:
    """Tests for saving and loading vectors."""
    
    def test_save_creates_files(self) -> None:
        """Test that save creates both JSON and PT files."""
        tensor = torch.randn(100)
        vector = SteeringVector(
            tensor=tensor,
            layer=15,
            layer_name="model.layers.15",
            model_name="test-model",
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_vector"
            vector.save(path)
            
            assert path.with_suffix(".json").exists()
            assert path.with_suffix(".pt").exists()
    
    def test_save_json_content(self) -> None:
        """Test JSON file contains correct metadata."""
        tensor = torch.randn(100)
        metadata = {"description": "Test vector", "tags": ["test"]}
        
        vector = SteeringVector(
            tensor=tensor,
            layer=15,
            layer_name="model.layers.15",
            model_name="test-model",
            metadata=metadata,
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_vector"
            vector.save(path)
            
            with open(path.with_suffix(".json"), "r") as f:
                data = json.load(f)
            
            assert data["version"] == "1.0.0"
            assert data["model_name"] == "test-model"
            assert data["layer"] == 15
            assert data["layer_name"] == "model.layers.15"
            assert data["method"] == "mean_difference"
            assert data["magnitude"] == vector.magnitude
            assert data["shape"] == [100]
            assert data["metadata"] == metadata
    
    def test_load_vector(self) -> None:
        """Test loading a saved vector."""
        tensor = torch.randn(100)
        original = SteeringVector(
            tensor=tensor,
            layer=15,
            layer_name="model.layers.15",
            model_name="test-model",
            metadata={"test": "value"},
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_vector"
            original.save(path)
            
            loaded = SteeringVector.load(path)
            
            assert torch.allclose(loaded.tensor, original.tensor)
            assert loaded.layer == original.layer
            assert loaded.layer_name == original.layer_name
            assert loaded.model_name == original.model_name
            assert loaded.method == original.method
            assert loaded.magnitude == original.magnitude
            assert loaded.metadata == original.metadata
            assert loaded.created_at == original.created_at
    
    def test_load_missing_json(self) -> None:
        """Test error when JSON file is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nonexistent"
            
            with pytest.raises(FileNotFoundError, match="Metadata file not found"):
                SteeringVector.load(path)
    
    def test_load_missing_pt(self) -> None:
        """Test error when PT file is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_vector"
            
            # Create JSON but not PT
            metadata = {
                "version": "1.0.0",
                "model_name": "test-model",
                "layer": 0,
                "layer_name": "model.layers.0",
                "method": "mean_difference",
                "magnitude": 1.0,
                "shape": [100],
                "dtype": "float32",
                "created_at": "2026-01-28T00:00:00Z",
                "metadata": {},
            }
            
            with open(path.with_suffix(".json"), "w") as f:
                json.dump(metadata, f)
            
            with pytest.raises(FileNotFoundError, match="Tensor file not found"):
                SteeringVector.load(path)
    
    def test_load_shape_mismatch(self) -> None:
        """Test error when tensor shape doesn't match metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_vector"
            
            # Create mismatched files
            metadata = {
                "version": "1.0.0",
                "model_name": "test-model",
                "layer": 0,
                "layer_name": "model.layers.0",
                "method": "mean_difference",
                "magnitude": 1.0,
                "shape": [100],  # Says 100
                "dtype": "float32",
                "created_at": "2026-01-28T00:00:00Z",
                "metadata": {},
            }
            
            with open(path.with_suffix(".json"), "w") as f:
                json.dump(metadata, f)
            
            # Save tensor with different shape
            torch.save(torch.randn(200), path.with_suffix(".pt"))  # Actually 200
            
            with pytest.raises(ValueError, match="Shape mismatch"):
                SteeringVector.load(path)
    
    def test_roundtrip_preserves_data(self) -> None:
        """Test that save/load roundtrip preserves all data."""
        tensor = torch.randn(3072)
        metadata = {
            "description": "Production safety vector",
            "tags": ["safety", "production"],
            "positive_samples": 100,
            "negative_samples": 100,
        }
        
        original = SteeringVector(
            tensor=tensor,
            layer=15,
            layer_name="model.layers.15",
            model_name="meta-llama/Llama-3.2-3B",
            method="mean_difference",
            metadata=metadata,
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "vectors" / "safety_v1"
            original.save(path)
            loaded = SteeringVector.load(path)
            
            # Verify tensor data
            assert torch.allclose(loaded.tensor, original.tensor, atol=1e-6)
            
            # Verify all attributes
            assert loaded.layer == original.layer
            assert loaded.layer_name == original.layer_name
            assert loaded.model_name == original.model_name
            assert loaded.method == original.method
            assert abs(loaded.magnitude - original.magnitude) < 1e-6
            assert loaded.metadata == original.metadata
            assert loaded.created_at == original.created_at


class TestSteeringVectorRepr:
    """Tests for string representation."""
    
    def test_repr(self) -> None:
        """Test __repr__ output."""
        tensor = torch.randn(3072)
        vector = SteeringVector(
            tensor=tensor,
            layer=15,
            layer_name="model.layers.15",
            model_name="meta-llama/Llama-3.2-3B",
        )
        
        repr_str = repr(vector)
        
        assert "SteeringVector" in repr_str
        assert "meta-llama/Llama-3.2-3B" in repr_str
        assert "layer=15" in repr_str
        assert "(3072,)" in repr_str
        assert "mean_difference" in repr_str
