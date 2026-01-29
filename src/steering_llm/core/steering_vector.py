"""
SteeringVector class for storing and managing steering vectors.

This module implements the core SteeringVector data structure that holds
steering vector tensors along with their metadata.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch


@dataclass
class SteeringVector:
    """
    A steering vector for modifying LLM behavior.
    
    Stores a tensor that can be added to model activations at a specific layer
    to steer the model's behavior in a desired direction.
    
    Attributes:
        tensor: The steering vector tensor (shape: [hidden_dim])
        layer: Target layer index
        layer_name: Target layer name (e.g., "model.layers.15")
        model_name: HuggingFace model identifier
        method: Discovery method used (e.g., "mean_difference")
        magnitude: L2 norm of the vector
        metadata: Additional metadata (description, tags, etc.)
        created_at: Timestamp when vector was created
    """
    
    tensor: torch.Tensor
    layer: int
    layer_name: str
    model_name: str
    method: str = "mean_difference"
    magnitude: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[str] = None
    
    def __post_init__(self) -> None:
        """Validate and compute derived attributes."""
        # Validate tensor
        if not isinstance(self.tensor, torch.Tensor):
            raise TypeError(f"tensor must be torch.Tensor, got {type(self.tensor)}")
        
        if self.tensor.ndim != 1:
            raise ValueError(f"tensor must be 1-dimensional, got shape {self.tensor.shape}")
        
        # Validate layer
        if not isinstance(self.layer, int) or self.layer < 0:
            raise ValueError(f"layer must be non-negative integer, got {self.layer}")
        
        # Compute magnitude if not provided
        if self.magnitude is None:
            self.magnitude = float(torch.norm(self.tensor).item())
        
        # Set creation timestamp if not provided
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc).isoformat()
    
    @property
    def shape(self) -> torch.Size:
        """Get the shape of the steering vector."""
        return self.tensor.shape
    
    @property
    def dtype(self) -> torch.dtype:
        """Get the dtype of the steering vector."""
        return self.tensor.dtype
    
    @property
    def device(self) -> torch.device:
        """Get the device of the steering vector."""
        return self.tensor.device
    
    def to_device(self, device: Union[str, torch.device]) -> "SteeringVector":
        """
        Move steering vector to specified device.
        
        Args:
            device: Target device (e.g., "cuda", "cpu", "mps")
        
        Returns:
            New SteeringVector instance on target device
        """
        if isinstance(device, str):
            device = torch.device(device)
        
        return SteeringVector(
            tensor=self.tensor.to(device),
            layer=self.layer,
            layer_name=self.layer_name,
            model_name=self.model_name,
            method=self.method,
            magnitude=self.magnitude,
            metadata=self.metadata.copy(),
            created_at=self.created_at,
        )
    
    def validate(self, expected_dim: Optional[int] = None) -> None:
        """
        Validate steering vector integrity.
        
        Args:
            expected_dim: Expected hidden dimension (optional)
        
        Raises:
            ValueError: If validation fails
        """
        # Check magnitude consistency
        computed_magnitude = float(torch.norm(self.tensor).item())
        if self.magnitude is not None:
            magnitude_diff = abs(computed_magnitude - self.magnitude)
            if magnitude_diff > 1e-3:
                raise ValueError(
                    f"Magnitude mismatch: stored={self.magnitude:.4f}, "
                    f"computed={computed_magnitude:.4f}"
                )
        
        # Check expected dimension
        if expected_dim is not None:
            actual_dim = self.tensor.shape[0]
            if actual_dim != expected_dim:
                raise ValueError(
                    f"Dimension mismatch: expected={expected_dim}, "
                    f"actual={actual_dim}"
                )
        
        # Check for NaN or Inf
        if torch.isnan(self.tensor).any():
            raise ValueError("Steering vector contains NaN values")
        
        if torch.isinf(self.tensor).any():
            raise ValueError("Steering vector contains infinite values")
    
    def save(self, path: Union[str, Path]) -> None:
        """
        Save steering vector to disk.
        
        Creates two files:
        - {path}.json: Human-readable metadata
        - {path}.pt: Efficient tensor storage
        
        Args:
            path: Output path (without extension)
        
        Example:
            >>> vector.save("vectors/safety_v1")
            # Creates: vectors/safety_v1.json and vectors/safety_v1.pt
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save metadata as JSON
        metadata_dict = {
            "version": "1.0.0",
            "model_name": self.model_name,
            "layer": self.layer,
            "layer_name": self.layer_name,
            "method": self.method,
            "magnitude": self.magnitude,
            "shape": list(self.tensor.shape),
            "dtype": str(self.tensor.dtype).replace("torch.", ""),
            "created_at": self.created_at,
            "metadata": self.metadata,
        }
        
        json_path = path.with_suffix(".json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metadata_dict, f, indent=2)
        
        # Save tensor as PyTorch file
        pt_path = path.with_suffix(".pt")
        torch.save(self.tensor, pt_path)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "SteeringVector":
        """
        Load steering vector from disk.
        
        Args:
            path: Input path (without extension)
        
        Returns:
            Loaded SteeringVector instance
        
        Raises:
            FileNotFoundError: If files don't exist
            ValueError: If files are corrupted or incompatible
        
        Example:
            >>> vector = SteeringVector.load("vectors/safety_v1")
        """
        path = Path(path)
        
        # Load metadata
        json_path = path.with_suffix(".json")
        if not json_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {json_path}")
        
        with open(json_path, "r", encoding="utf-8") as f:
            metadata_dict = json.load(f)
        
        # Load tensor
        pt_path = path.with_suffix(".pt")
        if not pt_path.exists():
            raise FileNotFoundError(f"Tensor file not found: {pt_path}")
        
        tensor = torch.load(pt_path, map_location="cpu")
        
        # Validate loaded data
        expected_shape = tuple(metadata_dict["shape"])
        if tensor.shape != expected_shape:
            raise ValueError(
                f"Shape mismatch: metadata={expected_shape}, "
                f"tensor={tensor.shape}"
            )
        
        # Create SteeringVector instance
        return cls(
            tensor=tensor,
            layer=metadata_dict["layer"],
            layer_name=metadata_dict["layer_name"],
            model_name=metadata_dict["model_name"],
            method=metadata_dict["method"],
            magnitude=metadata_dict["magnitude"],
            metadata=metadata_dict.get("metadata", {}),
            created_at=metadata_dict["created_at"],
        )
    
    def __repr__(self) -> str:
        """String representation of SteeringVector."""
        return (
            f"SteeringVector(model={self.model_name}, layer={self.layer}, "
            f"shape={tuple(self.shape)}, magnitude={self.magnitude:.4f}, "
            f"method={self.method})"
        )
