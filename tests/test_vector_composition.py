"""Unit tests for vector composition functionality."""

import pytest
import torch
import numpy as np

from steering_llm.core.steering_vector import SteeringVector
from steering_llm.core.vector_composition import VectorComposition


class TestWeightedSum:
    """Tests for weighted_sum composition."""
    
    def test_weighted_sum_two_vectors(self) -> None:
        """Test combining two vectors with weights."""
        vec1 = SteeringVector(
            tensor=torch.ones(100) * 2.0,
            layer=5,
            layer_name="model.layers.5",
            model_name="test-model",
            method="mean_difference",
        )
        
        vec2 = SteeringVector(
            tensor=torch.ones(100) * 4.0,
            layer=5,
            layer_name="model.layers.5",
            model_name="test-model",
            method="mean_difference",
        )
        
        combined = VectorComposition.weighted_sum(
            vectors=[vec1, vec2],
            weights=[0.5, 0.5],
        )
        
        # Expected: 0.5*2 + 0.5*4 = 3.0
        expected = torch.ones(100) * 3.0
        assert torch.allclose(combined.tensor, expected)
        assert combined.layer == 5
        assert "weighted_sum" in combined.method
    
    def test_weighted_sum_equal_weights_default(self) -> None:
        """Test that default weights are equal."""
        vec1 = SteeringVector(
            tensor=torch.ones(100) * 1.0,
            layer=5,
            layer_name="model.layers.5",
            model_name="test-model",
        )
        
        vec2 = SteeringVector(
            tensor=torch.ones(100) * 3.0,
            layer=5,
            layer_name="model.layers.5",
            model_name="test-model",
        )
        
        combined = VectorComposition.weighted_sum(
            vectors=[vec1, vec2],
        )
        
        # Expected: 1*1 + 1*3 = 4.0
        expected = torch.ones(100) * 4.0
        assert torch.allclose(combined.tensor, expected)
    
    def test_weighted_sum_normalization(self) -> None:
        """Test that normalization produces unit vector."""
        vec1 = SteeringVector(
            tensor=torch.randn(100),
            layer=5,
            layer_name="model.layers.5",
            model_name="test-model",
        )
        
        vec2 = SteeringVector(
            tensor=torch.randn(100),
            layer=5,
            layer_name="model.layers.5",
            model_name="test-model",
        )
        
        combined = VectorComposition.weighted_sum(
            vectors=[vec1, vec2],
            weights=[0.7, 0.3],
            normalize=True,
        )
        
        # Check that result is normalized
        norm = torch.norm(combined.tensor)
        assert torch.isclose(norm, torch.tensor(1.0), atol=1e-5)
    
    def test_weighted_sum_multiple_vectors(self) -> None:
        """Test combining 5+ vectors."""
        vectors = []
        for i in range(5):
            vec = SteeringVector(
                tensor=torch.ones(50) * (i + 1),
                layer=10,
                layer_name="model.layers.10",
                model_name="test-model",
            )
            vectors.append(vec)
        
        weights = [0.1, 0.2, 0.3, 0.2, 0.2]
        combined = VectorComposition.weighted_sum(
            vectors=vectors,
            weights=weights,
        )
        
        # Expected: 0.1*1 + 0.2*2 + 0.3*3 + 0.2*4 + 0.2*5 = 3.2
        expected = torch.ones(50) * 3.2
        assert torch.allclose(combined.tensor, expected, atol=1e-5)
        assert combined.metadata["num_vectors"] == 5
    
    def test_weighted_sum_empty_vectors_error(self) -> None:
        """Test error when vectors list is empty."""
        with pytest.raises(ValueError, match="vectors list cannot be empty"):
            VectorComposition.weighted_sum(vectors=[])
    
    def test_weighted_sum_mismatched_weights_error(self) -> None:
        """Test error when number of weights doesn't match vectors."""
        vec1 = SteeringVector(
            tensor=torch.randn(100),
            layer=5,
            layer_name="model.layers.5",
            model_name="test-model",
        )
        
        vec2 = SteeringVector(
            tensor=torch.randn(100),
            layer=5,
            layer_name="model.layers.5",
            model_name="test-model",
        )
        
        with pytest.raises(ValueError, match="Number of weights.*must match"):
            VectorComposition.weighted_sum(
                vectors=[vec1, vec2],
                weights=[1.0],  # Only 1 weight for 2 vectors
            )
    
    def test_weighted_sum_different_layers_error(self) -> None:
        """Test error when vectors target different layers."""
        vec1 = SteeringVector(
            tensor=torch.randn(100),
            layer=5,
            layer_name="model.layers.5",
            model_name="test-model",
        )
        
        vec2 = SteeringVector(
            tensor=torch.randn(100),
            layer=10,  # Different layer
            layer_name="model.layers.10",
            model_name="test-model",
        )
        
        with pytest.raises(ValueError, match="All vectors must target the same layer"):
            VectorComposition.weighted_sum(vectors=[vec1, vec2])
    
    def test_weighted_sum_different_shapes_error(self) -> None:
        """Test error when vectors have different shapes."""
        vec1 = SteeringVector(
            tensor=torch.randn(100),
            layer=5,
            layer_name="model.layers.5",
            model_name="test-model",
        )
        
        vec2 = SteeringVector(
            tensor=torch.randn(200),  # Different shape
            layer=5,
            layer_name="model.layers.5",
            model_name="test-model",
        )
        
        with pytest.raises(ValueError, match="All vectors must have the same shape"):
            VectorComposition.weighted_sum(vectors=[vec1, vec2])


class TestComputeSimilarity:
    """Tests for compute_similarity function."""
    
    def test_similarity_identical_vectors(self) -> None:
        """Test that identical vectors have similarity 1.0."""
        vec = SteeringVector(
            tensor=torch.randn(100),
            layer=5,
            layer_name="model.layers.5",
            model_name="test-model",
        )
        
        similarity = VectorComposition.compute_similarity(vec, vec)
        assert abs(similarity - 1.0) < 1e-5
    
    def test_similarity_opposite_vectors(self) -> None:
        """Test that opposite vectors have similarity -1.0."""
        tensor = torch.randn(100)
        
        vec1 = SteeringVector(
            tensor=tensor,
            layer=5,
            layer_name="model.layers.5",
            model_name="test-model",
        )
        
        vec2 = SteeringVector(
            tensor=-tensor,  # Opposite direction
            layer=5,
            layer_name="model.layers.5",
            model_name="test-model",
        )
        
        similarity = VectorComposition.compute_similarity(vec1, vec2)
        assert abs(similarity - (-1.0)) < 1e-5
    
    def test_similarity_orthogonal_vectors(self) -> None:
        """Test that orthogonal vectors have similarity ~0.0."""
        # Create orthogonal vectors
        vec1 = SteeringVector(
            tensor=torch.tensor([1.0, 0.0, 0.0, 0.0]),
            layer=5,
            layer_name="model.layers.5",
            model_name="test-model",
        )
        
        vec2 = SteeringVector(
            tensor=torch.tensor([0.0, 1.0, 0.0, 0.0]),
            layer=5,
            layer_name="model.layers.5",
            model_name="test-model",
        )
        
        similarity = VectorComposition.compute_similarity(vec1, vec2)
        assert abs(similarity) < 1e-5
    
    def test_similarity_different_shapes_error(self) -> None:
        """Test error when vectors have different shapes."""
        vec1 = SteeringVector(
            tensor=torch.randn(100),
            layer=5,
            layer_name="model.layers.5",
            model_name="test-model",
        )
        
        vec2 = SteeringVector(
            tensor=torch.randn(200),
            layer=5,
            layer_name="model.layers.5",
            model_name="test-model",
        )
        
        with pytest.raises(ValueError, match="Vectors must have the same shape"):
            VectorComposition.compute_similarity(vec1, vec2)


class TestDetectConflicts:
    """Tests for detect_conflicts function."""
    
    def test_detect_conflicts_high_correlation(self) -> None:
        """Test detection of highly correlated vectors."""
        # Create highly correlated vectors
        base_tensor = torch.randn(100)
        
        vec1 = SteeringVector(
            tensor=base_tensor,
            layer=5,
            layer_name="model.layers.5",
            model_name="test-model",
        )
        
        vec2 = SteeringVector(
            tensor=base_tensor + torch.randn(100) * 0.1,  # Slightly perturbed
            layer=5,
            layer_name="model.layers.5",
            model_name="test-model",
        )
        
        conflicts = VectorComposition.detect_conflicts([vec1, vec2], threshold=0.7)
        
        # Should detect conflict
        assert len(conflicts) == 1
        assert conflicts[0][0] == 0
        assert conflicts[0][1] == 1
        assert conflicts[0][2] > 0.7
    
    def test_detect_conflicts_anti_correlation(self) -> None:
        """Test detection of anti-correlated vectors."""
        tensor = torch.randn(100)
        
        vec1 = SteeringVector(
            tensor=tensor,
            layer=5,
            layer_name="model.layers.5",
            model_name="test-model",
        )
        
        vec2 = SteeringVector(
            tensor=-tensor,  # Opposite
            layer=5,
            layer_name="model.layers.5",
            model_name="test-model",
        )
        
        conflicts = VectorComposition.detect_conflicts([vec1, vec2], threshold=0.7)
        
        # Should detect conflict (similarity = -1.0)
        assert len(conflicts) == 1
        assert conflicts[0][2] < -0.7
    
    def test_detect_conflicts_no_conflicts(self) -> None:
        """Test that orthogonal vectors have no conflicts."""
        # Create orthogonal vectors
        vec1 = SteeringVector(
            tensor=torch.tensor([1.0, 0.0, 0.0, 0.0]),
            layer=5,
            layer_name="model.layers.5",
            model_name="test-model",
        )
        
        vec2 = SteeringVector(
            tensor=torch.tensor([0.0, 1.0, 0.0, 0.0]),
            layer=5,
            layer_name="model.layers.5",
            model_name="test-model",
        )
        
        vec3 = SteeringVector(
            tensor=torch.tensor([0.0, 0.0, 1.0, 0.0]),
            layer=5,
            layer_name="model.layers.5",
            model_name="test-model",
        )
        
        conflicts = VectorComposition.detect_conflicts([vec1, vec2, vec3], threshold=0.7)
        
        # No conflicts expected
        assert len(conflicts) == 0
    
    def test_detect_conflicts_multiple_pairs(self) -> None:
        """Test detection with multiple conflicting pairs."""
        # Create 3 vectors where vec1~vec2 and vec2~vec3
        base = torch.randn(100)
        
        vec1 = SteeringVector(
            tensor=base,
            layer=5,
            layer_name="model.layers.5",
            model_name="test-model",
        )
        
        vec2 = SteeringVector(
            tensor=base + torch.randn(100) * 0.05,
            layer=5,
            layer_name="model.layers.5",
            model_name="test-model",
        )
        
        vec3 = SteeringVector(
            tensor=base + torch.randn(100) * 0.05,
            layer=5,
            layer_name="model.layers.5",
            model_name="test-model",
        )
        
        conflicts = VectorComposition.detect_conflicts([vec1, vec2, vec3], threshold=0.7)
        
        # Should detect multiple conflicts
        assert len(conflicts) >= 2  # At least 2 pairs


class TestOrthogonalize:
    """Tests for orthogonalize function."""
    
    def test_orthogonalize_two_vectors(self) -> None:
        """Test orthogonalizing two vectors."""
        vec1 = SteeringVector(
            tensor=torch.tensor([1.0, 0.0, 0.0]),
            layer=5,
            layer_name="model.layers.5",
            model_name="test-model",
        )
        
        vec2 = SteeringVector(
            tensor=torch.tensor([1.0, 1.0, 0.0]),  # Not orthogonal to vec1
            layer=5,
            layer_name="model.layers.5",
            model_name="test-model",
        )
        
        ortho_vecs = VectorComposition.orthogonalize([vec1, vec2])
        
        # Check that result has 2 vectors
        assert len(ortho_vecs) == 2
        
        # Check that they are orthogonal
        similarity = VectorComposition.compute_similarity(ortho_vecs[0], ortho_vecs[1])
        assert abs(similarity) < 1e-5
        
        # Check that first vector is normalized
        norm1 = torch.norm(ortho_vecs[0].tensor)
        assert torch.isclose(norm1, torch.tensor(1.0), atol=1e-5)
        
        # Check that second vector is normalized
        norm2 = torch.norm(ortho_vecs[1].tensor)
        assert torch.isclose(norm2, torch.tensor(1.0), atol=1e-5)
    
    def test_orthogonalize_three_vectors(self) -> None:
        """Test orthogonalizing three vectors."""
        vec1 = SteeringVector(
            tensor=torch.tensor([1.0, 0.0, 0.0, 0.0]),
            layer=5,
            layer_name="model.layers.5",
            model_name="test-model",
        )
        
        vec2 = SteeringVector(
            tensor=torch.tensor([1.0, 1.0, 0.0, 0.0]),
            layer=5,
            layer_name="model.layers.5",
            model_name="test-model",
        )
        
        vec3 = SteeringVector(
            tensor=torch.tensor([1.0, 1.0, 1.0, 0.0]),
            layer=5,
            layer_name="model.layers.5",
            model_name="test-model",
        )
        
        ortho_vecs = VectorComposition.orthogonalize([vec1, vec2, vec3])
        
        # Check that all pairs are orthogonal
        for i in range(3):
            for j in range(i + 1, 3):
                similarity = VectorComposition.compute_similarity(ortho_vecs[i], ortho_vecs[j])
                assert abs(similarity) < 1e-4, f"Vectors {i} and {j} not orthogonal: {similarity}"
    
    def test_orthogonalize_preserves_metadata(self) -> None:
        """Test that orthogonalization preserves important metadata."""
        vec1 = SteeringVector(
            tensor=torch.randn(100),
            layer=5,
            layer_name="model.layers.5",
            model_name="test-model",
            method="mean_difference",
        )
        
        vec2 = SteeringVector(
            tensor=torch.randn(100),
            layer=5,
            layer_name="model.layers.5",
            model_name="test-model",
            method="caa",
        )
        
        ortho_vecs = VectorComposition.orthogonalize([vec1, vec2])
        
        # Check that layer info is preserved
        assert ortho_vecs[0].layer == 5
        assert ortho_vecs[1].layer == 5
        
        # Check that metadata indicates orthogonalization
        assert "orthogonalized" in ortho_vecs[0].method
        assert "orthogonalized" in ortho_vecs[1].method
        
        assert ortho_vecs[0].metadata["original_method"] == "mean_difference"
        assert ortho_vecs[1].metadata["original_method"] == "caa"
    
    def test_orthogonalize_empty_error(self) -> None:
        """Test error when vectors list is empty."""
        with pytest.raises(ValueError, match="vectors list cannot be empty"):
            VectorComposition.orthogonalize([])
    
    def test_orthogonalize_different_shapes_error(self) -> None:
        """Test error when vectors have different shapes."""
        vec1 = SteeringVector(
            tensor=torch.randn(100),
            layer=5,
            layer_name="model.layers.5",
            model_name="test-model",
        )
        
        vec2 = SteeringVector(
            tensor=torch.randn(200),
            layer=5,
            layer_name="model.layers.5",
            model_name="test-model",
        )
        
        with pytest.raises(ValueError, match="All vectors must have the same shape"):
            VectorComposition.orthogonalize([vec1, vec2])
    
    def test_orthogonalize_different_layers_error(self) -> None:
        """Test error when vectors target different layers."""
        vec1 = SteeringVector(
            tensor=torch.randn(100),
            layer=5,
            layer_name="model.layers.5",
            model_name="test-model",
        )
        
        vec2 = SteeringVector(
            tensor=torch.randn(100),
            layer=10,
            layer_name="model.layers.10",
            model_name="test-model",
        )
        
        with pytest.raises(ValueError, match="All vectors must target the same layer"):
            VectorComposition.orthogonalize([vec1, vec2])


class TestAnalyzeComposition:
    """Tests for analyze_composition function."""
    
    def test_analyze_empty_vectors(self) -> None:
        """Test analysis with empty vector list."""
        analysis = VectorComposition.analyze_composition([])
        
        assert analysis["num_vectors"] == 0
        assert "summary" in analysis
    
    def test_analyze_single_vector(self) -> None:
        """Test analysis with single vector."""
        vec = SteeringVector(
            tensor=torch.randn(100),
            layer=5,
            layer_name="model.layers.5",
            model_name="test-model",
        )
        
        analysis = VectorComposition.analyze_composition([vec])
        
        assert analysis["num_vectors"] == 1
        assert len(analysis["magnitudes"]) == 1
        assert "avg_magnitude" in analysis
        assert "recommendations" in analysis
    
    def test_analyze_conflicting_vectors(self) -> None:
        """Test analysis detects conflicts."""
        # Create highly correlated vectors
        base = torch.randn(100)
        
        vec1 = SteeringVector(
            tensor=base,
            layer=5,
            layer_name="model.layers.5",
            model_name="test-model",
        )
        
        vec2 = SteeringVector(
            tensor=base + torch.randn(100) * 0.05,
            layer=5,
            layer_name="model.layers.5",
            model_name="test-model",
        )
        
        analysis = VectorComposition.analyze_composition([vec1, vec2])
        
        assert analysis["num_vectors"] == 2
        assert len(analysis["conflicts"]) > 0
        assert any("conflict" in rec.lower() for rec in analysis["recommendations"])
    
    def test_analyze_independent_vectors(self) -> None:
        """Test analysis with independent vectors."""
        # Create orthogonal vectors
        vec1 = SteeringVector(
            tensor=torch.tensor([1.0, 0.0, 0.0, 0.0]),
            layer=5,
            layer_name="model.layers.5",
            model_name="test-model",
        )
        
        vec2 = SteeringVector(
            tensor=torch.tensor([0.0, 1.0, 0.0, 0.0]),
            layer=5,
            layer_name="model.layers.5",
            model_name="test-model",
        )
        
        analysis = VectorComposition.analyze_composition([vec1, vec2])
        
        assert analysis["num_vectors"] == 2
        assert len(analysis["conflicts"]) == 0
        assert any("independent" in rec.lower() for rec in analysis["recommendations"])
    
    def test_analyze_similarity_matrix(self) -> None:
        """Test that similarity matrix has correct structure."""
        vec1 = SteeringVector(
            tensor=torch.randn(100),
            layer=5,
            layer_name="model.layers.5",
            model_name="test-model",
        )
        
        vec2 = SteeringVector(
            tensor=torch.randn(100),
            layer=5,
            layer_name="model.layers.5",
            model_name="test-model",
        )
        
        vec3 = SteeringVector(
            tensor=torch.randn(100),
            layer=5,
            layer_name="model.layers.5",
            model_name="test-model",
        )
        
        analysis = VectorComposition.analyze_composition([vec1, vec2, vec3])
        
        # Check similarity matrix shape
        sim_matrix = np.array(analysis["similarity_matrix"])
        assert sim_matrix.shape == (3, 3)
        
        # Check diagonal is 1.0
        assert np.allclose(np.diag(sim_matrix), 1.0)
        
        # Check symmetry
        assert np.allclose(sim_matrix, sim_matrix.T)
