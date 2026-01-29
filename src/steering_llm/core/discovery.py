"""
Discovery module for creating steering vectors from contrast datasets.

This module provides methods for discovering steering vectors by analyzing
differences in model activations between positive and negative examples.
"""

from typing import Any, Dict, List, Optional, Union

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer, AutoTokenizer

from steering_llm.core.steering_vector import SteeringVector


class Discovery:
    """
    Methods for discovering steering vectors from contrast datasets.
    
    This class provides static methods for creating steering vectors by
    analyzing model activations on positive vs negative examples.
    """
    
    @staticmethod
    def mean_difference(
        positive: List[str],
        negative: List[str],
        model: PreTrainedModel,
        layer: int,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        model_name: Optional[str] = None,
        batch_size: int = 8,
        max_length: int = 128,
        device: Optional[Union[str, torch.device]] = None,
    ) -> SteeringVector:
        """
        Create steering vector using mean difference method.
        
        Computes: mean(positive_activations) - mean(negative_activations)
        
        This method extracts activations from the specified layer for both
        positive and negative examples, then computes the difference of their
        means to create a steering vector.
        
        Args:
            positive: List of texts exhibiting desired behavior
            negative: List of texts exhibiting undesired behavior
            model: HuggingFace model to extract activations from
            layer: Target layer index (0-based)
            tokenizer: Tokenizer (auto-detected if None)
            model_name: Model identifier (auto-detected if None)
            batch_size: Batch size for processing examples
            max_length: Maximum sequence length
            device: Device to use (auto-detected if None)
        
        Returns:
            SteeringVector instance
        
        Raises:
            ValueError: If inputs are invalid or layer doesn't exist
            RuntimeError: If activation extraction fails
        
        Example:
            >>> from transformers import AutoModelForCausalLM
            >>> model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B")
            >>> vector = Discovery.mean_difference(
            ...     positive=["I love helping!", "You're amazing!"],
            ...     negative=["I hate this.", "You're terrible."],
            ...     model=model,
            ...     layer=15
            ... )
        """
        # Validate inputs
        if not positive:
            raise ValueError("positive examples list cannot be empty")
        if not negative:
            raise ValueError("negative examples list cannot be empty")
        
        if not isinstance(layer, int) or layer < 0:
            raise ValueError(f"layer must be non-negative integer, got {layer}")
        
        # Auto-detect tokenizer if not provided
        if tokenizer is None:
            if model_name is None:
                model_name = getattr(model.config, "_name_or_path", "unknown")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        
        # Auto-detect device if not provided
        if device is None:
            device = next(model.parameters()).device
        elif isinstance(device, str):
            device = torch.device(device)
        
        # Auto-detect model name if not provided
        if model_name is None:
            model_name = getattr(model.config, "_name_or_path", "unknown")
        
        # Detect layer name
        layer_name = Discovery._detect_layer_name(model, layer)
        
        # Extract activations for positive examples
        print(f"Extracting activations for {len(positive)} positive examples...")
        pos_activations = Discovery._extract_activations(
            texts=positive,
            model=model,
            tokenizer=tokenizer,
            layer=layer,
            layer_name=layer_name,
            batch_size=batch_size,
            max_length=max_length,
            device=device,
        )
        
        # Extract activations for negative examples
        print(f"Extracting activations for {len(negative)} negative examples...")
        neg_activations = Discovery._extract_activations(
            texts=negative,
            model=model,
            tokenizer=tokenizer,
            layer=layer,
            layer_name=layer_name,
            batch_size=batch_size,
            max_length=max_length,
            device=device,
        )
        
        # Compute mean difference
        mean_pos = torch.mean(pos_activations, dim=0)
        mean_neg = torch.mean(neg_activations, dim=0)
        steering_vector = mean_pos - mean_neg
        
        # Create SteeringVector instance
        return SteeringVector(
            tensor=steering_vector.cpu(),
            layer=layer,
            layer_name=layer_name,
            model_name=model_name,
            method="mean_difference",
            metadata={
                "positive_samples": len(positive),
                "negative_samples": len(negative),
                "batch_size": batch_size,
                "max_length": max_length,
            },
        )
    
    @staticmethod
    def _detect_layer_name(model: PreTrainedModel, layer: int) -> str:
        """
        Detect layer name from model architecture.
        
        Args:
            model: HuggingFace model
            layer: Layer index
        
        Returns:
            Layer name (e.g., "model.layers.15")
        
        Raises:
            ValueError: If layer cannot be detected
        """
        # Common patterns for different architectures
        patterns = [
            f"model.layers.{layer}",  # Llama, Mistral, Gemma
            f"transformer.h.{layer}",  # GPT-2
            f"model.decoder.layers.{layer}",  # OPT
        ]
        
        for pattern in patterns:
            if hasattr(model, "model") and hasattr(model.model, "layers"):
                if layer < len(model.model.layers):
                    return f"model.layers.{layer}"
        
        raise ValueError(
            f"Cannot detect layer name for layer {layer}. "
            f"Supported architectures: Llama, Mistral, Gemma"
        )
    
    @staticmethod
    def _extract_activations(
        texts: List[str],
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        layer: int,
        layer_name: str,
        batch_size: int,
        max_length: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Extract activations from specified layer for given texts.
        
        Args:
            texts: Input texts
            model: HuggingFace model
            tokenizer: Tokenizer
            layer: Layer index
            layer_name: Layer name
            batch_size: Batch size
            max_length: Max sequence length
            device: Device to use
        
        Returns:
            Stacked activations tensor [num_texts, hidden_dim]
        
        Raises:
            RuntimeError: If extraction fails
        """
        activations_list = []
        
        # Get target layer module
        target_module = Discovery._get_layer_module(model, layer_name)
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Storage for activations
            batch_activations = []
            
            # Define hook function
            def hook_fn(module: torch.nn.Module, input: Any, output: Any) -> None:
                # Extract activation (usually a tuple with tensor as first element)
                if isinstance(output, tuple):
                    activation = output[0]
                else:
                    activation = output
                
                # Take mean over sequence dimension [batch, seq_len, hidden_dim] -> [batch, hidden_dim]
                activation_mean = activation.mean(dim=1)
                batch_activations.append(activation_mean.detach())
            
            # Register hook
            handle = target_module.register_forward_hook(hook_fn)
            
            try:
                # Forward pass
                with torch.no_grad():
                    model(**inputs)
                
                # Collect activations
                if not batch_activations:
                    raise RuntimeError(f"No activations captured from layer {layer_name}")
                
                activations_list.append(batch_activations[0])
            
            finally:
                # Remove hook
                handle.remove()
        
        # Stack all activations
        all_activations = torch.cat(activations_list, dim=0)
        return all_activations
    
    @staticmethod
    def _get_layer_module(model: PreTrainedModel, layer_name: str) -> torch.nn.Module:
        """
        Get layer module by name.
        
        Args:
            model: HuggingFace model
            layer_name: Layer name (e.g., "model.layers.15")
        
        Returns:
            Layer module
        
        Raises:
            ValueError: If layer not found
        """
        parts = layer_name.split(".")
        module = model
        
        for part in parts:
            if not hasattr(module, part):
                # Try integer indexing
                try:
                    idx = int(part)
                    module = module[idx]
                except (ValueError, TypeError, IndexError):
                    raise ValueError(f"Layer not found: {layer_name}")
            else:
                module = getattr(module, part)
        
        return module
