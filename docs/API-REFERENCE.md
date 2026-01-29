# SteeringLLM API Quick Reference

Complete API reference for Stories #1-6 (Foundation).

---

## Installation

```bash
pip install -e .
```

---

## Core Classes

### SteeringVector

**Storage container for steering vectors.**

```python
from steering_llm import SteeringVector

# Create
vector = SteeringVector(
    tensor=torch.randn(3072),
    layer=15,
    layer_name="model.layers.15",
    model_name="meta-llama/Llama-3.2-3B",
    metadata={"description": "Safety vector"}
)

# Properties
vector.shape          # torch.Size([3072])
vector.dtype          # torch.float32
vector.device         # cuda:0 or cpu
vector.magnitude      # 2.347 (L2 norm)
vector.layer          # 15
vector.model_name     # "meta-llama/Llama-3.2-3B"

# Methods
vector.save("vectors/safety_v1")           # Save to disk
loaded = SteeringVector.load("vectors/safety_v1")  # Load from disk
vector.to_device("cuda")                   # Move to device
vector.validate(expected_dim=3072)         # Validate integrity
```

---

### Discovery

**Methods for creating steering vectors from contrast datasets.**

```python
from steering_llm import Discovery, SteeringModel

model = SteeringModel.from_pretrained("meta-llama/Llama-3.2-3B")

# Mean difference method
vector = Discovery.mean_difference(
    positive=["I love this!", "Great work!"],
    negative=["I hate this.", "Terrible work."],
    model=model.model,  # Underlying HF model
    layer=15,
    batch_size=8,
    max_length=128
)
```

---

### SteeringModel

**Main interface for applying steering vectors at inference.**

#### Loading Models

```python
from steering_llm import SteeringModel

# Basic
model = SteeringModel.from_pretrained("meta-llama/Llama-3.2-3B")

# With quantization
model = SteeringModel.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    device_map="auto",
    load_in_8bit=True
)

# With custom tokenizer
model = SteeringModel.from_pretrained(
    "google/gemma-2-2b",
    tokenizer_name="custom-tokenizer"
)
```

#### Applying Steering

```python
# Apply steering vector
model.apply_steering(vector, alpha=1.5)

# Generate with steering active
output = model.generate(
    input_ids,
    max_length=100,
    temperature=0.7
)

# Check active steering
active = model.list_active_steering()
# [{'layer': 15, 'alpha': 1.5, 'model_name': '...', 'magnitude': 2.3}]

# Remove steering
model.remove_steering(layer=15)  # Specific layer
model.remove_steering()          # All layers
model.remove_all_steering()      # Alias
```

#### Convenience Method

```python
# Apply, generate, and remove in one call
output = model.generate_with_steering(
    prompt="Tell me about yourself",
    vector=safety_vector,
    alpha=2.0,
    max_length=100,
    temperature=0.7
)
# Steering automatically removed after generation

# Batch generation
outputs = model.generate_with_steering(
    prompt=["Prompt 1", "Prompt 2", "Prompt 3"],
    vector=vector,
    alpha=1.5
)
```

#### Access Underlying Model

```python
# SteeringModel delegates to underlying HuggingFace model
config = model.config
device = model.device
dtype = model.dtype

# Can use as drop-in replacement
model.eval()
model.to("cuda")
```

---

## Complete Workflow Example

```python
from steering_llm import SteeringModel, Discovery

# 1. Load model
model = SteeringModel.from_pretrained("meta-llama/Llama-3.2-3B")

# 2. Define contrast examples
positive = [
    "I love helping people!",
    "You're doing great!",
    "That's wonderful news!",
]

negative = [
    "I hate helping people.",
    "You're doing terribly.",
    "That's awful news.",
]

# 3. Create steering vector
friendly_vector = Discovery.mean_difference(
    positive=positive,
    negative=negative,
    model=model.model,
    layer=15
)

# 4. Save vector for reuse
friendly_vector.save("vectors/friendliness_v1")

# 5. Apply steering and generate
model.apply_steering(friendly_vector, alpha=2.0)

output = model.generate(
    "Tell me about yourself",
    max_length=100
)

print(output)

# 6. Check what's active
active = model.list_active_steering()
print(f"Active: {active}")

# 7. Remove steering
model.remove_steering()

# 8. Later: Load and use saved vector
loaded_vector = SteeringVector.load("vectors/friendliness_v1")
output = model.generate_with_steering(
    "Hello!",
    vector=loaded_vector,
    alpha=1.5
)
```

---

## Advanced Usage

### Multiple Steering Vectors

```python
# Apply multiple vectors to different layers
model.apply_steering(safety_vector, alpha=2.0)   # layer 15
model.apply_steering(tone_vector, alpha=1.5)     # layer 20

# List all active
active = model.list_active_steering()
# [
#   {'layer': 15, 'alpha': 2.0, ...},
#   {'layer': 20, 'alpha': 1.5, ...}
# ]

# Remove specific
model.remove_steering(layer=15)

# Remove all
model.remove_all_steering()
```

### Error Handling

```python
# Dimension mismatch
try:
    model.apply_steering(wrong_size_vector)
except ValueError as e:
    print(f"Vector incompatible: {e}")

# Double steering on same layer
try:
    model.apply_steering(vector1)
    model.apply_steering(vector2)  # Same layer!
except RuntimeError as e:
    print(f"Already active: {e}")
    model.remove_steering()
    model.apply_steering(vector2)

# Always cleanup on error
try:
    model.apply_steering(vector)
    # ... do work ...
except Exception as e:
    print(f"Error: {e}")
finally:
    model.remove_steering()  # Guaranteed cleanup
```

### Using with Tokenizer

```python
# Tokenize manually
inputs = model.tokenizer(
    ["Prompt 1", "Prompt 2"],
    return_tensors="pt",
    padding=True
)

# Generate with manual inputs
model.apply_steering(vector, alpha=1.5)
outputs = model.generate(**inputs, max_length=100)
decoded = model.tokenizer.batch_decode(outputs, skip_special_tokens=True)
model.remove_steering()
```

---

## API Summary

### SteeringVector
- `__init__(tensor, layer, layer_name, model_name, ...)`
- `save(path)` - Save to disk
- `load(path)` - Load from disk  
- `to_device(device)` - Move to device
- `validate(expected_dim)` - Validate integrity

### Discovery
- `mean_difference(positive, negative, model, layer, ...)` - Create vector

### SteeringModel
- `from_pretrained(model_name, **kwargs)` - Load model
- `apply_steering(vector, alpha)` - Apply vector
- `remove_steering(layer=None)` - Remove vector(s)
- `remove_all_steering()` - Remove all
- `list_active_steering()` - List active
- `generate_with_steering(prompt, vector, alpha, **kwargs)` - Convenience

---

## Supported Architectures

✅ **Llama** (Llama 3.2, Llama 3.1, Llama 2)  
✅ **Mistral** (Mistral 7B, Mixtral)  
✅ **Gemma** (Gemma 2, Gemma 1)  
🚧 **GPT-2** (partial support)

---

## Tips & Best Practices

1. **Start with alpha=1.0**, then adjust up/down based on results
2. **Use generate_with_steering()** for automatic cleanup
3. **Save vectors** after creation for reuse
4. **Validate vectors** before applying to production
5. **Monitor active steering** with list_active_steering()
6. **Handle errors gracefully** with try/finally blocks
7. **Test on dev data** before production deployment

---

## Performance Characteristics

- **Hook registration**: ~1ms per layer
- **Steering overhead**: ~2-5ms per forward pass
- **Memory overhead**: ~1KB per hook + vector size
- **Supports**: CPU, CUDA, MPS devices
- **Batch friendly**: Works with any batch size

---

## Next Steps

- **Story #7**: Integration tests with real models
- **Story #8**: Context manager API
- **Story #9**: Performance optimization
- **Story #10**: Multi-vector composition

---

**Last Updated**: 2026-01-28  
**Version**: 0.1.0  
**Status**: Foundation Complete ✅
