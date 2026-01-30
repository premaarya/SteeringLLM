# Basic Steering Usage Guide

> **File:** [examples/steering_basic_usage.py](../../examples/steering_basic_usage.py)

This guide walks you through the fundamental concepts of activation steering using SteeringLLM.

---

## Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Basic Steering Workflow                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌──────────┐     ┌──────────────┐     ┌─────────────┐     ┌───────────┐  │
│   │  Load    │ ──► │   Create     │ ──► │   Apply     │ ──► │ Generate  │  │
│   │  Model   │     │   Steering   │     │   Steering  │     │   with    │  │
│   │          │     │   Vector     │     │   Vector    │     │  Steering │  │
│   └──────────┘     └──────────────┘     └─────────────┘     └───────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

```bash
# Install SteeringLLM
pip install steering-llm

# Or install from source
git clone https://github.com/jnPiyush/SteeringLLM.git
cd SteeringLLM
pip install -e .
```

---

## Step-by-Step Implementation

### Step 1: Import Required Modules

```python
from steering_llm import SteeringModel, Discovery
```

| Module | Purpose |
|--------|---------|
| `SteeringModel` | Wrapper around HuggingFace models that enables steering |
| `Discovery` | Methods to create steering vectors from examples |

---

### Step 2: Load a Model with Steering Capabilities

```python
# Load model
model = SteeringModel.from_pretrained("meta-llama/Llama-3.2-3B")
```

**How it works:**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SteeringModel.from_pretrained()                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   1. Downloads model from HuggingFace Hub (or loads from cache)             │
│   2. Wraps model with steering hooks at each transformer layer              │
│   3. Identifies model architecture for correct layer access                 │
│   4. Returns SteeringModel instance ready for steering                      │
│                                                                             │
│   ┌──────────────────┐                                                      │
│   │  HuggingFace     │                                                      │
│   │  Model           │                                                      │
│   │  ┌────────────┐  │     ┌─────────────────────────────────────────┐      │
│   │  │ Layer 0    │  │     │  SteeringModel                          │      │
│   │  ├────────────┤  │     │  ┌─────────────────────────────────┐    │      │
│   │  │ Layer 1    │◄─┼────►│  │ + Steering hooks on each layer  │    │      │
│   │  ├────────────┤  │     │  │ + Vector injection capability   │    │      │
│   │  │ ...        │  │     │  │ + Activation tracking           │    │      │
│   │  ├────────────┤  │     │  └─────────────────────────────────┘    │      │
│   │  │ Layer N    │  │     └─────────────────────────────────────────┘      │
│   │  └────────────┘  │                                                      │
│   └──────────────────┘                                                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Supported Models:**
- Llama (3.2, 2, etc.)
- Mistral / Mixtral
- GPT-2, GPT-Neo, GPT-J
- Phi-2, Phi-3
- Gemma, Qwen, Falcon, OPT, BLOOM

---

### Step 3: Create a Steering Vector

```python
# Define contrast examples
positive_examples = [
    "I love helping people!",
    "You're doing great!",
    "That's wonderful news!",
]

negative_examples = [
    "I hate helping people.",
    "You're doing terribly.",
    "That's awful news.",
]

# Create steering vector
result = Discovery.mean_difference(
    positive=positive_examples,
    negative=negative_examples,
    model=model.model,  # Access underlying HF model
    layer=15,           # Target layer (adjust based on model)
)

vector = result.vector
```

**How Discovery.mean_difference() works:**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Mean Difference Algorithm                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Positive Examples                    Negative Examples                    │
│   ┌────────────────────┐               ┌────────────────────┐               │
│   │ "I love helping!"  │               │ "I hate helping."  │               │
│   │ "You're doing      │               │ "You're doing      │               │
│   │  great!"           │               │  terribly."        │               │
│   │ "Wonderful news!"  │               │ "Awful news."      │               │
│   └─────────┬──────────┘               └─────────┬──────────┘               │
│             │                                    │                          │
│             ▼                                    ▼                          │
│   ┌────────────────────┐               ┌────────────────────┐               │
│   │ Extract activations│               │ Extract activations│               │
│   │ at layer 15        │               │ at layer 15        │               │
│   └─────────┬──────────┘               └─────────┬──────────┘               │
│             │                                    │                          │
│             ▼                                    ▼                          │
│   ┌────────────────────┐               ┌────────────────────┐               │
│   │ Compute mean       │               │ Compute mean       │               │
│   │ (positive centroid)│               │ (negative centroid)│               │
│   │ μ+ = [0.8, 0.3...]│               │ μ- = [0.2, -0.1..]│               │
│   └─────────┬──────────┘               └─────────┬──────────┘               │
│             │                                    │                          │
│             └────────────────┬───────────────────┘                          │
│                              │                                              │
│                              ▼                                              │
│                    ┌─────────────────────┐                                  │
│                    │  Steering Vector    │                                  │
│                    │  v = μ+ - μ-        │                                  │
│                    │  [0.6, 0.4, ...]    │                                  │
│                    └─────────────────────┘                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Parameter Guide:**

| Parameter | Description | Recommended Value |
|-----------|-------------|-------------------|
| `positive` | List of examples with desired behavior | 5-20 diverse examples |
| `negative` | List of examples with opposite behavior | Same count as positive |
| `model` | The underlying HuggingFace model | `model.model` |
| `layer` | Transformer layer to extract from | 40-60% of model depth |

---

### Step 4: Apply Steering and Generate

**Option A: Persistent Steering**

```python
# Apply steering (stays active for multiple generations)
model.apply_steering(vector, alpha=2.0)

# Generate (steering is applied)
output = model.generate(
    "Tell me about yourself",
    max_length=100,
)
print(output)

# Check what's active
active = model.list_active_steering()
print(f"Active steering: {active}")

# Remove when done
model.remove_steering()
```

**Option B: One-shot Steering (Recommended for most cases)**

```python
# Automatically applies and removes steering
output = model.generate_with_steering(
    prompt="Tell me about yourself",
    vector=vector,
    alpha=2.0,
    max_length=100,
)
```

---

### Understanding the Alpha Parameter

The `alpha` parameter controls steering strength:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Alpha Parameter Effects                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Formula:  steered_activation = original + (alpha × steering_vector)       │
│                                                                             │
│   ─────────────────────────────────────────────────────────────────────     │
│                                                                             │
│   alpha = 0.0   │████████████████████░░░░░░░░░░░░░░░░│  No effect           │
│   alpha = 1.0   │████████████████████████████░░░░░░░░│  Subtle steering     │
│   alpha = 2.0   │████████████████████████████████████│  Moderate (typical)  │
│   alpha = 3.0   │████████████████████████████████████│  Strong steering     │
│   alpha > 4.0   │████████████████████████████████████│  ⚠️ May degrade      │
│                                                                             │
│   ─────────────────────────────────────────────────────────────────────     │
│                                                                             │
│   Guidelines:                                                               │
│   • Start with alpha=2.0 and adjust based on results                        │
│   • Higher alpha = stronger effect but may reduce coherence                 │
│   • Different concepts may need different alpha values                      │
│   • Test with your specific use case                                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Complete Working Example

```python
from steering_llm import SteeringModel, Discovery

# 1. Load model
print("Loading model...")
model = SteeringModel.from_pretrained("gpt2")

# 2. Define contrast examples for "helpfulness"
positive = [
    "I'd be happy to help you with that!",
    "Let me assist you with your question.",
    "Great question! Here's what I know.",
    "I'm here to help - let me explain.",
]

negative = [
    "Figure it out yourself.",
    "That's not my problem.",
    "I don't care about that.",
    "Why are you asking me?",
]

# 3. Create steering vector
print("Creating steering vector...")
result = Discovery.mean_difference(
    positive=positive,
    negative=negative,
    model=model.model,
    layer=6  # GPT-2 has 12 layers, so layer 6 = 50%
)

# 4. Generate with and without steering
prompt = "How do I learn Python?"

print("\n--- Without Steering ---")
output_normal = model.generate(prompt, max_length=50)
print(output_normal)

print("\n--- With Steering (alpha=2.0) ---")
output_steered = model.generate_with_steering(
    prompt=prompt,
    vector=result.vector,
    alpha=2.0,
    max_length=50
)
print(output_steered)
```

---

## Expected Output Differences

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      Before vs After Steering                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Prompt: "How do I learn Python?"                                          │
│                                                                             │
│   ┌─────────────────────────────────┐  ┌─────────────────────────────────┐  │
│   │        WITHOUT STEERING         │  │         WITH STEERING           │  │
│   ├─────────────────────────────────┤  ├─────────────────────────────────┤  │
│   │                                 │  │                                 │  │
│   │ "Python is a programming       │  │ "Great question! I'd be happy   │  │
│   │  language that can be used     │  │  to help you learn Python!      │  │
│   │  for various applications..."  │  │  Here are some steps to get     │  │
│   │                                 │  │  started..."                    │  │
│   │ (Neutral, factual)             │  │ (Enthusiastic, helpful)         │  │
│   │                                 │  │                                 │  │
│   └─────────────────────────────────┘  └─────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Troubleshooting

| Issue | Possible Cause | Solution |
|-------|---------------|----------|
| No visible effect | Alpha too low | Increase alpha to 2.0-3.0 |
| Incoherent output | Alpha too high | Decrease alpha |
| Wrong behavior | Bad contrast examples | Use clearer, more distinct examples |
| CUDA out of memory | Model too large | Use quantization or smaller model |

---

## Next Steps

- **Multi-vector composition:** [Vector Composition Guide](./vector-composition-guide.md)
- **LangChain integration:** [LangChain Guide](./langchain-integration-guide.md)
- **Azure deployment:** [Azure Agent Guide](./azure-agent-guide.md)
- **RAG applications:** [LlamaIndex RAG Guide](./llamaindex-rag-guide.md)
