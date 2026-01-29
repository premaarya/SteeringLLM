# SteeringLLM

Runtime LLM behavior modification through activation steering.

## Overview

SteeringLLM enables you to modify LLM behavior at inference time without retraining. Apply steering vectors to guide model outputs toward desired characteristics (e.g., more helpful, safer, more creative).

## Quick Start

```python
from steering_llm import SteeringModel, Discovery

# Load model
model = SteeringModel.from_pretrained("meta-llama/Llama-3.2-3B")

# Create steering vector from contrast examples
vector = Discovery.mean_difference(
    positive=["I love helping people!", "You're amazing!"],
    negative=["I hate this.", "You're terrible."],
    model=model,
    layer=15
)

# Generate with steering
output = model.generate_with_steering(
    "Tell me about yourself",
    vector=vector,
    alpha=2.0,
    max_length=100
)
```

## Installation

```bash
pip install steering-llm
```

### Development Installation

```bash
git clone https://github.com/jnPiyush/SteeringLLM.git
cd SteeringLLM
pip install -e ".[dev]"
```

## Features

- ✅ **Steering Vector Primitives**: Create, apply, and remove steering vectors
- ✅ **HuggingFace Integration**: Extended model support with quantization
- 🚧 **Advanced Discovery**: Multi-layer and ensemble methods

## Supported Models

SteeringLLM supports a wide range of transformer architectures:

| Model Family | Architectures | Example Models |
|--------------|---------------|----------------|
| **Llama** | llama | meta-llama/Llama-3.2-3B, meta-llama/Llama-2-7b-hf |
| **Mistral** | mistral | mistralai/Mistral-7B-v0.1, mistralai/Mixtral-8x7B-v0.1 |
| **Gemma** | gemma, gemma2 | google/gemma-2-2b, google/gemma-7b |
| **Phi** | phi, phi3 | microsoft/phi-2, microsoft/Phi-3-mini-4k-instruct |
| **Qwen** | qwen2, qwen2_moe | Qwen/Qwen2.5-7B-Instruct, Qwen/Qwen1.5-7B |
| **GPT** | gpt2, gpt_neo, gpt_neox, gptj | gpt2, EleutherAI/gpt-neo-2.7B, EleutherAI/gpt-j-6b |
| **OPT** | opt | facebook/opt-1.3b, facebook/opt-6.7b |
| **BLOOM** | bloom | bigscience/bloom-560m, bigscience/bloom-1b7 |
| **Falcon** | falcon | tiiuae/falcon-7b, tiiuae/falcon-40b |

### Quantization Support

Steering works with quantized models via BitsAndBytes:

```python
# 8-bit quantization
model = SteeringModel.from_pretrained(
    "meta-llama/Llama-3.2-3B",
    load_in_8bit=True,
    device_map="auto"
)

# 4-bit quantization
model = SteeringModel.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    load_in_4bit=True,
    device_map="auto"
)
```

### Device Support

- ✅ **CPU**: Full support
- ✅ **CUDA**: Multi-GPU with `device_map="auto"`
- ✅ **MPS**: Apple Silicon support

## Requirements

- Python 3.9+
- PyTorch 2.0+
- Transformers 4.36+

## Architecture

See [docs/adr/ADR-001-steeringllm-architecture.md](docs/adr/ADR-001-steeringllm-architecture.md) for architectural decisions.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

## License

MIT License - see LICENSE file for details.
