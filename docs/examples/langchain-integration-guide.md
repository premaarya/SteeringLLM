# LangChain Integration Guide

> **File:** [examples/langchain_steering_agent.py](../../examples/langchain_steering_agent.py)

This guide demonstrates how to integrate SteeringLLM with LangChain for building steered agents and chains.

---

## Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    LangChain + SteeringLLM Architecture                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────┐     ┌─────────────────────────────────────────────┐   │
│   │   LangChain     │     │            SteeringLLM Layer                │   │
│   │                 │     │   ┌─────────────────────────────────────┐   │   │
│   │  ┌───────────┐  │     │   │         LangChainSteeringLLM        │   │   │
│   │  │  Chains   │──┼────►│   │  ┌───────────┐   ┌───────────────┐  │   │   │
│   │  ├───────────┤  │     │   │  │ Steering  │ + │   HuggingFace │  │   │   │
│   │  │  Agents   │  │     │   │  │ Vectors   │   │     Model     │  │   │   │
│   │  ├───────────┤  │     │   │  └───────────┘   └───────────────┘  │   │   │
│   │  │  Prompts  │  │     │   └─────────────────────────────────────┘   │   │
│   │  └───────────┘  │     │                                             │   │
│   └─────────────────┘     └─────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

```bash
# Install SteeringLLM with agent support
pip install steering-llm[agents]

# Or install LangChain separately
pip install langchain langchain-community
```

---

## Example 1: Basic LangChain Chain with Steering

### Concept

Use steering to make LLM responses consistently helpful within any LangChain chain.

### Implementation

```python
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from steering_llm import SteeringModel, Discovery
from steering_llm.agents import LangChainSteeringLLM
```

### Step 1: Load and Prepare the Steering Model

```python
# Load model with steering capabilities
steering_model = SteeringModel.from_pretrained(
    "gpt2",
    device_map="auto",
    torch_dtype="auto"
)
```

### Step 2: Create a Helpful Steering Vector

```python
helpful_vector = Discovery.mean_difference(
    positive=[
        "I'm happy to help you with that!",
        "Let me assist you.",
        "I'd be glad to explain that.",
    ],
    negative=[
        "I don't care about that.",
        "That's not my problem.",
        "Figure it out yourself.",
    ],
    model=steering_model.model,
    layer=10
)
```

### Step 3: Create LangChain LLM Wrapper

```python
llm = LangChainSteeringLLM(
    steering_model=steering_model,
    vectors=[helpful_vector],  # Can pass multiple vectors
    alpha=2.0,
    max_length=100,
    temperature=0.7
)
```

### Step 4: Build and Run the Chain

```python
# Create prompt template
prompt = PromptTemplate(
    input_variables=["topic"],
    template="Write a brief explanation about {topic}:"
)

# Create chain
chain = LLMChain(llm=llm, prompt=prompt)

# Run with steering active
result = chain.run(topic="machine learning")
print(result)
```

### Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Chain Execution Flow                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   User Input: topic="machine learning"                                      │
│              │                                                              │
│              ▼                                                              │
│   ┌─────────────────────────────────────────────────┐                       │
│   │  PromptTemplate                                 │                       │
│   │  "Write a brief explanation about {topic}:"    │                       │
│   └─────────────────────┬───────────────────────────┘                       │
│                         │                                                   │
│                         ▼                                                   │
│   ┌─────────────────────────────────────────────────┐                       │
│   │  LangChainSteeringLLM                           │                       │
│   │  ┌─────────────────────────────────────────┐   │                       │
│   │  │  Apply helpful_vector (alpha=2.0)       │   │                       │
│   │  │  ┌─────────────────────────────────┐    │   │                       │
│   │  │  │  GPT-2 + Steering at Layer 10   │    │   │                       │
│   │  │  └─────────────────────────────────┘    │   │                       │
│   │  └─────────────────────────────────────────┘   │                       │
│   └─────────────────────┬───────────────────────────┘                       │
│                         │                                                   │
│                         ▼                                                   │
│   Output: "I'd be happy to explain! Machine learning is..."                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Example 2: Safety-Constrained Agent

### Concept

Create an agent that produces consistently safe responses by applying a safety steering vector.

### Implementation

```python
from steering_llm.agents import create_safety_agent

# Create safety steering vector using CAA (stronger method)
safety_vector = Discovery.caa(
    positive=[
        "I'm here to help in a safe and respectful way.",
        "Let's discuss this appropriately.",
        "I aim to be helpful and harmless.",
    ],
    negative=[
        "[toxic content examples]",
        "[harmful content examples]",
        "[biased content examples]",
    ],
    model=steering_model.model,
    layer=15
)

# Create safety agent
agent = create_safety_agent(
    steering_model=steering_model,
    safety_vector=safety_vector,
    alpha=2.5,  # Higher alpha for stronger safety
    max_length=150
)

# Generate safe responses
response = agent.generate("How can I help someone learn programming?")
```

### Safety Vector Effect

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       Safety Steering Effect                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Potentially Risky Prompt                                                  │
│              │                                                              │
│              ▼                                                              │
│   ┌─────────────────────────────────────────────────┐                       │
│   │           Safety Agent                          │                       │
│   │   ┌─────────────────────────────────────────┐   │                       │
│   │   │  Safety Vector Applied                  │   │                       │
│   │   │  • Steers away from harmful content     │   │                       │
│   │   │  • Promotes respectful language         │   │                       │
│   │   │  • Maintains helpfulness                │   │                       │
│   │   └─────────────────────────────────────────┘   │                       │
│   └─────────────────────┬───────────────────────────┘                       │
│                         │                                                   │
│                         ▼                                                   │
│   ┌─────────────────────────────────────────────────┐                       │
│   │  Safe, Helpful, Respectful Response            │                       │
│   └─────────────────────────────────────────────────┘                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Example 3: Domain Expert Agent

### Concept

Combine multiple steering vectors to create a domain expert that uses both technical vocabulary and formal language.

### Implementation

```python
from steering_llm.agents import create_domain_expert_agent

# Medical domain vector
medical_vector = Discovery.caa(
    positive=[
        "The patient presented with clinical symptoms requiring diagnosis.",
        "Medical treatment involves careful assessment of the condition.",
        "Clinical evidence suggests therapeutic intervention.",
    ],
    negative=[
        "The person was feeling bad and needed help.",
        "They went to the doctor because they were sick.",
        "The medicine made them feel better.",
    ],
    model=steering_model.model,
    layer=12
)

# Formal language vector
formal_vector = Discovery.mean_difference(
    positive=[
        "Furthermore, it is important to consider the implications.",
        "The analysis indicates several key findings.",
        "Consequently, the evidence suggests a clear conclusion.",
    ],
    negative=[
        "Also, you should think about this stuff.",
        "The results show some things.",
        "So, basically, it means this.",
    ],
    model=steering_model.model,
    layer=18
)

# Create expert agent with weighted vectors
agent = create_domain_expert_agent(
    steering_model=steering_model,
    domain_vectors=[medical_vector, formal_vector],
    weights=[0.7, 0.3],  # 70% medical, 30% formal
    max_length=200
)

# Generate domain-specific response
response = agent.generate("What should someone do if they have persistent headaches?")
```

### Multi-Vector Composition

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Domain Expert Vector Composition                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────┐                                                       │
│   │ Medical Vector  │────┐                                                  │
│   │   weight: 0.7   │    │                                                  │
│   └─────────────────┘    │     ┌─────────────────────────────────────┐      │
│                          ├────►│     Weighted Combination            │      │
│   ┌─────────────────┐    │     │                                     │      │
│   │ Formal Vector   │────┘     │  combined = 0.7×medical + 0.3×formal│      │
│   │   weight: 0.3   │          │                                     │      │
│   └─────────────────┘          └──────────────────┬──────────────────┘      │
│                                                   │                         │
│                                                   ▼                         │
│                                    ┌─────────────────────────────────┐      │
│                                    │  Response uses:                 │      │
│                                    │  • Medical terminology          │      │
│                                    │  • Clinical precision           │      │
│                                    │  • Formal structure             │      │
│                                    └─────────────────────────────────┘      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Example 4: Conversational Agent with Tools

### Concept

Build a conversational agent that maintains a helpful, engaging tone while using tools.

### Implementation

```python
from langchain.agents import AgentType, initialize_agent, load_tools

# Create conversational steering vector
conversational_vector = Discovery.mean_difference(
    positive=[
        "I'd be happy to help you with that question.",
        "That's an interesting topic! Let me explain.",
        "Great question! Here's what I know.",
    ],
    negative=[
        "I don't want to talk about that.",
        "That's boring.",
        "Ask someone else.",
    ],
    model=steering_model.model,
    layer=10
)

# Create steered LLM
llm = LangChainSteeringLLM(
    steering_model=steering_model,
    vectors=[conversational_vector],
    alpha=1.5,
    max_length=150
)

# Load tools and create agent
tools = load_tools(["llm-math", "wikipedia"], llm=llm)
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Run agent
result = agent.run("What is the square root of 144?")
```

---

## Example 5: Context Manager for Temporary Steering

### Concept

Apply steering temporarily within a specific code block, ensuring automatic cleanup.

### Implementation

```python
# Create steered LLM
llm = LangChainSteeringLLM(
    steering_model=steering_model,
    vectors=[formal_vector],
    alpha=2.0,
    max_length=100
)

prompt = "Describe artificial intelligence"

# Without steering
print("Without steering:")
response_without = steering_model.generate(prompt, max_length=100)
print(response_without)

# With steering (context manager)
print("\nWith steering:")
with llm:
    response_with = llm.generate(prompt)
    print(response_with)

# Steering automatically removed
print(f"\nSteering still active? {llm.is_steering_active}")  # False
```

### Context Manager Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Context Manager Lifecycle                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   with llm:                          ← __enter__: Apply steering            │
│   │                                                                         │
│   │   response = llm.generate(...)   ← Steering ACTIVE                      │
│   │                                                                         │
│   └──────────────────────────────────← __exit__: Remove steering            │
│                                                                             │
│   # After block                      ← Steering INACTIVE                    │
│                                                                             │
│   ─────────────────────────────────────────────────────────────────────     │
│                                                                             │
│   Benefits:                                                                 │
│   ✅ Automatic cleanup - no memory leaks                                    │
│   ✅ Exception-safe - steering removed even if error occurs                 │
│   ✅ Clear scope - easy to see where steering applies                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## LangChainSteeringLLM Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `steering_model` | `SteeringModel` | The wrapped HuggingFace model | Required |
| `vectors` | `List[Tensor]` | Steering vectors to apply | Required |
| `alpha` | `float` | Steering strength | `2.0` |
| `max_length` | `int` | Maximum generation length | `100` |
| `temperature` | `float` | Sampling temperature | `0.7` |
| `top_p` | `float` | Nucleus sampling | `0.9` |
| `top_k` | `int` | Top-k sampling | `50` |

---

## Best Practices

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        LangChain Integration Best Practices                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ✅ DO                                 ❌ DON'T                            │
│   ─────────────────────────────────     ─────────────────────────────────   │
│   • Use context manager for             • Apply steering without removing   │
│     temporary steering                                                      │
│                                                                             │
│   • Test different alpha values         • Use very high alpha (>4.0)        │
│     for your use case                     without testing                   │
│                                                                             │
│   • Combine complementary vectors       • Combine conflicting vectors       │
│     (e.g., helpful + formal)              (e.g., verbose + concise)         │
│                                                                             │
│   • Create vectors from diverse         • Use only 1-2 contrast examples    │
│     examples (5-20 each)                                                    │
│                                                                             │
│   • Match layer selection to model      • Use same layer for all models     │
│     depth (40-60% typically)                                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Common Patterns

### Pattern 1: Safety + Domain

```python
# Combine safety constraints with domain expertise
llm = LangChainSteeringLLM(
    steering_model=model,
    vectors=[safety_vector, domain_vector],
    alphas=[2.5, 1.5],  # Stronger safety, moderate domain
)
```

### Pattern 2: Dynamic Steering

```python
# Switch vectors based on context
def get_llm_for_context(context: str):
    if "medical" in context:
        return LangChainSteeringLLM(model, vectors=[medical_vector])
    elif "legal" in context:
        return LangChainSteeringLLM(model, vectors=[legal_vector])
    else:
        return LangChainSteeringLLM(model, vectors=[general_vector])
```

### Pattern 3: A/B Testing

```python
# Compare steered vs unsteered responses
def ab_test(prompt):
    unsteered = model.generate(prompt)
    steered = llm.generate(prompt)
    return {"control": unsteered, "treatment": steered}
```

---

## Next Steps

- **Azure deployment:** [Azure Agent Guide](./azure-agent-guide.md)
- **RAG applications:** [LlamaIndex RAG Guide](./llamaindex-rag-guide.md)
- **Evaluation:** Check out the evaluation module for measuring steering effectiveness
