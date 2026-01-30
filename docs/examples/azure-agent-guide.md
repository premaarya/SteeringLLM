# Azure Agent Framework Integration Guide

> **File:** [examples/azure_agent_foundry.py](../../examples/azure_agent_foundry.py)

This guide demonstrates how to integrate SteeringLLM with Microsoft's Agent Framework for Azure AI Foundry deployment and enterprise features.

---

## Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Azure AI Foundry + SteeringLLM                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                     Azure AI Foundry                                │   │
│   │   ┌─────────────┐   ┌─────────────┐   ┌─────────────────────────┐   │   │
│   │   │  Prompt     │   │   Azure     │   │     Azure Monitor       │   │   │
│   │   │   Flow      │   │   Portal    │   │  (Application Insights) │   │   │
│   │   └──────┬──────┘   └──────┬──────┘   └────────────┬────────────┘   │   │
│   │          │                 │                       │                │   │
│   │          └────────────┬────┴───────────────────────┘                │   │
│   │                       │                                             │   │
│   │                       ▼                                             │   │
│   │          ┌─────────────────────────────────────────┐                │   │
│   │          │        AzureSteeringAgent               │                │   │
│   │          │   ┌───────────────────────────────┐     │                │   │
│   │          │   │  SteeringModel + Vectors      │     │                │   │
│   │          │   │  + Tracing + Deployment       │     │                │   │
│   │          │   └───────────────────────────────┘     │                │   │
│   │          └─────────────────────────────────────────┘                │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

```bash
# Install SteeringLLM with Azure support
pip install steering-llm[azure]

# Or install dependencies separately
pip install agent-framework azure-monitor-opentelemetry
```

---

## Example 1: Basic Azure Agent with Steering

### Concept

Create a production-ready agent with steering for consistent, helpful responses.

### Implementation

```python
from steering_llm import SteeringModel, Discovery
from steering_llm.agents import AzureSteeringAgent
```

### Step 1: Load the Model

```python
steering_model = SteeringModel.from_pretrained(
    "gpt2",
    device_map="auto"
)
```

### Step 2: Create a Steering Vector

```python
helpful_vector = Discovery.mean_difference(
    positive=[
        "I'm here to assist you with your questions.",
        "Let me help you understand this better.",
        "I'd be pleased to provide information.",
    ],
    negative=[
        "I don't want to answer that.",
        "That's not interesting.",
        "Ask someone else.",
    ],
    model=steering_model.model,
    layer=10
)
```

### Step 3: Create the Azure Agent

```python
agent = AzureSteeringAgent(
    steering_model=steering_model,
    agent_name="helpful_assistant",
    vectors=[helpful_vector],
    alpha=2.0,
    max_tokens=150,
    temperature=0.7
)

# Generate responses
response = agent.generate("What is machine learning?")
print(response)
```

### Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      AzureSteeringAgent Architecture                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   User Request                                                              │
│        │                                                                    │
│        ▼                                                                    │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                    AzureSteeringAgent                               │   │
│   │                                                                     │   │
│   │   ┌─────────────────┐    ┌──────────────────────────────────────┐   │   │
│   │   │   Preprocessing │    │        SteeringModel                 │   │   │
│   │   │   • Tokenization│───►│  ┌────────────────────────────────┐  │   │   │
│   │   │   • Formatting  │    │  │  helpful_vector applied        │  │   │   │
│   │   └─────────────────┘    │  │  at layer 10 (alpha=2.0)       │  │   │   │
│   │                          │  └────────────────────────────────┘  │   │   │
│   │                          └──────────────┬───────────────────────┘   │   │
│   │                                         │                           │   │
│   │   ┌─────────────────┐                   │                           │   │
│   │   │  Postprocessing │◄──────────────────┘                           │   │
│   │   │  • Formatting   │                                               │   │
│   │   │  • Validation   │                                               │   │
│   │   └────────┬────────┘                                               │   │
│   │            │                                                        │   │
│   └────────────┼────────────────────────────────────────────────────────┘   │
│                │                                                            │
│                ▼                                                            │
│   Steered Response                                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Example 2: Azure Agent with Tracing

### Concept

Enable Azure Monitor tracing to track agent performance and debug issues in production.

### Implementation

```python
# Configure tracing
tracing_config = {
    "connection_string": "InstrumentationKey=your-key-here;IngestionEndpoint=https://..."
}

# Create agent with tracing enabled
agent = AzureSteeringAgent(
    steering_model=steering_model,
    agent_name="traced_agent",
    vectors=[vector],
    alpha=1.5,
    enable_tracing=True,
    tracing_config=tracing_config,
    max_tokens=100
)

# Generate (traces sent to Azure Monitor)
response = agent.generate("Explain the benefits of cloud computing.")
```

### What Gets Traced

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Azure Monitor Tracing Data                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                    Application Insights                             │   │
│   │                                                                     │   │
│   │   Request Traces:                                                   │   │
│   │   ├── agent.generate() called                                       │   │
│   │   │   ├── Input prompt: "Explain the benefits..."                   │   │
│   │   │   ├── Vectors applied: [helpful_vector]                         │   │
│   │   │   ├── Alpha: 1.5                                                │   │
│   │   │   └── Duration: 234ms                                           │   │
│   │   │                                                                 │   │
│   │   │   ├── steering.apply()                                          │   │
│   │   │   │   ├── Layer: 10                                             │   │
│   │   │   │   └── Duration: 12ms                                        │   │
│   │   │   │                                                             │   │
│   │   │   ├── model.generate()                                          │   │
│   │   │   │   ├── Tokens generated: 87                                  │   │
│   │   │   │   └── Duration: 189ms                                       │   │
│   │   │   │                                                             │   │
│   │   │   └── steering.remove()                                         │   │
│   │   │       └── Duration: 8ms                                         │   │
│   │   │                                                                 │   │
│   │   Custom Events:                                                    │   │
│   │   ├── SteeringApplied                                               │   │
│   │   ├── GenerationComplete                                            │   │
│   │   └── SteeringRemoved                                               │   │
│   │                                                                     │   │
│   │   Performance Metrics:                                              │   │
│   │   ├── Latency (p50, p95, p99)                                       │   │
│   │   ├── Throughput (requests/sec)                                     │   │
│   │   └── Error rate                                                    │   │
│   │                                                                     │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Example 3: Azure AI Foundry Deployment

### Concept

Export agent configuration for deployment to Azure AI Foundry.

### Implementation

```python
# Create production agent with multiple vectors
safety_vector = Discovery.caa(
    positive=["Safe and appropriate content."],
    negative=["Harmful or inappropriate content."],
    model=steering_model.model,
    layer=15
)

domain_vector = Discovery.mean_difference(
    positive=["Technical and professional language."],
    negative=["Casual conversation."],
    model=steering_model.model,
    layer=10
)

agent = AzureSteeringAgent(
    steering_model=steering_model,
    agent_name="production_agent",
    vectors=[safety_vector, domain_vector],
    alpha=2.0,
    max_tokens=200
)

# Generate deployment configuration
deployment_config = agent.to_azure_deployment(
    endpoint="https://your-resource.cognitiveservices.azure.com/",
    api_key="your-api-key-here",
    deployment_name="steering-agent-v1"
)

print(json.dumps(deployment_config, indent=2))
```

### Deployment Configuration Output

```json
{
  "deployment_name": "steering-agent-v1",
  "endpoint": "https://your-resource.cognitiveservices.azure.com/",
  "model_config": {
    "base_model": "gpt2",
    "steering_vectors": 2,
    "alpha": 2.0,
    "max_tokens": 200
  },
  "agent_config": {
    "name": "production_agent",
    "enable_tracing": false,
    "enable_safety": true
  },
  "deployment_type": "azure_ai_foundry"
}
```

---

## Example 4: Prompt Flow Integration

### Concept

Configure agent for use in Azure Prompt Flow for visual workflow design.

### Implementation

```python
from steering_llm.agents import create_prompt_flow_config

# Create customer service agent
agent = AzureSteeringAgent(
    steering_model=steering_model,
    agent_name="customer_service_agent",
    vectors=[service_vector],
    alpha=2.0,
    max_tokens=150
)

# Create Prompt Flow configuration
flow_config = create_prompt_flow_config(
    agent=agent,
    flow_name="customer_service_flow",
    inputs=["user_query", "customer_context"],
    outputs=["agent_response", "confidence_score"]
)
```

### Prompt Flow Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Prompt Flow Workflow                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌──────────────────┐                                                      │
│   │    user_query    │──────┐                                               │
│   └──────────────────┘      │                                               │
│                             │     ┌─────────────────────────────────────┐   │
│   ┌──────────────────┐      ├────►│                                     │   │
│   │ customer_context │──────┘     │   customer_service_agent            │   │
│   └──────────────────┘            │   (AzureSteeringAgent)              │   │
│                                   │                                     │   │
│                                   │   • Apply service_vector            │   │
│                                   │   • Generate steered response       │   │
│                                   │   • Calculate confidence            │   │
│                                   │                                     │   │
│                                   └──────────────┬──────────────────────┘   │
│                                                  │                          │
│                             ┌────────────────────┼────────────────────┐     │
│                             │                    │                    │     │
│                             ▼                    ▼                    │     │
│               ┌──────────────────┐  ┌────────────────────┐            │     │
│               │  agent_response  │  │ confidence_score   │            │     │
│               └──────────────────┘  └────────────────────┘            │     │
│                                                                             │
│   Benefits:                                                                 │
│   • Visual flow design in Prompt Flow                                       │
│   • A/B testing different steering configurations                           │
│   • Integration with other Azure services                                   │
│   • Built-in monitoring and analytics                                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Example 5: Multi-Agent Orchestration

### Concept

Orchestrate multiple specialized agents with different steering configurations.

### Implementation

```python
from steering_llm.agents import create_multi_agent_orchestration

# Safety agent (content moderation)
safety_agent = AzureSteeringAgent(
    steering_model=steering_model,
    agent_name="safety_agent",
    vectors=[safety_vector],
    alpha=2.5
)

# Expert agent (domain knowledge)
expert_agent = AzureSteeringAgent(
    steering_model=steering_model,
    agent_name="expert_agent",
    vectors=[expert_vector],
    alpha=1.8
)

# Create orchestration
orchestration = create_multi_agent_orchestration(
    agents=[safety_agent, expert_agent],
    orchestration_strategy="sequential"
)
```

### Orchestration Strategies

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Multi-Agent Orchestration Strategies                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  SEQUENTIAL: Safety Check → Expert Response                         │   │
│   │                                                                     │   │
│   │  Input ──► [Safety Agent] ──► [Expert Agent] ──► Output             │   │
│   │              │                    │                                 │   │
│   │              ▼                    ▼                                 │   │
│   │         "Is it safe?"       "Add expertise"                         │   │
│   │                                                                     │   │
│   │  Use case: Content moderation before generation                     │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  PARALLEL: Both Process, Combine Results                            │   │
│   │                                                                     │   │
│   │            ┌──► [Safety Agent] ───┐                                 │   │
│   │  Input ────┤                      ├──► [Combine] ──► Output         │   │
│   │            └──► [Expert Agent] ───┘                                 │   │
│   │                                                                     │   │
│   │  Use case: Multi-perspective analysis                               │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  HIERARCHICAL: Coordinator Routes to Specialists                    │   │
│   │                                                                     │   │
│   │                    [Coordinator Agent]                              │   │
│   │                     /      |      \                                 │   │
│   │                    ▼       ▼       ▼                                │   │
│   │               [Medical] [Legal] [Technical]                         │   │
│   │                                                                     │   │
│   │  Use case: Multi-domain question answering                          │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Example 6: Async Generation for High Throughput

### Concept

Use async methods for better performance when handling multiple concurrent requests.

### Implementation

```python
import asyncio

agent = AzureSteeringAgent(
    steering_model=steering_model,
    agent_name="async_agent",
    vectors=[helpful_vector],
    alpha=1.5,
    max_tokens=100
)

async def generate_multiple():
    prompts = [
        "What is Python?",
        "Explain machine learning.",
        "What is cloud computing?",
    ]
    
    # Generate all responses concurrently
    tasks = [agent.agenerate(prompt) for prompt in prompts]
    responses = await asyncio.gather(*tasks)
    
    for prompt, response in zip(prompts, responses):
        print(f"Q: {prompt}")
        print(f"A: {response}\n")

# Run async generation
asyncio.run(generate_multiple())
```

### Async Performance Benefits

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Sync vs Async Comparison                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   SYNCHRONOUS (Sequential)                                                  │
│   ─────────────────────────────────────────────────────────────             │
│   Request 1: ████████████                                                   │
│   Request 2:             ████████████                                       │
│   Request 3:                         ████████████                           │
│                                                                             │
│   Total Time: ─────────────────────────────────────────► 3x                 │
│                                                                             │
│   ─────────────────────────────────────────────────────────────────────     │
│                                                                             │
│   ASYNCHRONOUS (Concurrent)                                                 │
│   ─────────────────────────────────────────────────────────────             │
│   Request 1: ████████████                                                   │
│   Request 2: ████████████                                                   │
│   Request 3: ████████████                                                   │
│                                                                             │
│   Total Time: ────────────► 1x                                              │
│                                                                             │
│   ─────────────────────────────────────────────────────────────────────     │
│                                                                             │
│   Benefits:                                                                 │
│   • 3x faster for 3 concurrent requests                                     │
│   • Better resource utilization                                             │
│   • Scales well in production                                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## AzureSteeringAgent Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `steering_model` | `SteeringModel` | The wrapped model | Required |
| `agent_name` | `str` | Agent identifier | Required |
| `vectors` | `List[Tensor]` | Steering vectors | Required |
| `alpha` | `float` | Steering strength | `2.0` |
| `max_tokens` | `int` | Max generation length | `100` |
| `temperature` | `float` | Sampling temperature | `0.7` |
| `enable_tracing` | `bool` | Enable Azure Monitor | `False` |
| `tracing_config` | `dict` | Tracing configuration | `None` |

---

## Production Deployment Checklist

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Production Deployment Checklist                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ☐ Pre-Deployment                                                          │
│      ☐ Test steering vectors with representative prompts                    │
│      ☐ Validate alpha values don't degrade coherence                        │
│      ☐ Run evaluation benchmarks (toxicity, helpfulness)                    │
│      ☐ Document steering configuration                                      │
│                                                                             │
│   ☐ Azure Configuration                                                     │
│      ☐ Set up Azure AI Foundry workspace                                    │
│      ☐ Configure Application Insights connection string                     │
│      ☐ Set up appropriate RBAC permissions                                  │
│      ☐ Configure networking (VNet if needed)                                │
│                                                                             │
│   ☐ Monitoring                                                              │
│      ☐ Enable tracing in production                                         │
│      ☐ Set up alerts for latency and error rate                             │
│      ☐ Create dashboard for steering metrics                                │
│      ☐ Plan for log retention                                               │
│                                                                             │
│   ☐ Post-Deployment                                                         │
│      ☐ Monitor initial production traffic                                   │
│      ☐ A/B test steering configurations                                     │
│      ☐ Iterate on vectors based on feedback                                 │
│      ☐ Document lessons learned                                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Next Steps

- **RAG applications:** [LlamaIndex RAG Guide](./llamaindex-rag-guide.md)
- **Basic usage:** [Basic Usage Guide](./basic-usage-guide.md)
- **Evaluation:** Check out the evaluation module for measuring steering effectiveness
