# LlamaIndex RAG Integration Guide

> **File:** [examples/llamaindex_rag_steering.py](../../examples/llamaindex_rag_steering.py)

This guide demonstrates how to integrate SteeringLLM with LlamaIndex for building steered Retrieval-Augmented Generation (RAG) applications.

---

## Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       RAG + Steering Architecture                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   User Query: "What is hypertension?"                                       │
│              │                                                              │
│              ▼                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                        LlamaIndex                                   │   │
│   │                                                                     │   │
│   │   ┌──────────────────┐    ┌───────────────────────────────────┐    │   │
│   │   │  Vector Store    │    │         Query Engine              │    │   │
│   │   │  (Retrieval)     │───►│  ┌─────────────────────────────┐  │    │   │
│   │   │                  │    │  │   LlamaIndexSteeringLLM     │  │    │   │
│   │   │  "Hypertension   │    │  │                             │  │    │   │
│   │   │   is chronic..." │    │  │   medical_vector applied    │  │    │   │
│   │   │                  │    │  │   → Clinical terminology    │  │    │   │
│   │   └──────────────────┘    │  │   → Professional tone       │  │    │   │
│   │                           │  └─────────────────────────────┘  │    │   │
│   │                           └───────────────────────────────────┘    │   │
│   │                                                                     │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│              │                                                              │
│              ▼                                                              │
│   "Hypertension, a chronic cardiovascular condition characterized by..."    │
│   (Domain-appropriate response with clinical terminology)                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

```bash
# Install SteeringLLM with agent support
pip install steering-llm[agents]

# Or install LlamaIndex separately
pip install llama-index llama-index-embeddings-huggingface
```

---

## Example 1: Basic RAG with Steering

### Concept

Enhance RAG responses with steering to ensure technical, precise explanations.

### Implementation

```python
from llama_index.core import VectorStoreIndex, Document
from steering_llm import SteeringModel, Discovery
from steering_llm.agents import LlamaIndexSteeringLLM
```

### Step 1: Load Model and Create Steering Vector

```python
# Load model
steering_model = SteeringModel.from_pretrained(
    "gpt2",
    device_map="auto"
)

# Create technical steering vector
technical_vector = Discovery.mean_difference(
    positive=[
        "Technical documentation and precise explanations.",
        "Detailed analysis with specific terminology.",
        "Comprehensive technical overview.",
    ],
    negative=[
        "Simple casual explanation.",
        "Basic overview without details.",
        "Informal discussion.",
    ],
    model=steering_model.model,
    layer=12
)
```

### Step 2: Create Steered LLM

```python
llm = LlamaIndexSteeringLLM(
    steering_model=steering_model,
    vectors=[technical_vector],
    alpha=1.8,
    max_tokens=200,
    temperature=0.7
)
```

### Step 3: Build Index and Query

```python
# Create documents
documents = [
    Document(text="Python is a high-level programming language known for its simplicity."),
    Document(text="Machine learning enables systems to learn from data."),
    Document(text="Neural networks are inspired by biological neural networks."),
]

# Build vector store index
index = VectorStoreIndex.from_documents(documents)

# Create query engine with steered LLM
query_engine = index.as_query_engine(llm=llm)

# Query
response = query_engine.query("What is Python?")
print(response)
```

### RAG Query Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          RAG Query Flow                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Query: "What is Python?"                                                  │
│              │                                                              │
│              ▼                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  Step 1: RETRIEVAL                                                  │   │
│   │  ──────────────────                                                 │   │
│   │  • Query embedded using embedding model                             │   │
│   │  • Vector similarity search in index                                │   │
│   │  • Top-k relevant documents retrieved                               │   │
│   │                                                                     │   │
│   │  Retrieved: "Python is a high-level programming language..."        │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│              │                                                              │
│              ▼                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  Step 2: AUGMENTATION                                               │   │
│   │  ────────────────────                                               │   │
│   │  • Context + Query combined into prompt                             │   │
│   │  • "Given the following context: {context}                          │   │
│   │     Answer the question: {query}"                                   │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│              │                                                              │
│              ▼                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  Step 3: GENERATION (with Steering)                                 │   │
│   │  ──────────────────────────────────                                 │   │
│   │  • LlamaIndexSteeringLLM processes prompt                           │   │
│   │  • technical_vector applied at layer 12                             │   │
│   │  • Response generated with technical precision                      │   │
│   │                                                                     │   │
│   │  Output: "Python is a high-level, interpreted programming           │   │
│   │          language with dynamic typing and garbage collection..."    │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Example 2: Domain-Adapted RAG

### Concept

Create a medical RAG system that responds with clinical terminology and professional tone.

### Implementation

```python
from steering_llm.agents import create_rag_steering_llm

# Create medical domain vector (using CAA for stronger effect)
medical_vector = Discovery.caa(
    positive=[
        "The patient presented with clinical symptoms requiring diagnosis.",
        "Medical assessment indicated treatment protocols.",
        "Clinical evidence supports therapeutic intervention.",
    ],
    negative=[
        "The person was sick and went to the doctor.",
        "They felt bad and got medicine.",
        "The doctor helped them feel better.",
    ],
    model=steering_model.model,
    layer=15
)

# Create domain-adapted RAG LLM
llm = create_rag_steering_llm(
    steering_model=steering_model,
    domain_vector=medical_vector,
    alpha=2.0,
    max_tokens=250,
    temperature=0.5  # Lower temperature for medical accuracy
)

# Create medical knowledge base
documents = [
    Document(text="Hypertension is a chronic medical condition characterized by elevated blood pressure."),
    Document(text="Diabetes mellitus involves impaired insulin production or insulin resistance."),
    Document(text="Cardiac arrhythmias are irregular heart rhythms that may require intervention."),
]

# Build and query
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(llm=llm)

response = query_engine.query("What is hypertension?")
```

### Response Comparison

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                   Medical Domain Steering Effect                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Query: "What is hypertension?"                                            │
│                                                                             │
│   ┌────────────────────────────────────────────────────────────────────┐    │
│   │                     WITHOUT STEERING                               │    │
│   ├────────────────────────────────────────────────────────────────────┤    │
│   │                                                                    │    │
│   │  "Hypertension is when your blood pressure is too high.           │    │
│   │   It means the force of blood against your artery walls           │    │
│   │   is consistently elevated. This can cause health problems."      │    │
│   │                                                                    │    │
│   │  Tone: Casual, simplified                                         │    │
│   │  Vocabulary: Everyday language                                    │    │
│   │                                                                    │    │
│   └────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│   ┌────────────────────────────────────────────────────────────────────┐    │
│   │                      WITH MEDICAL STEERING                         │    │
│   ├────────────────────────────────────────────────────────────────────┤    │
│   │                                                                    │    │
│   │  "Hypertension, clinically defined as sustained systolic          │    │
│   │   blood pressure ≥140 mmHg or diastolic ≥90 mmHg, represents      │    │
│   │   a chronic cardiovascular condition with significant morbidity.  │    │
│   │   Primary hypertension accounts for 90-95% of cases, with         │    │
│   │   secondary causes requiring differential diagnosis."             │    │
│   │                                                                    │    │
│   │  Tone: Clinical, professional                                     │    │
│   │  Vocabulary: Medical terminology                                  │    │
│   │                                                                    │    │
│   └────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Example 3: Multi-Vector RAG

### Concept

Apply multiple steering vectors simultaneously: domain expertise + formal style + safety.

### Implementation

```python
from steering_llm.agents import create_multi_vector_rag_llm

# Create multiple steering vectors
domain_vector = Discovery.caa(
    positive=["Technical and precise explanations."],
    negative=["Vague or imprecise descriptions."],
    model=steering_model.model,
    layer=10
)

style_vector = Discovery.mean_difference(
    positive=["Formal professional writing style."],
    negative=["Casual conversational style."],
    model=steering_model.model,
    layer=15
)

safety_vector = Discovery.caa(
    positive=["Safe and appropriate content."],
    negative=["Inappropriate or harmful content."],
    model=steering_model.model,
    layer=18
)

# Create multi-vector RAG LLM
llm = create_multi_vector_rag_llm(
    steering_model=steering_model,
    vectors=[domain_vector, style_vector, safety_vector],
    weights=[0.5, 0.3, 0.2],  # 50% domain, 30% style, 20% safety
    composition_method="weighted",
    max_tokens=300
)
```

### Multi-Vector Composition

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     Multi-Vector Composition                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────┐                                                       │
│   │  Domain Vector  │────┐   weight: 0.5 (50%)                              │
│   │  (Layer 10)     │    │   → Technical precision                          │
│   └─────────────────┘    │                                                  │
│                          │                                                  │
│   ┌─────────────────┐    │   ┌─────────────────────────────────────────┐    │
│   │  Style Vector   │────┼──►│                                         │    │
│   │  (Layer 15)     │    │   │     Weighted Combination                │    │
│   └─────────────────┘    │   │                                         │    │
│   weight: 0.3 (30%)      │   │  combined = Σ(weight_i × vector_i)      │    │
│   → Formal tone          │   │                                         │    │
│                          │   └──────────────────┬──────────────────────┘    │
│   ┌─────────────────┐    │                      │                           │
│   │  Safety Vector  │────┘                      │                           │
│   │  (Layer 18)     │                           │                           │
│   └─────────────────┘                           │                           │
│   weight: 0.2 (20%)                             │                           │
│   → Safe content                                │                           │
│                                                 ▼                           │
│                               ┌────────────────────────────────────┐        │
│                               │  Combined Steering Effect          │        │
│                               │                                    │        │
│                               │  ✅ Technical accuracy             │        │
│                               │  ✅ Professional language          │        │
│                               │  ✅ Safety guardrails              │        │
│                               └────────────────────────────────────┘        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Example 4: Context-Aware RAG

### Concept

Dynamically switch steering based on user expertise level (beginner vs expert).

### Implementation

```python
# Create context-specific vectors
beginner_vector = Discovery.mean_difference(
    positive=["Simple explanations for beginners."],
    negative=["Complex technical jargon."],
    model=steering_model.model,
    layer=10
)

expert_vector = Discovery.caa(
    positive=["Advanced technical details for experts."],
    negative=["Oversimplified explanations."],
    model=steering_model.model,
    layer=12
)

# Create LLMs for each context
beginner_llm = LlamaIndexSteeringLLM(
    steering_model=steering_model,
    vectors=[beginner_vector],
    alpha=2.0,
    max_tokens=150
)

expert_llm = LlamaIndexSteeringLLM(
    steering_model=steering_model,
    vectors=[expert_vector],
    alpha=2.0,
    max_tokens=150
)

# Build index
index = VectorStoreIndex.from_documents(documents)

# Query with different contexts
beginner_engine = index.as_query_engine(llm=beginner_llm)
expert_engine = index.as_query_engine(llm=expert_llm)

query = "What is programming?"

print("Beginner Mode:")
print(beginner_engine.query(query))

print("\nExpert Mode:")
print(expert_engine.query(query))
```

### Context-Aware Response Comparison

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Context-Aware Responses                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Query: "What is programming?"                                             │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                      BEGINNER MODE                                  │   │
│   │                      (beginner_vector applied)                      │   │
│   ├─────────────────────────────────────────────────────────────────────┤   │
│   │                                                                     │   │
│   │  "Programming is like giving instructions to a computer.           │   │
│   │   Just like you follow a recipe to make food, a computer           │   │
│   │   follows your instructions to do tasks. You write these           │   │
│   │   instructions in special languages like Python or JavaScript."    │   │
│   │                                                                     │   │
│   │  ✅ Analogies (recipe, instructions)                                │   │
│   │  ✅ Simple vocabulary                                               │   │
│   │  ✅ Concrete examples                                               │   │
│   │                                                                     │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                       EXPERT MODE                                   │   │
│   │                       (expert_vector applied)                       │   │
│   ├─────────────────────────────────────────────────────────────────────┤   │
│   │                                                                     │   │
│   │  "Programming involves the design and implementation of            │   │
│   │   algorithms expressed through formal language constructs.         │   │
│   │   Modern paradigms include object-oriented, functional,            │   │
│   │   and concurrent programming models, each with distinct            │   │
│   │   computational semantics and type systems."                       │   │
│   │                                                                     │   │
│   │  ✅ Technical terminology                                           │   │
│   │  ✅ Formal definitions                                              │   │
│   │  ✅ Advanced concepts                                               │   │
│   │                                                                     │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Example 5: Streaming RAG Responses

### Concept

Stream responses for better user experience in interactive applications.

### Implementation

```python
# Create steered LLM
llm = LlamaIndexSteeringLLM(
    steering_model=steering_model,
    vectors=[detailed_vector],
    alpha=1.5,
    max_tokens=200
)

# Build index and query engine
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(llm=llm, streaming=True)

# Stream response
query = "What is quantum computing?"
streaming_response = query_engine.query(query)

# Print tokens as they arrive
for text in streaming_response.response_gen:
    print(text, end="", flush=True)
```

### Streaming Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Streaming Response Flow                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   User                            Server                                    │
│     │                               │                                       │
│     │  Query: "What is...?"         │                                       │
│     │ ─────────────────────────────►│                                       │
│     │                               │                                       │
│     │                               │ ┌─────────────────────────────────┐   │
│     │                               │ │ 1. Retrieve documents           │   │
│     │                               │ │ 2. Prepare context              │   │
│     │                               │ │ 3. Start generation with        │   │
│     │                               │ │    steering                     │   │
│     │                               │ └─────────────────────────────────┘   │
│     │                               │                                       │
│     │  Token: "Quantum"             │                                       │
│     │ ◄─────────────────────────────│                                       │
│     │  Token: " computing"          │                                       │
│     │ ◄─────────────────────────────│                                       │
│     │  Token: " utilizes"           │                                       │
│     │ ◄─────────────────────────────│                                       │
│     │  Token: " quantum"            │                                       │
│     │ ◄─────────────────────────────│                                       │
│     │  ...                          │                                       │
│     │                               │                                       │
│     │  Token: [END]                 │                                       │
│     │ ◄─────────────────────────────│                                       │
│                                                                             │
│   Benefits:                                                                 │
│   • User sees response immediately                                          │
│   • Better perceived performance                                            │
│   • Can cancel early if needed                                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## LlamaIndexSteeringLLM Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `steering_model` | `SteeringModel` | The wrapped model | Required |
| `vectors` | `List[Tensor]` | Steering vectors | Required |
| `alpha` | `float` | Steering strength | `2.0` |
| `max_tokens` | `int` | Max generation length | `200` |
| `temperature` | `float` | Sampling temperature | `0.7` |
| `context_window` | `int` | Context size | `2048` |

---

## RAG + Steering Best Practices

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     RAG + Steering Best Practices                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Document Preparation                                                      │
│   ─────────────────────                                                     │
│   ✅ Chunk documents appropriately (512-1024 tokens)                        │
│   ✅ Include metadata for filtering                                         │
│   ✅ Use domain-appropriate embeddings                                      │
│                                                                             │
│   Steering Configuration                                                    │
│   ──────────────────────                                                    │
│   ✅ Match steering to document domain                                      │
│   ✅ Use lower alpha (1.5-2.0) to preserve factuality                       │
│   ✅ Test with representative queries                                       │
│                                                                             │
│   Response Quality                                                          │
│   ────────────────                                                          │
│   ✅ Lower temperature (0.3-0.7) for factual RAG                            │
│   ✅ Include source attribution                                             │
│   ✅ Validate responses against retrieved content                           │
│                                                                             │
│   Performance                                                               │
│   ───────────                                                               │
│   ✅ Use streaming for better UX                                            │
│   ✅ Cache steering vectors (they're reusable)                              │
│   ✅ Consider async for concurrent queries                                  │
│                                                                             │
│   ❌ AVOID                                                                  │
│   ─────────                                                                 │
│   • Very high alpha (>3.0) - may hallucinate                                │
│   • Conflicting steering vectors                                            │
│   • Steering that contradicts retrieved facts                               │
│   • Same steering for all domains                                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Common RAG + Steering Patterns

### Pattern 1: Domain-Specific RAG

```python
# Create domain experts for different document collections
medical_rag = create_rag_with_steering(medical_docs, medical_vector)
legal_rag = create_rag_with_steering(legal_docs, legal_vector)
technical_rag = create_rag_with_steering(tech_docs, technical_vector)
```

### Pattern 2: Audience-Adaptive RAG

```python
# Detect user expertise and adapt
def get_rag_for_user(user_profile, index):
    if user_profile.is_expert:
        return index.as_query_engine(llm=expert_llm)
    else:
        return index.as_query_engine(llm=beginner_llm)
```

### Pattern 3: Safety-First RAG

```python
# Always apply safety, optionally add domain
llm = create_multi_vector_rag_llm(
    steering_model=model,
    vectors=[safety_vector, domain_vector],
    weights=[0.4, 0.6],  # Safety always present
)
```

---

## Next Steps

- **Basic usage:** [Basic Usage Guide](./basic-usage-guide.md)
- **LangChain integration:** [LangChain Guide](./langchain-integration-guide.md)
- **Azure deployment:** [Azure Agent Guide](./azure-agent-guide.md)
