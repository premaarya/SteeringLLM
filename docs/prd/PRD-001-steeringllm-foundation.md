# PRD: SteeringLLM Foundation - Core Library & Roadmap

**Epic**: #1  
**Status**: Draft  
**Author**: Product Manager Agent  
**Date**: 2026-01-28  
**Stakeholders**: Engineering Team, Research Team

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Target Users](#2-target-users)
3. [Goals & Success Metrics](#3-goals--success-metrics)
4. [Requirements](#4-requirements)
5. [User Stories & Features](#5-user-stories--features)
6. [User Flows](#6-user-flows)
7. [Dependencies & Constraints](#7-dependencies--constraints)
8. [Risks & Mitigations](#8-risks--mitigations)
9. [Timeline & Milestones](#9-timeline--milestones)
10. [Out of Scope](#10-out-of-scope)
11. [Open Questions](#11-open-questions)
12. [Appendix](#12-appendix)

---

## 1. Problem Statement

### What problem are we solving?

Current LLM customization approaches require expensive retraining or fine-tuning, making it impractical to dynamically control model behavior at inference time. Researchers and ML engineers lack tools to modify model behavior in an interpretable, reversible way without permanently altering model weights.

### Why is this important?

**Business Value**: 
- Fine-tuning a 7B model costs $500-2000 per run (compute + time)
- Safety teams need real-time control without redeployment
- Domain adaptation should be fast and reversible

**User Impact**:
- Researchers: Spend 60%+ time on infrastructure vs experimentation
- ML Engineers: Cannot deploy behavioral controls in production reliably  
- Safety Teams: Lack auditable, reversible intervention tools

### What happens if we don't solve this?

- Organizations continue expensive, slow fine-tuning cycles
- Safety interventions remain brittle and hard to audit
- Research on model internals stays fragmented across incompatible tools
- Gap between research (RepE, nnsight) and production remains wide

---

## 2. Target Users

### Primary Users

**User Persona 1: AI Safety Researcher**
- **Demographics**: PhD students, safety research teams at AI labs
- **Goals**: Test safety interventions, understand model failure modes, publish papers
- **Pain Points**: Existing tools are research prototypes, hard to reproduce results, no standardized APIs
- **Behaviors**: Use RepE/nnsight for experiments, manually adapt code for each model

**User Persona 2: ML Engineer (Production)**
- **Demographics**: Industry engineers deploying LLMs in applications
- **Goals**: Implement domain adaptation, safety guardrails, A/B test behaviors
- **Pain Points**: Research code not production-ready, high latency overhead, unclear how to maintain
- **Behaviors**: Fine-tune models (expensive), use prompt engineering (unreliable), avoid steering (too complex)

**User Persona 3: Independent AI Developer**
- **Demographics**: Hobbyists, indie hackers, small startups
- **Goals**: Build domain-specific chatbots without retraining, experiment with model control
- **Pain Points**: Can't afford fine-tuning infrastructure, need simple APIs
- **Behaviors**: Use OpenAI/Anthropic APIs (expensive, no control) or struggle with open models

---

## 3. Goals & Success Metrics

### Business Goals

| Goal | Metric | Target (Year 1) |
|------|--------|-----------------|
| **Adoption** | GitHub stars | 1,000+ |
| **Community** | Active contributors | 20+ |
| **Usage** | PyPI downloads/month | 10,000+ |
| **Integrations** | Projects using SteeringLLM | 5+ |

### User Success Metrics

| Persona | Metric | Target |
|---------|--------|--------|
| **Researchers** | Time to first experiment | <5 minutes |
| **ML Engineers** | Production deployment latency overhead | <10% |
| **Developers** | Setup complexity (lines of code) | <20 lines |

### Technical Metrics (Phase 1)

- **Model Support**: 3+ models (Llama 3.2, Mistral 7B, Gemma 2)
- **Performance**: Steering vector application <50ms (on GPU)
- **Coverage**: 80%+ code test coverage
- **Documentation**: Quick start guide, 2+ example notebooks

---

## 4. Requirements

### Functional Requirements (P0 - Must Have)

**FR-001: Steering Vector Creation**
- Create steering vectors from contrast datasets (positive/negative examples)
- Mean difference discovery method
- Save/load vectors in standard format (JSON + .pt)

**FR-002: Vector Application**
- Apply steering vectors at inference time
- Support alpha parameter (steering strength 0.0-3.0)
- Reversible (remove vector restores baseline behavior)

**FR-003: HuggingFace Integration**
- Load models via AutoModelForCausalLM
- Automatic layer detection for Llama, Mistral, Gemma
- Device placement handling (CPU/CUDA)

**FR-004: Basic Evaluation**
- Before/after comparison utilities
- Text generation with metrics
- Simple visualization of effects

**FR-005: Documentation**
- Installation guide (pip install)
- Quick start tutorial (<5 min to first result)
- API reference
- 2+ example notebooks

### Non-Functional Requirements

**NFR-001: Performance**
- Steering vector application: <50ms on GPU, <200ms on CPU
- Memory overhead: <10% of base model memory

**NFR-002: Usability**
- Installation: `pip install steering-llm`
- Zero-config for supported models
- Clear error messages with actionable guidance

**NFR-003: Reliability**
- Graceful degradation when vector incompatible
- Automatic fallback to base model on error
- No silent failures

---

## 5. User Stories & Features

### Feature 1: Steering Vector Primitives

**Story 1.1: Create steering vector from contrast datasets**

**As a** researcher  
**I want to** create a steering vector from positive/negative example texts  
**So that** I can control model behavior without retraining

**Acceptance Criteria:**
- [ ] Given two text datasets, compute mean activation difference at specified layer
- [ ] Return SteeringVector object with metadata (layer, magnitude, method)
- [ ] Save/load vectors in JSON + .pt format
- [ ] Handle edge cases (empty datasets, invalid layers)

**Story 1.2: Apply steering vector at inference**

**As an** ML engineer  
**I want to** apply a steering vector during text generation  
**So that** I can modify outputs without retraining

**Acceptance Criteria:**
- [ ] Apply vector to target layer(s) via PyTorch hooks
- [ ] Support alpha parameter (0.0 to 3.0+)
- [ ] Generate text with steering enabled
- [ ] Support batch inference

**Story 1.3: Remove steering vector**

**As a** developer  
**I want to** remove steering and revert to baseline  
**So that** I can A/B test or debug issues

**Acceptance Criteria:**
- [ ] Remove vector from model (restore original behavior)
- [ ] Verify outputs match fresh model (same seed)
- [ ] Support removing individual vectors in multi-vector setup

### Feature 2: HuggingFace Integration

**Story 2.1: Load HuggingFace models with auto layer detection**

**As a** user  
**I want to** load any HuggingFace model automatically  
**So that** I don't need model-specific code

**Acceptance Criteria:**
- [ ] Load via SteeringModel.from_pretrained(model_name)
- [ ] Auto-detect transformer layers (Llama, Mistral, Gemma)
- [ ] Handle device placement (CPU/CUDA/MPS)
- [ ] Support quantization (int8, fp16)

**Story 2.2: Inspect model activations**

**As a** researcher  
**I want to** extract activations at any layer  
**So that** I can analyze steering effects

**Acceptance Criteria:**
- [ ] Extract activations for given input at specified layer
- [ ] Support residual stream, attention output, MLP output
- [ ] Return as torch.Tensor with proper device placement

### Feature 3: Basic Evaluation

**Story 3.1: Compare outputs before/after steering**

**As a** user  
**I want to** see side-by-side comparison of steered vs baseline  
**So that** I can validate steering effectiveness

**Acceptance Criteria:**
- [ ] Generate with and without steering for same prompts
- [ ] Compute similarity metrics (BLEU, semantic similarity)
- [ ] Display side-by-side comparison

**Story 3.2: Visualize steering effects**

**As a** researcher  
**I want to** visualize how steering changes generation  
**So that** I can debug unexpected behaviors

**Acceptance Criteria:**
- [ ] Plot token probability distributions before/after
- [ ] Show top-k token changes per generation step
- [ ] Export visualizations as images

---

## 6. User Flows

### Flow 1: Quick Start (5 minutes)

```python
# 1. Install
pip install steering-llm

# 2. Load model
from steering_llm import SteeringModel
model = SteeringModel.from_pretrained("meta-llama/Llama-3.2-3B")

# 3. Create steering vector
from steering_llm import Discovery
vector = Discovery.mean_difference(
    positive=["I love helping!", "You're amazing!"],
    negative=["I hate this.", "You're terrible."],
    model=model,
    layer=15
)

# 4. Generate with steering
output = model.generate_with_steering(
    "Tell me about yourself.",
    vector=vector,
    alpha=2.0
)
print(output)
# => "I'm a helpful AI assistant focused on supporting users..."
```

### Flow 2: Production Deployment

```python
# 1. Load pre-computed safety vector
from steering_llm import SteeringVector, SteeringModel
safety_vector = SteeringVector.load("safety_v1.json")

# 2. Apply to production model
model = SteeringModel.from_pretrained("mistralai/Mistral-7B-v0.1")
model.apply_steering(safety_vector, alpha=1.5)

# 3. Serve via API
from fastapi import FastAPI
app = FastAPI()

@app.post("/generate")
def generate(prompt: str):
    return {"output": model.generate(prompt, max_length=100)}
```

### Flow 3: A/B Testing

```python
# 1. Load model and vectors
model = SteeringModel.from_pretrained("meta-llama/Llama-3.2-3B")
safety_v1 = SteeringVector.load("safety_v1.json")
safety_v2 = SteeringVector.load("safety_v2.json")

# 2. Test on dataset
from steering_llm.evaluation import compare_vectors
results = compare_vectors(
    model=model,
    vectors={"v1": safety_v1, "v2": safety_v2},
    prompts=test_prompts,
    metrics=["toxicity", "helpfulness"]
)

# 3. Analyze
print(results.summary())
# => "v2 reduces toxicity by 12% vs v1 (p<0.01)"
```

---

## 7. Dependencies & Constraints

### Technical Dependencies

| Dependency | Version | Purpose |
|------------|---------|---------|
| Python | 3.9+ | Runtime |
| PyTorch | 2.0+ | Deep learning backend |
| transformers | 4.36+ | Model loading |
| numpy | 1.24+ | Numerical operations |
| pytest | 7.4+ | Testing |

### Resource Constraints

**Compute:**
- GPU: CUDA-capable with ≥16GB VRAM (for 7B models)
- CPU: Supported but 10-50x slower
- Memory: 2x model size (base + steering overhead)

**Storage:**
- Models: 5-50GB per model (HuggingFace cache)
- Vectors: 10-100MB per vector

---

## 8. Risks & Mitigations

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| **Model architecture changes** | High | Medium | Modular architecture, version pinning, adapter pattern |
| **HuggingFace API breaking changes** | High | Low | Pin major versions, integration tests in CI |
| **Performance overhead too high** | Medium | Low | Profile early, optimize hot paths |
| **Low community adoption** | High | Medium | Excellent docs, demos, partner with LangChain/Instructor |

---

## 9. Timeline & Milestones

### Phase 1: Foundation (Q2 2026) - THIS EPIC

**Duration**: 8 weeks  
**Team**: 2 engineers + 1 researcher

| Week | Milestone | Deliverables |
|------|-----------|--------------|
| 1-2 | Setup | Repo structure, CI/CD, dev environment |
| 3-4 | Core Library | SteeringVector, ActivationHook, mean difference |
| 5-6 | HF Integration | SteeringModel, Llama/Mistral/Gemma support |
| 7-8 | Documentation | Quick start, API docs, 2 example notebooks |

**Success Criteria:**
- [ ] Create/apply steering in <20 lines of code
- [ ] Support 3+ models (Llama 3.2, Mistral, Gemma)
- [ ] 50+ GitHub stars
- [ ] 80%+ test coverage

### Future Phases (Roadmap)

**Phase 2: Integration (Q3 2026)**
- Advanced discovery (CAA, linear probing)
- 10+ model support
- Multi-vector composition

**Phase 3: Tools (Q4 2026)**
- Safety benchmarks (ToxiGen, RealToxicityPrompts)
- Visualization dashboard
- Agent integration (LangChain, LlamaIndex)

**Phase 4: Production (Q1 2027)**
- Performance optimization (<10ms overhead)
- Cloud deployment (Docker, SageMaker)
- Enterprise features

---

## 10. Out of Scope

### For Phase 1 (Foundation)

- ❌ Closed-source models (GPT-4, Claude) - focus on open models
- ❌ Multi-modal steering (vision, audio) - text-only
- ❌ Advanced discovery methods (CAA, probing) - mean difference only
- ❌ Production optimization - prototype performance acceptable
- ❌ Web UI/dashboard - CLI/notebook only

### Future Considerations

- Vision-language models (LLaVA, CLIP steering)
- Automated vector optimization via RL
- Federated learning for privacy-preserving steering
- Commercial support/SLAs

---

## 11. Open Questions

**Technical:**

1. **Q**: Should we support GGUF format (llama.cpp) or only PyTorch?
   - **Decision Needed**: Week 2 (impacts architecture)
   - **Options**: PyTorch-only (simpler), GGUF support (broader reach)

2. **Q**: Vector storage format - JSON+.pt vs HDF5 vs custom?
   - **Decision Needed**: Week 3 (impacts API)
   - **Recommendation**: JSON+.pt (human-readable + efficient)

3. **Q**: Should steering vectors be model-specific or transferable?
   - **Decision Needed**: Week 6 (research question)
   - **Impact**: If transferable, enables "vector zoo" for reuse

**Product:**

4. **Q**: Should we provide pre-computed steering vectors (safety, helpfulness)?
   - **Decision Needed**: Post-Phase 1
   - **Trade-off**: Lower barrier vs curation burden

5. **Q**: Licensing - Apache 2.0 vs MIT?
   - **Decision Needed**: Week 1
   - **Recommendation**: Apache 2.0 (patent protection)

---

## 12. Appendix

### A. Glossary

| Term | Definition |
|------|------------|
| **Steering Vector** | Direction in activation space that modifies behavior when added |
| **Activation Engineering** | Techniques for manipulating internal model representations |
| **Mean Difference** | Computing vector as mean(positive) - mean(negative) |
| **Layer** | Single transformer block in model architecture |
| **Residual Stream** | Main information flow through transformer layers |

### B. References

**Papers:**
- "Representation Engineering" (Zou et al., 2023)
- "Steering Llama 2 via Contrastive Activation Addition" (Turner et al., 2023)

**Tools:**
- RepE: https://github.com/andyzoujm/representation-engineering
- nnsight: https://github.com/ndif-team/nnsight
- steering-vectors: https://github.com/steering-vectors/steering-vectors

### C. Success Stories (Target)

**Story 1: AI Startup**
- Built medical chatbot by steering Llama 3.2
- 40% accuracy improvement vs base model
- Zero fine-tuning cost

**Story 2: Safety Team**
- Applied safety steering to reduce toxicity 85%
- No capability degradation
- Real-time intervention possible

---

**Document Status**: Draft  
**Next Review**: Week 2 (Architecture feedback)  
**Approved By**: TBD
