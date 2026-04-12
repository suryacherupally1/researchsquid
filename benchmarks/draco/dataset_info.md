# DRACO: A Cross-Domain Benchmark for Deep Research

> Source: https://huggingface.co/datasets/perplexity-ai/draco  
> ArXiv: [2602.11685](https://arxiv.org/abs/2602.11685)  
> License: MIT | Size: 100 tasks | Language: English

---

## Overview

DRACO is a benchmark dataset for evaluating deep research systems on complex, open-ended tasks with expert-curated rubrics. It contains **100 tasks** spanning **10 domains** with evaluation criteria focused on factual accuracy, analytical depth, presentation quality, and citation quality.

**Key Details:**
- **Creator:** Perplexity AI
- **License:** MIT
- **Size:** 100 rows (< 1K)
- **Format:** JSON/Parquet (single `test.jsonl` split)
- **Languages:** English

---

## Dataset Characteristics

### Domain Distribution

| Domain | Share | Avg Criteria |
|--------|-------|--------------|
| Finance | 20% | 47.6 |
| Shopping/Product Comparison | 16% | 39.7 |
| Academic | 12% | 41.6 |
| Technology | 10% | 36.7 |
| General Knowledge | 9% | 39.2 |
| UX Design | 9% | 36.9 |
| Law | 6% | 33.2 |
| Medicine | 6% | 33.7 |
| Needle in a Haystack | 6% | 30.2 |
| Personalized Assistant | 6% | 35.5 |

### Task Characteristics

- Tasks originate from real Perplexity Deep Research user queries (Sept–Oct 2025)
- Sampled from queries where users expressed dissatisfaction, ensuring difficulty
- Require multi-hop agentic retrieval, synthesis across heterogeneous sources, and domain expertise
- Vary along 6 dimensions: persona, output format, source specificity, temporal scope, cross-entity comparison, geographic breadth

---

## Rubric Structure

Each task includes a rubric with criteria organized into **4 evaluation axes**:

### Evaluation Axes

| Axis | Section ID | Weight Range | Avg Criteria | % of Total |
|------|-----------|--------------|--------------|-----------|
| Factual Accuracy | `factual-accuracy` | -500 to +20 | 20.5 | 52% |
| Breadth & Depth | `breadth-and-depth-of-analysis` | -100 to +10 | 8.6 | 22% |
| Presentation Quality | `presentation-quality` | -50 to +20 | 5.6 | 14% |
| Citation Quality | `citation-quality` | -150 to +10 | 4.8 | 12% |

**Key Points:**
- 3,934 total criteria across all 100 tasks
- 415 criteria (10.5%) carry negative weights
- Most severe penalties (-500) reserved for dangerous medical content
- Non-medical penalties typically range from -10 to -25
- ~45% of rubrics revised through saturation testing
- Current best-system saturation: ~71% (substantial headroom)

---

## Data Format

**Single file:** `test.jsonl` (100 entries)

### Entry Structure

```json
{
  "id": "string (UUID)",
  "domain": "string",
  "problem": "string (research query)",
  "answer": "string (JSON-encoded rubric)"
}
```

### Parsed Rubric Format

```json
{
  "id": "string (slug)",
  "sections": [
    {
      "id": "factual-accuracy | breadth-and-depth-of-analysis | presentation-quality | citation-quality",
      "title": "string",
      "criteria": [
        {
          "id": "string (criterion slug)",
          "weight": "integer",
          "requirement": "string (evaluation instruction)"
        }
      ]
    }
  ]
}
```

---

## Evaluation Methodology

### Grading Protocol

- Uses **LLM-as-a-judge** approach
- Binary verdicts: **MET** or **UNMET** for each criterion
- Judge receives: original query + system response + criterion

**Criterion Types:**
- **Positive weight:** MET = criterion satisfied, UNMET = not satisfied
- **Negative weight:** MET = error present, UNMET = error absent

### Scoring Formula

```
raw_score = sum(v_i * w_i for all i)
normalized_score = clamp(raw_score / sum(w_i for w_i > 0), 0, 1) * 100%
```

Where `v_i = 1` if MET, `v_i = 0` if UNMET.

**Range:** 0–100%

**Note:** Negative-weight criteria reduce raw score when MET, allowing scores below positive-criteria performance alone.

---

## Sample Tasks

### Example 1: Finance
**Problem:** Analyze CME Group's cash generation efficiency...  
**Criteria examples:** Operating cash flow Q1 2025 ($1,116.6m), OCF conversion rates, capital allocation analysis

### Example 2: Medicine
**Problem:** Rescue inhaler adverse reaction analysis  
**Criteria examples:** Abnormal adjustment warnings, symptom documentation, medical guidance appropriateness

### Example 3: Technology
**Problem:** Compare YOLO v8, EfficientDet-D4, NVIDIA TAO DetectNet_v2...  
**Criteria examples:** INT8 quantized inference latency, mAP degradation, deployment feasibility

---

## Intended Use

- Evaluating deep research systems (agentic research agents)
- Measuring factual accuracy in long-form outputs
- Assessing analytical depth and breadth
- Evaluating presentation quality and citation practices
- Domain-specific benchmarking

---

## Limitations

- **Domain coverage:** Reflects observed usage, not exhaustive coverage
- **Static snapshot:** Information accurate to late 2025 construction period
- **LLM judge variance:** Relative rankings stable; absolute scores vary by judge model
- **Results context:** Compare within consistent judge configurations

---

## Citation

```bibtex
@misc{draco2026,
  title={DRACO: A Cross-Domain Benchmark for Deep Research Accuracy, Completeness, and Objectivity},
  author={Joey Zhong and Hao Zhang and Clare Southern and Jeremy Yang and Thomas Wang and Kate Jung and Shu Zhang and Denis Yarats and Johnny Ho and Jerry Ma},
  year={2026},
  url={https://arxiv.org/abs/2602.11685}
}
```
