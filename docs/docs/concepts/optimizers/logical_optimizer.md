# Logical Optimizer

The **Logical Optimizer** in Nirvana transforms the logical execution plan (the lineage DAG) to reduce execution cost and improve performance *before* any data is processed. It applies a series of rewrite rules that are semantically equivalent but more efficient.

## Optimization Goal
To find a plan $P'$ such that $Cost(P') < Cost(P)$ while maintaining high accuracy (Error < Tolerance).

## Core Rules

### 1. Filter Pushdown
Moves filter operations earlier in the pipeline (closer to the data source).
- **Benefit**: Reduces the amount of data processed by subsequent expensive operators (like Map or Join).

### 2. Filter Pullup
Moves filter operations later in the pipeline.
- **Benefit**: Combining with Filter Pushdown, search for more optimization possibilities (like first pull filters up to the final merged plans (by joins) and then push them down to the both sub-plans to reduce the amount of data processed by subsequent expensive operators).

### 3. Map Pullup
Reorders map operations relative to others.
- **Map Pullup**: Delays mapping until after filtering, to avoid mapping rows that will be discarded.

### 4. Non-LLM Pushdown
Pushes purely deterministic predicates (for example, numeric comparisons) ahead of LLM operators by compiling them to Python UDFs.
- **Benefit**: Lets cheap, deterministic checks eliminate rows before any LLM call is issued.

### 5. Non-LLM Replacement
Replaces LLM-based operators with cheaper, deterministic Python functions or regexes when possible.
- **Example**: If a filter asks for "numbers > 5", it can be rewritten as a simple lambda logic instead of an LLM call.

### 6. Operator Fusion
Combines adjacent operators into a single LLM call.
- **Benefit**: Reduces API overhead and context limitations. For example, combining a `Filter` and a `Map` into a single "Extract and Filter" prompt.

## Optimization Loop

The `LogicalOptimizer` uses a search-based approach:

1.  **Sampling**: It samples candidate plans.
2.  **Estimation**: It runs the plan on a small "dev" dataset to estimate cost and accuracy.
3.  **Selection**: It applies transformation rules to generate new candidates and keeps the best ones (Pareto frontier of Cost vs Accuracy).

```python
# Simplified Logic
while round < max_rounds:
    candidate = sample_candidate()
    new_plan = apply_rule(candidate, rule_name)
    cost, accuracy = estimate(new_plan)
    candidates.append(new_plan)
```
