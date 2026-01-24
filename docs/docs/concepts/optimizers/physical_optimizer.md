# Physical Optimizer

The **Physical Optimizer** is responsible for selecting the concrete execution strategy for each operator in the logical plan. Its primary job is **Model Selection**.

## Model Selection Strategy

Nirvana supports a hierarchy of models (e.g., from small/fast models like `gpt-4o-mini` to large/smart models like `gpt-4`). The physical optimizer determines the cheapest model that achieves acceptable quality for each specific operator.

### Algorithm: Cascading Refinement

The optimizer iteratively tests models on a small sample of data:

1.  **Baseline**: Run the operator with the "Weakest" (cheapest) model and the "Strongest" (most expensive, ground truth) model.
2.  **Comparison**: Compare the outputs.
    - If the Weakest model matches the Strongest model (high similarity/accuracy), use the Weakest model.
    - If not, try the next model in the hierarchy.
3.  **Improvement Margin**: The optimizer selects a stronger model only if it provides a significant quality gain (above `improve_margin`) relative to the cost increase.

### Consistency Check
For **Map** operations, output consistency is often measured using **Embedding Similarity**.
- It creates embeddings for the outputs of different models.
- If the cosine similarity is high, the outputs are considered semantically equivalent, even if the text differs slightly.

## Execution

The `PhysicalOptimizer.optimize()` method returns a plan where each operator node is annotated with the specific `model` to use during execution.

```python
# In nirvana/optim/physical.py
node.operator.model = selected_model
```
