# Nirvana

<div class="grid cards" markdown>

-   :material-rocket-launch:{ .lg .middle } __Get Started__

    ---

    Quick installation and basic usage guide

    [:octicons-arrow-right-24: Get Started](get_started.md)

-   :material-school:{ .lg .middle } __Tutorial__

    ---

    Comprehensive guide to using Nirvana

    [:octicons-arrow-right-24: Tutorial](tutorial.md)

-   :material-book-open-page-variant:{ .lg .middle } __API Reference__

    ---

    Complete API documentation

    [:octicons-arrow-right-24: API Reference](api_reference.md)

-   :material-code-tags:{ .lg .middle } __Development__

    ---

    Core concepts and internals for developers

    [:octicons-arrow-right-24: Development](development.md)

</div>

---

## What is Nirvana?

Nirvana is an **LLM-powered semantic data analytics programming framework** that enables natural language queries over structured and unstructured data. It provides a pandas-like interface with semantic operators that use large language models to process data based on natural language instructions.

## Key Features

- **Semantic Operators**: Map, filter, reduce, join, and rank operations powered by LLMs
- **Data Lineage**: Automatic tracking of query plans as directed acyclic graphs
- **Query Optimization**: Logical and physical plan optimization for efficient execution
- **Multimodal Support**: Handle text, images, audio, and other unstructured data types
- **Cost Tracking**: Monitor token costs for each operation

## Quick Example

```python
import pandas as pd
import nirvana as nv

# Configure LLM
nv.configure_llm_backbone(model_name="gpt-4o", api_key="...")

# Create DataFrame
df = nv.DataFrame(pd.DataFrame({
    "title": ["The Godfather", "The Dark Knight"],
    "overview": ["...", "..."]
}))

# Build semantic query
df.semantic_map(
    user_instruction="Extract the genre from the overview",
    input_column="overview",
    output_column="genre"
)
df.semantic_filter(
    user_instruction="Rating is higher than 8.5",
    input_column="IMDB_Rating"
)

# Optimize and execute
output, cost, runtime = df.optimize_and_execute()
```

## Installation

```bash
pip install nirvana-ai
```

Or install from source:

```bash
pip install git+https://github.com/JunHao-Zhu/nirvana.git
```

## Resources

- **Paper**: [arXiv:2511.19830](https://arxiv.org/abs/2511.19830)
- **PyPI**: [nirvana-ai](https://pypi.org/project/nirvana-ai/)
- **GitHub**: [Repository](https://github.com/JunHao-Zhu/nirvana)

## Citation

If you use Nirvana in your research, please cite:

```bibtex
@article{nirvana2024,
  title={Nirvana: An LLM-powered Semantic Data Analytics Framework},
  author={Zhu, Junhao and others},
  journal={arXiv preprint arXiv:2511.19830},
  year={2024}
}
```

