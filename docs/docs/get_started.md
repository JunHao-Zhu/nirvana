# Get Started

Nirvana is an LLM-powered semantic data analytics programming framework that enables natural language queries over structured and unstructured data.

## Installation

Install Nirvana from PyPI:

```bash
pip install nirvana-ai
```

Or install the latest version from the main branch:

```bash
pip install git+https://github.com/JunHao-Zhu/nirvana.git
```

## Quick Start

### 1. Configure the LLM Backbone

Before using semantic operators, configure the LLM settings:

```python
import nirvana as nv

nv.configure_llm_backbone(
    model_name="gpt-4o",
    api_key="<Your API Key>",
    base_url=None  # Optional, inferred from model_name
)
```

Supported model prefixes:
- `gpt-*` or `text-embedding-*`: OpenAI (defaults to `OPENAI_API_KEY`)
- `deepseek-*`: DeepSeek (defaults to `DEEPSEEK_API_KEY`)
- `qwen-*`: Qwen (defaults to `QWEN_API_KEY`)
- `gemini-*`: Gemini (defaults to `GEMINI_API_KEY`)

### 2. Create a DataFrame

Nirvana's `DataFrame` extends pandas DataFrame with support for unstructured data types (images, text, audio):

```python
import pandas as pd
import nirvana as nv

# Create from a pandas DataFrame
df = pd.DataFrame({
    "title": ["The Godfather", "The Dark Knight"],
    "overview": [
        "An organized crime dynasty's aging patriarch transfers control...",
        "When the menace known as the Joker wreaks havoc..."
    ]
})
df = nv.DataFrame(df)

# Or load from a file
df = nv.DataFrame.from_external_file("movie_data.csv")
```

### 3. Apply Semantic Operations

Build a query using semantic operators:

```python
# Extract genre from movie overviews
df.semantic_map(
    user_instruction="According to the movie overview, extract the genre of each movie.",
    input_column="overview",
    output_column="genre"
)

# Filter movies with high ratings
df.semantic_filter(
    user_instruction="The rating is higher than 8.5.",
    input_column="IMDB_Rating"
)

# Aggregate results
df.semantic_reduce(
    user_instruction="Count the number of crime movies.",
    input_column="genre"
)
```

### 4. Optimize and Execute

Optimize the query plan and execute:

```python
# Configure optimization
config = nv.optim.OptimizeConfig(
    do_logical_optimization=True,
    do_physical_optimization=True,
    sample_size=5,
    improve_margin=0.2
)

# Execute with optimization
output, cost, runtime = df.optimize_and_execute(optim_config=config)

print(f"Output: {output}")
print(f"Cost: ${cost:.4f}")
print(f"Runtime: {runtime:.2f}s")
```

## What's Next?

- Learn about [semantic operators](tutorial.md#operators) in detail
- Explore [query optimization](tutorial.md#plan-optimization) features
- Understand [data lineage](development.md) concepts
- Check the [API reference](api_reference.md) for detailed documentation

