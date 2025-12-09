---
sidebar_position: 1
hide:
  - navigation
  - toc

---

# Nirvana: LLM-powered Semantic Data Analytics Programming Framework
<!--- BADGES: START --->
[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2511.19830)
[![PyPI](https://img.shields.io/pypi/v/nirvana-ai)](https://pypi.org/project/nirvana-ai/)
[![Documentation](https://img.shields.io/badge/Documentation-docs-green)](https://JunHao-Zhu.github.io/nirvana)
<!--- BADGES: END --->

Nirvana is an **LLM-powered semantic data analytics programming framework** that enables semantic data analytics queries over multi-modal data (e.g., text, images, audio). It provides a pandas-like interface with semantic operators that use large language models to process data based on natural language instructions. It also allows an optimizer to find the best execution plan for a given query to strick a balance between quality, runtime, and cost. With Nirvana, users focus only on "what they want to do", instead of "how they achieve it".

<!-- ## Key Features

- **Semantic Operators**: Map, filter, reduce, join, and rank operations powered by LLMs
- **Data Lineage**: Automatic tracking of query plans as directed acyclic graphs
- **Query Optimization**: Logical and physical plan optimization for efficient execution
- **Multimodal Support**: Handle text, images, audio, and other unstructured data types
- **Cost Tracking**: Monitor token costs for each operation -->

!!! info "Step 0: Install nirvana and set up initial llm"
    === "via PyPI"
        ```bash
        pip install nirvana-ai
        ```

    === "via uv"
        ```bash
        uv pip install nirvana-ai
        ```

    === "from source"
        ```bash
        pip install git+https://github.com/JunHao-Zhu/nirvana.git
        ```

    Before you get started with enjoying features of Nirvana, the first thing to do is to set up a default llm. Taking gpt-4o as an example,you can authenticate by setting the `OPENAI_API_KEY` env variable or passing `api_key` below.
    ```python
    import nirvana as nv
    nv.configure_llm_backbone(model_name="gpt-4o", api_key="YOUR_OPENAI_API_KEY")
    ```

## Apply Semantic Operators to DataFrame

Suppose that you have only a simple semantic processing task on hand, for which you want to apply semantic operators to the data and obtain results in a few lines of code as soon as possible. You can easily use function wrappers of semantic operators on your data frame. Here is an example.

!!! info "Extract the genre from the movie overview"

    ```python linenums="1"
    df = pd.DataFrame(
    {
        "title": ["The Godfather", "The Dark Knight"], 
        "overview": [
            "An organized crime dynasty's aging patriarch transfers control of his clandestine empire to his reluctant son.", 
            "When the menace known as the Joker wreaks havoc and chaos on the people of Gotham, Batman must accept one of the greatest psychological and physical tests of his ability to fight injustice."
        ]
    })
    nv.ops.map(df, "According to the movie overview, extract the genre of each movie.", input_column="overview", output_columns = ["genre"], strategy="plain")
    ```

    **Possible Output:**
    ```text
    MapOpOutputs(
        field_name = ["genre"],
        output = {"genre": ["crime, drama", "action, thriller, superhero"]}
    )
    ```

More usages of semantic operators can be found in [operators](https://github.com/JunHao-Zhu/nirvana/api/ops/)

## Enable Query Optimization

If you have a complex semantic query over large datasets on hand, you probabily want to process the query in a faster, lower-cost way. In this case, Nirvana enables lazy execution and query optimization to automatically find a plan that scales down runtime and monetary costs. Here is a usage example.

!!! info ""
    ```python linenums="1"
    movie = nv.DataFrame.from_external_file("/testdata/movie_data.csv")
    movie.semantic_map(user_instruction="According to the movie overview, extract the genre of each movie.", input_column="Overview", output_column="Genre")
    movie.semantic_filter(user_instruction="The rating is higher than 7.", input_column="IMDB_Rating")
    movie.semantic_filter(user_instruction="The rating is lower than 8.", input_column="IMDB_Rating")
    movie.semantic_filter(user_instruction="The movie is a crime movie.", input_column="Genre")
    movie.semantic_reduce(user_instruction="Summerize the common plot structure of these high-rated crime movies.", input_column="Overview")

    config = nv.optim.OptimizeConfig(do_logical_optimization=True, do_physical_optimization=True)
    result, cost, runtime = df.optimize_and_execute(optim_config=config)
    ```
For details and usages of query optimization refers to [optimization](https://github.com/JunHao-Zhu/nirvana/api/optimizer/)
