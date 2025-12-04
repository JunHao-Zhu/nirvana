# Development

This section covers core concepts and internals of Nirvana for developers who want to understand or extend the framework.

## Data Lineage

Data lineage is the core abstraction in Nirvana that enables lazy execution, query optimization, and cost tracking.

### Overview

Data lineage is represented as a **directed acyclic graph (DAG)** where:
- **Nodes** represent operators (scan, map, filter, join, reduce, rank)
- **Edges** represent data flow between operators
- Each node tracks input/output fields and dependencies

### LineageNode

The `LineageNode` class (in `nirvana/lineage/abstractions.py`) is the fundamental building block:

```python
class LineageNode(NodeBase):
    def __init__(
        self,
        op_name: str,
        op_kwargs: dict,
        node_fields: dict,
        datasource: pd.DataFrame | None = None,
        **kwargs
    ):
        self.operator = op_mapping[op_name](**op_kwargs)
        self.node_fields = NodeFields(**node_fields)
        self.datasource = datasource
        self._left_child = None
        self._right_child = None
```

**Key Components:**

1. **Operator**: The actual operation instance (MapOperation, FilterOperation, etc.)
2. **NodeFields**: Tracks input and output fields:
   ```python
   @dataclass
   class NodeFields:
       left_input_fields: list[str]
       right_input_fields: list[str]
       output_fields: list[str]
   ```
3. **Child Nodes**: Left and right children for binary operations (e.g., join)

### Building Lineage

When you call semantic operations on a DataFrame, nodes are added to the lineage:

```python
df = nv.DataFrame(data)

# Creates a scan node
df.initialize()  # Called automatically in __init__

# Adds a map node
df.semantic_map(...)  # Creates LineageNode with MapOperation

# Adds a filter node
df.semantic_filter(...)  # Creates LineageNode with FilterOperation
```

The `LineageMixin` (in `nirvana/lineage/mixin.py`) provides the lineage management:

```python
class LineageMixin:
    def initialize(self):
        # Create scan node
        node = LineageNode(
            op_name="scan",
            op_kwargs={"source": "dataframe", "output_columns": self.columns},
            node_fields={"left_input_fields": [], "right_input_fields": [], "output_fields": self.columns},
            datasource=self._data
        )
        self.leaf_node = node
    
    def add_operator(self, op_name: str, op_kwargs: dict, data_kwargs: dict, **kwargs):
        node = LineageNode(op_name, op_kwargs=op_kwargs, node_fields=data_kwargs)
        if op_name == "join":
            node.set_left_child(self.leaf_node)
            node.set_right_child(kwargs["other"].leaf_node)
        else:
            node.set_left_child(self.leaf_node)
        self.leaf_node = node
```

### Execution Model

Execution follows a **post-order traversal** of the lineage graph:

```python
def execute_along_lineage(leaf_node: LineageNode):
    def _execute_node(node: LineageNode) -> pd.DataFrame:
        # Recursively execute children first
        if node.left_child:
            left_output = _execute_node(node.left_child)
        if node.right_child:
            right_output = _execute_node(node.right_child)
        
        # Execute current node
        if node.op_name == "scan":
            return node.datasource
        elif node.op_name == "join":
            return node.run([left_output, right_output])
        else:
            return node.run(left_output)
    
    return _execute_node(leaf_node)
```

**Execution Flow:**

1. **Scan**: Returns the datasource DataFrame
2. **Unary Operations** (map, filter, reduce): Execute on left child output
3. **Binary Operations** (join): Execute on both children's outputs

### Node Execution

Each `LineageNode` has a `run()` method that:

1. Executes the operator's `execute()` method
2. Collates the results into a DataFrame
3. Returns a `NodeOutput` with the result, cost, and metadata

```python
async def run(self, input: pd.DataFrame | list[pd.DataFrame] | None = None) -> NodeOutput:
    if self.op_name == "scan":
        return NodeOutput(output=self.datasource, cost=0.0)
    
    elif self.op_name == "join":
        op_outputs = await self.operator.execute(left_data=input[0], right_data=input[1])
        # Collate join results
        input[0]["keys"] = op_outputs.left_join_keys
        input[1]["keys"] = op_outputs.right_join_keys
        output = input[0].join(input[1], on="keys", how=self.operator.how).drop("keys", axis=1)
        return NodeOutput(output=output, cost=op_outputs.cost)
    
    elif self.op_name == "filter":
        op_outputs = await self.operator.execute(input_data=input)
        if op_outputs.output is None:
            return NodeOutput(output=input, cost=op_outputs.cost)
        return NodeOutput(output=input[op_outputs.output], cost=op_outputs.cost)
    
    # ... other operators
```

## Operators

### BaseOperation

All operators inherit from `BaseOperation` (in `nirvana/ops/base.py`):

```python
class BaseOperation(ABC):
    llm: LLMClient = None  # Shared LLM client
    
    def __init__(
        self,
        op_name: str,
        user_instruction: str = "",
        context: list[dict] | str | None = None,
        model: str | None = None,
        tool: BaseTool | None = None,
        implementation: str | None = "plain",
        rate_limit: int = 16,
        assertions: list[Callable] | None = [],
    ):
        self.op_name = op_name
        self.user_instruction = user_instruction
        self.model = model if model else self.llm.default_model
        self.semaphore = asyncio.Semaphore(rate_limit)
        # ...
    
    @abstractmethod
    def execute(self, *args, **kwargs):
        raise NotImplementedError
```

**Key Features:**

- **Shared LLM Client**: All operations use the same `LLMClient` instance
- **Rate Limiting**: Semaphore controls concurrent LLM calls
- **Implementation Strategies**: Support for different execution strategies
- **Tool Support**: Can use Python functions instead of LLM calls

### Operator Structure

Each operator follows this pattern:

1. **Output Class**: Dataclass defining the output structure
   ```python
   @dataclass
   class MapOpOutputs(BaseOpOutputs):
       field_name: list[str] | None
       output: dict[str, Iterable]
       cost: float
   ```

2. **Operation Class**: Implements the operator logic
   ```python
   class MapOperation(BaseOperation):
       def __init__(self, user_instruction, input_columns, output_columns, **kwargs):
           super().__init__(op_name="map", user_instruction=user_instruction, **kwargs)
           self.input_columns = input_columns
           self.output_columns = output_columns
       
       async def execute(self, input_data: pd.DataFrame, **kwargs) -> MapOpOutputs:
           # Implementation
   ```

3. **Wrapper Function**: Convenience function for direct usage
   ```python
   def map_wrapper(input_data, user_instruction, input_column, output_columns, **kwargs):
       map_op = MapOperation(...)
       return asyncio.run(map_op.execute(input_data=input_data, **kwargs))
   ```

### Implementation Strategies

Operators support different implementation strategies:

- **plain**: Direct LLM call
- **self-refine**: Generate, evaluate, refine if needed
- **fewshot**: In-context learning with examples
- **vote**: Multiple LLM calls with voting (for some operators)

Example from `MapOperation`:

```python
if self.implementation == "plain":
    execution_func = functools.partial(self._execute_by_plain_llm, ...)
elif self.implementation == "self_refine":
    execution_func = functools.partial(self._execute_by_self_refine, ...)
elif self.implementation == "fewshot":
    execution_func = functools.partial(self._execute_by_fewshot_llm, ...)
```

## Query Optimization

### Logical Optimization

Logical optimization applies rule-based transformations to the lineage graph.

**Optimization Rules** (in `nirvana/optim/rules/`):

1. **NonLLMReplace**: Replaces LLM calls with UDFs when possible
2. **MapPullup**: Moves map operations up in the plan
3. **FilterPullup**: Moves filters to earlier positions
4. **FilterPushdown**: Pushes filters down and duplicates over equivalency sets
5. **NonLLMPushdown**: Pushes non-LLM operations down

**Rule Application:**

```python
class LogicalOptimizer:
    def optimize(self, plan: LineageNode):
        plan = NonLLMReplace.transform(plan) if self.non_llm_replace else plan
        plan = MapPullup.transform(plan) if self.map_pullup else plan
        plan = FilterPullup.transform(plan) if self.filter_pullup else plan
        plan = FilterPushdown.transform(plan) if self.filter_pushdown else plan
        plan = NonLLMPushdown.transform(plan) if self.non_llm_pushdown else plan
        return plan
```

Each rule implements a `transform()` method that takes a `LineageNode` and returns a transformed `LineageNode`.

### Physical Optimization

Physical optimization selects the best LLM model for each operator based on cost and quality.

**Process:**

1. **Sample Data**: Use a subset of data for testing
2. **Test Models**: Try different models on the sample
3. **Evaluate**: Compare cost and quality
4. **Select**: Choose model that meets improvement threshold
5. **Execute**: Run on full dataset with selected model

```python
class PhysicalOptimizer:
    async def optimize_exec_model(
        self,
        node: LineageNode,
        input_data: pd.DataFrame,
        num_samples: int,
        improve_margin: float = 0.2
    ):
        # Split data into sample and test set
        sample_data, test_set = self.split_input_data(node.op_name, input_data, num_samples)
        
        # Test default model
        default_output = await node.execute_operation(sample_data)
        default_cost = default_output.cost
        
        # Try alternative models
        best_model = node.operator.model
        for model in self.available_models:
            node.operator.model = model
            test_output = await node.execute_operation(sample_data)
            if test_output.cost < default_cost * (1 - improve_margin):
                best_model = model
                break
        
        # Set best model and execute on full dataset
        node.set_exec_model(best_model)
        return await node.execute_operation(input_data)
```

## LLM Backbone

### LLMClient

The `LLMClient` (in `nirvana/executors/llm_backbone.py`) manages LLM interactions:

```python
class LLMClient:
    default_model: str | None = None
    client = None
    config: LLMArguments = LLMArguments()
    
    @classmethod
    def configure(cls, model_name: str, api_key: str | Path | None = None, **kwargs):
        # Configure provider based on model name
        api_key, base_url = _get_openai_compatible_provider_info(model_name, api_key)
        cls.client = _create_client(api_key=api_key, base_url=base_url, **kwargs)
        cls.default_model = model_name
        return cls()
    
    async def __call__(self, messages: list[dict], parse_tags: bool = False, **kwargs):
        # Make LLM call
        response = await self.client.responses.create(...)
        # Parse output
        # Return dict with output and cost
```

**Features:**

- **Provider Inference**: Automatically detects provider from model name
- **Cost Tracking**: Computes token costs based on model pricing
- **Output Parsing**: Supports XML tag parsing and code extraction
- **Retry Logic**: Handles timeouts with configurable retries

### Cost Computation

Cost is computed based on model pricing:

```python
def _compute_usage(self, response):
    model_name = response.model
    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens
    cached_tokens = response.usage.input_tokens_details.cached_tokens
    
    pricing = MODEL_PRICING[model_name]
    
    if model_name.startswith("qwen"):
        # Qwen: no cache pricing difference
        input_cost = (input_tokens / 1000) * pricing["Input"]
        output_cost = (output_tokens / 1000) * pricing["Output"]
    else:
        # Other providers: separate cache pricing
        input_cost = (input_tokens - cached_tokens) / 1000 * pricing["Input"]
        cache_cost = (cached_tokens / 1000) * pricing["Cache"]
        output_cost = (output_tokens / 1000) * pricing["Output"]
    
    return input_cost + output_cost + cache_cost
```

## Extending Nirvana

### Adding a New Operator

1. **Create Output Class**:
   ```python
   @dataclass
   class MyOpOutputs(BaseOpOutputs):
       result: Any
       cost: float
   ```

2. **Implement Operation Class**:
   ```python
   class MyOperation(BaseOperation):
       def __init__(self, user_instruction, **kwargs):
           super().__init__(op_name="myop", user_instruction=user_instruction, **kwargs)
       
       async def execute(self, input_data: pd.DataFrame, **kwargs) -> MyOpOutputs:
           # Implementation
           return MyOpOutputs(result=..., cost=...)
   ```

3. **Register in op_mapping**:
   ```python
   # In nirvana/lineage/abstractions.py
   op_mapping = {
       # ... existing operators
       "myop": MyOperation,
   }
   ```

4. **Add to DataFrame**:
   ```python
   # In nirvana/dataframe/frame.py
   def semantic_myop(self, user_instruction, input_column, **kwargs):
       op_kwargs = {"user_instruction": user_instruction, ...}
       data_kwargs = {...}
       self.add_operator(op_name="myop", op_kwargs=op_kwargs, data_kwargs=data_kwargs)
   ```

### Adding an Optimization Rule

1. **Create Rule Class**:
   ```python
   class MyRule:
       @staticmethod
       def transform(plan: LineageNode) -> LineageNode:
           # Transform the plan
           return transformed_plan
   ```

2. **Add to LogicalOptimizer**:
   ```python
   # In nirvana/optim/logical.py
   def optimize(self, plan: LineageNode):
       plan = MyRule.transform(plan) if self.my_rule else plan
       # ... other rules
       return plan
   ```

3. **Add Config Option**:
   ```python
   # In nirvana/optim/optimizer.py
   class OptimizeConfig(BaseModel):
       my_rule: bool = Field(default=True, description="Enable my rule")
   ```

## Architecture Overview

```
nirvana/
├── dataframe/          # DataFrame and data types
│   ├── frame.py        # DataFrame class
│   ├── arrays/         # Custom array types (Image, Audio, etc.)
│   └── elements/       # Schema and field definitions
├── ops/                # Semantic operators
│   ├── base.py         # BaseOperation
│   ├── map.py          # Map operator
│   ├── filter.py       # Filter operator
│   ├── join.py         # Join operator
│   ├── reduce.py       # Reduce operator
│   └── prompt_templates/  # LLM prompts for each operator
├── lineage/            # Data lineage
│   ├── abstractions.py # LineageNode, NodeFields
│   └── mixin.py        # LineageMixin for DataFrame
├── optim/              # Query optimization
│   ├── optimizer.py    # PlanOptimizer, OptimizeConfig
│   ├── logical.py      # LogicalOptimizer
│   ├── physical.py     # PhysicalOptimizer
│   └── rules/          # Optimization rules
└── executors/          # LLM execution
    ├── llm_backbone.py # LLMClient
    └── constants.py    # Model pricing, etc.
```

## Best Practices for Developers

1. **Async Operations**: All LLM calls are async - use `asyncio.run()` or `await` appropriately
2. **Cost Tracking**: Always track and return costs in operator outputs
3. **Error Handling**: Handle LLM failures gracefully with fallbacks
4. **Rate Limiting**: Use semaphores to control concurrent LLM calls
5. **Type Hints**: Use type hints for better IDE support and documentation

