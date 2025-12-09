# Data Lineage

Data lineage is the core abstraction in Nirvana that enables lazy execution, query optimization, and cost tracking.

## Overview

Data lineage is represented as a **directed acyclic graph (DAG)** where:
- **Nodes** represent operators (scan, map, filter, join, reduce, rank)
- **Edges** represent data flow between operators
- Each node tracks input/output fields and dependencies

## LineageNode

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

## Building Lineage

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

## Execution Model

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

## Node Execution

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
