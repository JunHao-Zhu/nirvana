# Data Lineage

Data lineage is the core abstraction in Nirvana that enables lazy execution, query optimization, and cost tracking.

## Overview

Data lineage is represented as a **directed acyclic graph (DAG)** where:
- **Nodes** represent operators (scan, map, filter, join, reduce, rank)
- **Edges** represent data flow between operators
- Each node tracks input/output fields and dependencies

## LineageNode

The `LineageNode` class (in `nirvana/lineage/abstractions.py`) is the fundamental building block. It supports asynchronous execution:

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

    async def run(self, input: pd.DataFrame | list[pd.DataFrame] | None = None) -> NodeOutput:
        # Executes the operator asynchronously and returns NodeOutput
        ...
```

**Key Components:**

1.  **Operator**: The actual operation instance (MapOperation, FilterOperation, etc.)
2.  **NodeFields**: Tracks input and output fields:
    ```python
    @dataclass
    class NodeFields:
        left_input_fields: list[str]
        right_input_fields: list[str]
        output_fields: list[str]
    ```
3.  **Child Nodes**: Left and right children for binary operations (e.g., join).

## Building Lineage

When you call semantic operations on a DataFrame, nodes are added to the lineage. `LineageMixin` manages this process.

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

Execution follows a **post-order traversal** of the lineage graph, and it is asynchronous:

```python
async def execute_node(node: LineageNode) -> tuple[pd.DataFrame, float]:
    if node.left_child:
        left_node_output, cost_from_left_subtree = await execute_node(node.left_child)
    if node.right_child:
        right_node_output, cost_from_right_subtree = await execute_node(node.right_child)
    
    if node.op_name == "scan":
        node_output = await node.run()
        # ...
    elif node.op_name == "join":
        node_output = await node.run([left_node_output, right_node_output])
        # ...
    else:
        node_output = await node.run(left_node_output)
        # ...
        
    return node_output.output, accumulated_cost
```

**Execution Flow:**

1.  **Scan**: Returns the datasource DataFrame.
2.  **Unary Operations** (map, filter, reduce, rank): Execute on left child output.
3.  **Binary Operations** (join): Execute on both children's outputs.

## Node Execution

Each `LineageNode` has a `run()` method that:

1.  Executes the operator's `execute()` method asynchronously.
2.  Collates the results into a DataFrame.
3.  Returns a `NodeOutput` with the result, cost, and metadata.

```python
async def run(self, input: pd.DataFrame | list[pd.DataFrame] | None = None) -> NodeOutput:
    if self.op_name == "scan":
        op_outputs = await self.operator.execute(input_data=self.datasource)
        return NodeOutput(output=op_outputs.output, cost=op_outputs.cost)
    
    elif self.op_name == "join":
        op_outputs = await self.operator.execute(left_data=input[0], right_data=input[1])
        # Handle join logic
        return NodeOutput(output=output, cost=op_outputs.cost)
    
    # ... other operators
```
