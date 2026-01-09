import re
import asyncio
from collections import deque

from nirvana.executors.llm_backbone import LLMClient
from nirvana.executors.tools import FunctionCallTool
from nirvana.lineage.abstractions import LineageNode


def build_code_from_lineage(last_node_in_plan: LineageNode) -> str:
    expressions = []
    datasets_info = []
    def _build_expression(node: LineageNode):
        if node.op_name == "scan":
            datasets_info.append(
                f"- Dataset {len(datasets_info) + 1}: df{len(datasets_info) + 1} with columns {node.node_fields.output_fields}"
            )
            return
        
        _build_expression(node.left_child)
        if node.right_child:
            _build_expression(node.right_child)

        if node.op_name == "filter":
            expression = (
                f"df{len(datasets_info)}.semantic_{node.op_name}(user_instruction=\"{node.operator.user_instruction}\", input_columns={node.operator.input_columns})"
            )
        elif node.op_name in {"rank", "reduce"}:
            expression = (
                f"df{len(datasets_info)}.semantic_{node.op_name}(user_instruction=\"{node.operator.user_instruction}\", input_column=\"{node.operator.input_columns[0]}\")"
            )
        elif node.op_name == "map":
            expression = (
                f"df{len(datasets_info)}.semantic_{node.op_name}(user_instruction=\"{node.operator.user_instruction}\", input_columns={node.operator.input_columns}, output_columns={node.operator.output_columns})"
            )
        elif node.op_name == "join":
            expression = (
                f"df{len(datasets_info)}.semantic_{node.op_name}(other=df{len(datasets_info) + 2}, user_instruction=\"{node.operator.user_instruction}\", left_on=\"{node.operator.left_on[0]}\", right_on=\"{node.operator.right_on[0]}\", how=\"{node.operator.how}\")"
            )
        else:
            raise ValueError(f"Unsupported operation {node.op_name} for code generation.")
        expressions.append(expression)
        return

    _build_expression(last_node_in_plan)
    code = "\n".join(expressions)
    dataset_info = "\n".join(datasets_info)
    return code.strip(), dataset_info


def extract_udfs_from_code(code: str) -> deque:
    udfs = deque()
    expressions = code.split("\n")
    for expr in expressions:
        match = re.search(r'\w+\.semantic_(\w+)\((.*)\)', expr, flags=re.DOTALL)
        if match:
            op_name = match.group(1)
            args_str = match.group(2)
            udf_str = re.search(r'func=([^,]+)', args_str)
            if udf_str:
                udf_code = udf_str.group(1).strip()
                try:
                    udfs.append((udf_code, eval(udf_code)))
                except Exception as e:
                    raise ValueError(f"Failed to evaluate UDF: {udf_code}") from e
            else:
                udfs.append(None)
    return udfs


def replace_with_udf_in_lineage(node: LineageNode, udfs: deque) -> LineageNode:
    def _replace_in_node(node: LineageNode):
        if node.op_name == "scan":
            return
        _replace_in_node(node.left_child)
        if node.right_child:
            _replace_in_node(node.right_child)

        udf = udfs.popleft()
        if udf:
            node.operator.tool = FunctionCallTool.from_function(
                name=udf[0], func=udf[1],
            )
        else:
            return
    
    _replace_in_node(node)
    return node


class NonLLMReplace:
    """
    Replace LLM-driven evaluations on NL instructions with equivalent UDFs.
    """

    rewrite_prompt = """Given a dataset and a user-specified query for semantic data processing.
Each dataset is given in the form of a pandas-like dataframe, and the query is represented by a sequence of .semantic_*(...) operators (* can be replaced with names of supported operators).

The supported operators and their required arguments are as follows.
1. map: Perform an element-wise projection specified by natural language on a given column to a new column. Required arguments:
- user_instruction: a natural language expression
- input_columns: the names of the columns on which the operation is performed
- output_columns: the field names that the operation generates
- func: a lambda function applied to each row of DataFrame (i.e., pd.Series) against input columns, returns a dict with output_column as keys and converted values as values
2. filter: Evaluate a natural language condition per value in a given column (returning boolean). Required arguments:
- user_instruction: the natural language condition
- input_columns: the names of columns on which the operation is performed
- func: a lambda function applied to each row of DataFrame (i.e., pd.Series) against input columns, returns a boolean value per value
3. join: Join a table with another table by keeping all tuple pairs that satisfy a natural language condition. Required arguments:
- other: the other dataset to join with
- user_instruction: the join condition in natural language
- left_on: the name of the column from the left table to join on
- right_on: the name of the column from the right table to join on
- how: the type of join to be performed (e.g., inner, left, right)
- func: a lambda function applied to two rows from left table and right table (i.e., two pd.Series) against left_on and right_on, respectively, returns a boolean value per tuple pair
4. reduce: Aggregate multiple values in a given column into a single result. Required arguments:
- user_instruction: the reducer function in natural language
- input_column: the name of column on which the operation is performed
- func: a lambda function processed by underlying pd.DataFrame.agg() for aggregating the data

Here is an example of a semantic data processing query that contains only map and filter operations:
```python
df.semantic_map(user_instruction="map instruction", input_columns=["col_a"], output_columns=["col_b"])
df.semantic_filter(user_instruction="filter instruction", input_columns=["col_c"])
```

Now, you are given following dataset(s):
{dataset_info}
and a data processing workflow as follows:
```python
{logical_plan}
```

You are tasked with replacing the NL instruction with an equivalent compute function for as many operations as possible. The rewrite aims to generate a plan that is semantically equivalent to the original plan but reduces LLM calls.
There are several constraints to follow.
- The replacement is applied only when `user_instruction` can be converted to a built-in function or a lambda expression.
- If no appropriate replacement applied, keep the original operation.
- Except adding argument `func`, do not change the data processing workflow.
- Any modification to the pre-defined operator interfaces is not allowed.
The rewrite is output as executable python code. Note that **every single operation is placed in a line, i.e., no line break for arguments**. If no rewrites proposed, return an empty code block. ONLY ONE code block can be contained in the output.
"""

    @classmethod
    def transform(cls, node: LineageNode, rewriter: LLMClient) -> tuple[LineageNode, float]:
        code, dataset_info = build_code_from_lineage(node)
        if not code:
            return node, 0.0

        prompt = cls.rewrite_prompt.format(
            dataset_info=dataset_info,
            logical_plan=code
        )
        response = asyncio.run(rewriter(prompt, parse_code=True, lang="python"))
        code, rewrite_cost = response["output"], response["cost"]

        udfs = extract_udfs_from_code(code)
        if udfs:
            new_plan = replace_with_udf_in_lineage(node, udfs)
            return new_plan, rewrite_cost
        else:
            return node, rewrite_cost
