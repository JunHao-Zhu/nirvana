import re
import asyncio
from collections import deque

from nirvana.executors.llm_backbone import LLMClient
from nirvana.executors.tools import FunctionCallTool
from nirvana.lineage.abstractions import LineageNode


def build_code_from_lineage(node: LineageNode) -> str:
    expressions = []
    datasets_info = []
    def _build_expression(node: LineageNode | None):
        if node is None:
            return None
        
        if node.op_name == "scan":
            datasets_info.append(
                f"- Dataset {len(datasets_info) + 1}: df{len(datasets_info) + 1} with columns {node.node_fields.output_fields}"
            )
            return None
        
        expression_from_left_parent = _build_expression(node.left_parent)
        if expression_from_left_parent:
            expressions.append(expression_from_left_parent)
        expression_from_right_parent = _build_expression(node.right_parent)
        if expression_from_right_parent:
            expressions.append(expression_from_right_parent)

        if node.op_name in ["filter", "reduce"]:
            expression = (
                f"df{len(datasets_info) + 1}.semantic_{node.op_name}(user_instruction=\"{node.operator.user_instruction}\", input_column=\"{node.operator.input_columns[0]}\")"
            )
        elif node.op_name == "map":
            expression = (
                f"df{len(datasets_info) + 1}.semantic_{node.op_name}(user_instruction=\"{node.operator.user_instruction}\", input_column=\"{node.operator.input_columns[0]}\", output_column=\"{node.operator.output_columns[0]}\")"
            )
        elif node.op_name == "join":
            expression = (
                f"df{len(datasets_info) + 1}.semantic_{node.op_name}(other=df{len(datasets_info) + 2}, user_instruction=\"{node.operator.user_instruction}\", left_on=\"{node.operator.left_on[0]}\", right_on=\"{node.operator.right_on[0]}\", how=\"{node.operator.how}\")"
            )
        else:
            raise ValueError(f"Unsupported operation {node.op_name} for code generation.")
        return expression

    _build_expression(node)
    code = "\n".join(expressions)
    dataset_info = "\n".join(datasets_info)
    return code, dataset_info


def extract_udfs_from_code(code: str) -> dict[str, callable]:
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
    def _replace_in_node(node: LineageNode | None):
        if node.op_name == "scan" or node is None:
            return
        _replace_in_node(node.left_parent)
        _replace_in_node(node.right_parent)

        udf = udfs.popleft()
        if udf is None:
            return
        else:
            node.operator.tool = FunctionCallTool.from_function(
                name=udf[0],
                func=udf[1],
            )
    
    _replace_in_node(node)
    return node


class NonLLMReplace:
    """
    Replace LLM-driven evaluations on NL instructions with equivalent UDFs.
    """

    rewrite_prompt = """Given a dataset and a user-specified logical plan for a data processing task, you are required to generate equivalent UDFs to replace the NL instructions over non-image/video/audio data.
The target of transformations is to generate a plan that is semantically equivalent to the original plan but reduces LLM calls.
The dataset is given in the form of a pandas dataframe, and a logical plan is represented by a sequence of .semantic_*(...) functions (* can be replaced with names of possible operators).

The supported operators and their required arguments are as follows.
1. map: Perform an element-wise projection specified by natural language on a given column to a new column. Required arguments:
- user_instruction: a natural language expression
- input_column: the name of the column on which the operation is performed
- output_column: the name of the new column that the operation generates
- func: Python function, returns a single value from a single value
2. filter: Evaluate a natural language condition per value in a given column (returning boolean). Required arguments:
- user_instruction: the natural language condition
- input_column: the name of column on which the operation is performed
- func: Python function, returns boolean value by a predicate
3. join: Join a table with another table by keeping all tuple pairs that satisfy a natural language condition. Required arguments:
- other: the other dataset to join with
- user_instruction: the join condition in natural language
- left_on: the name of the column from the left table to join on
- right_on: the name of the column from the right table to join on
- how: the type of join to be performed (e.g., inner, left, right)
- func: function to evaluate the join condition
4. reduce: Aggregate multiple values in a given column into a single result. Required arguments:
- user_instruction: the reducer function in natural language
- input_column: the name of column on which the operation is performed
- func: function to use for aggregating the data

Here is an example of a data processing workflow that contains only map and filter operations:
```python
df.semantic_map(user_instruction="map instruction", input_column="col_a", output_column="col_b")
df.semantic_filter(user_instruction="filter instruction", input_column="col_c")


Now, you are given following dataset(s):
{dataset_info}
and a data processing workflow as follows:
```python
{logical_plan}
```

Replace the NL instruction with an equivalent compute function for as many operations as possible. There are several constraints to follow.
- The replacement is applied only when `user_instruction` can be converted to a built-in function or a lambda expression.
- If no appropriate replacement applied, keep the original operation.
- Except adding argument `func`, do not change the data processing workflow.
- Any modification to the pre-defined operator interfaces is not allowed.
The rewrite is output as executable python code. Note that **each line in the code block represents a single complete function call.** If no rewrites proposed, return an empty code block. ONLY ONE code block can be contained in the output.
"""

    @staticmethod
    def transform(cls, plan: LineageNode, rewriter: LLMClient) -> LineageNode:
        code, dataset_info = build_code_from_lineage(plan)
        if not code:
            return plan

        prompt = cls.rewrite_prompt.format(
            dataset_info=dataset_info,
            logical_plan=code
        )
        response = asyncio.run(rewriter(prompt, parse_code=True, lang="python"))
        code, rewrite_cost = response["output"], response["cost"]

        udfs = extract_udfs_from_code(code)
        if not udfs:
            return plan
        
        new_plan = replace_with_udf_in_lineage(plan, udfs)
        return new_plan
