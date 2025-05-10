PLAN_OPIMIZE_PROMPT = """Given a dataset and a user-specified logical plan for a data processing task, you are required to optimize the plan by applying a set of equivalent transformations.
The transformations should be semantically equivalent to the original plan but can improve efficiency and reduce runtime costs/LLM calls.
The dataset is given in the form of a pandas dataframe, and a logical plan is represented by a sequence of .semantic_*(...) functions (* can be replaced with names of possible operators).
Here is an example that defines a logical plan containing map and filter operations:
```python
# df is the name of the dataset
df.semantic_map(user_instruction="map instruction", input_column="col_a", output_column="col_b")
df.semantic_filter(user_instruction="filter instruction", input_column="col_c")
```

The supported operators and their required arguments are as follows.
1. map: Perform an element-wise projection specified by natural language on a given column to a new column. Required arguments:
- user_instruction: a natural language expression
- func: Python function, returns a single value from a single value
- input_column: the name of the column on which the operation is performed
- output_column: the name of the new column that the operation generates
2. filter: Evaluate a natural language condition per value in a given column (returning boolean). Required arguments:
- user_instruction: the natural language condition
- func: Python function, returns boolean value by a predicate
- input_column: the name of column on which the operation is performed
3. reduce: Aggregate multiple values in a given column into a single result. Required arguments:
- user_instruction: the reducer function in natural language
- func: function to use for aggregating the data
- input_column: the name of column on which the operation is performed

The available transformation rules are as follows.
- Filter pushdown: Move a filter operator that does not rely on results of prior operator to occur before the prior operator.
- Operator fusion: Fuse multiple operators that are in the same category and work on the same column into one operator. To keep semantically equivalent, you are required to rewrite the instruction for the new operator.
- Non-LLM operator replacement: Replace the natural language instruction with an equivalent compute function only when `user_instruction` can be converted to a straightforward `func`, like `func` is a simple lambda or built-in function. Note that the `user_instruction` should be preserved in the new operator.

Now, you are given a dataset with columns: {columns}, and a logical plan:
```python
{logical_plan}
```

Optimize the logical plan by applying as many transformation rules as possible but transformations rigidly conform to the rules. Note that **any modification to the pre-defined operator interfaces is not allowed**, and **the argument input_column for the current operation can only be selected from the dataset's columns or output_column of previous operations.**
The optimized plan is output as executable python code. Note that **each line in the code block represents a single complete function call.** If no further optimization proposed, return an empty code block. ONLY ONE code block can be contained in the output.
"""


RESULT_EVALUATE_PROMPT = """Here are a golden analysis result obtained by a golden data processing plan and a result derived from an alternative data processing plan.
Evaluate the two data analysis results and return a rating between 0 and 10, where 0 means the two results are completely different and 10 means they are exactly the same.

Ground truth:
{ground_truth}

Result from the alternative plan:
{result}

You should carefully consider all values (and their semantics) in both analysis results. The rating score is enclosed within <score></score> tags, i.e., <score>Rating Score</score>.
"""
