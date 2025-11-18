import logging
import asyncio
from typing import List
from collections import defaultdict, Counter, deque
import re
import time
import numpy as np
import pandas as pd

from nirvana.executors.llm_backbone import LLMClient
from nirvana.lineage.abstractions import LineageNode
from nirvana.optim.rules import (
    FilterPushdown,
    FilterPullup,
    MapPullup,
    NonLLMPushdown,
    NonLLMReplace,
)

logger = logging.getLogger(__name__)


class PlanCost:
    error_tolerance = 0.8

    def __init__(self, plan: LineageNode, code: str, accuracy: float, cost: float, runtime: float):
        self.plan = plan
        self.code = code
        self.accuracy = accuracy
        self.cost = cost
        self.runtime = runtime

    def __lt__(self, other: "PlanCost") -> bool:
        if self.accuracy / (other.accuracy + 1e-6) >= self.error_tolerance:
            return self.cost < other.cost
        return False
    
    def __le__(self, other: "PlanCost") -> bool:
        if self.accuracy / (other.accuracy + 1e-6) >= self.error_tolerance:
            return self.cost <= other.cost
        return False
    
    def __gt__(self, other: "PlanCost") -> bool:
        if self.accuracy / (other.accuracy + 1e-6) >= self.error_tolerance:
            return self.cost > other.cost
        return True
    
    def __ge__(self, other: "PlanCost") -> bool:
        if self.accuracy / (other.accuracy + 1e-6) >= self.error_tolerance:
            return self.cost >= other.cost
        return True
    
    def __eq__(self, other: "PlanCost") -> bool:
        if self.accuracy / other.accuracy >= self.error_tolerance:
            return self.cost == other.cost
        return False
    

class LogicalOptimizer:
    def __init__(
            self,
            agent: LLMClient = None,
            filter_pullup: bool = True,
            filter_pushdown: bool = True,
            map_pullup: bool = True,
            non_llm_pushdown: bool = True,
            non_llm_replace: bool = True
    ):
        self.agent = agent
        self.filter_pullup = filter_pullup
        self.filter_pushdown = filter_pushdown
        self.map_pullup = map_pullup
        self.non_llm_pushdown = non_llm_pushdown
        self.non_llm_replace = non_llm_replace

    def clear(self):
        pass

    def optimize(self, plan: LineageNode):
        plan = NonLLMReplace.transform(plan) if self.non_llm_replace else plan
        plan = MapPullup.transform(plan) if self.map_pullup else plan
        plan = FilterPullup.transform(plan) if self.filter_pullup else plan
        plan = FilterPushdown.transform(plan) if self.filter_pushdown else plan
        plan = NonLLMPushdown.transform(plan) if self.non_llm_pushdown else plan
        return plan


# class LogicalOptimizer:
#     # TODO: here we only consider one join in the logical plan. Multiple joins will be considered in the future.
#     def __init__(self, max_round: int = 5, agent: LLMClient = None):
#         self.agent = agent
#         self.max_round = max_round
#         self.plan_candidates = []
#         self.num_dataset = 0

#     def clear(self):
#         self.plan_candidates = []

#     def _sample_from_candidates(self, lambda_=0.2):
#         token_costs = np.array([plan.cost for plan in self.plan_candidates])
#         token_costs = (np.max(token_costs) - token_costs) / (np.max(token_costs) - np.min(token_costs) + 1e-6)
#         costs_prob = np.exp(token_costs) / np.exp(token_costs).sum()
#         uniform_prob = np.ones_like(token_costs, dtype=float) / len(token_costs)
#         total_prob = lambda_ * uniform_prob + (1 - lambda_) * costs_prob

#         selected_index = np.random.choice(len(token_costs), p=total_prob)
#         return self.plan_candidates[selected_index]

#     def _build_code_from_plan(self, last_node_in_plan: LineageNode):
#         logical_plan = []
#         plan_stats = []
#         def _build_op(node: LineageNode):
#             if node.op_name == "scan":
#                 self.num_dataset += 1
#                 return None

#             code_from_left_child = _build_op(node.left_child)
#             if code_from_left_child:
#                 logical_plan.append(code_from_left_child)
#             if node.right_child:
#                 code_from_right_child = _build_op(node.right_child)
#                 if code_from_right_child:
#                     logical_plan.append(code_from_right_child)
            
#             plan_stats.append(collect_op_metadata(node, print_info=False))
#             op_kwargs = node.op_metadata
#             if node.op_name == "map":
#                 code = f"df{self.num_dataset}.semantic_map(user_instruction=\"{op_kwargs["user_instruction"]}\", input_column=\"{op_kwargs["input_column"]}\", output_column=\"{op_kwargs["output_column"]}\")"
#             elif node.op_name == "filter" or node.op_name == "reduce":
#                 code = f"df{self.num_dataset}.semantic_{node.op_name}(user_instruction=\"{op_kwargs["user_instruction"]}\", input_column=\"{op_kwargs["input_column"]}\")"
#             elif node.op_name == "join":
#                 code = f"df{self.num_dataset - 1}.semantic_join(other=df{self.num_dataset}, user_instruction=\"{op_kwargs['user_instruction']}\", left_on=\"{op_kwargs['left_on']}\", right_on=\"{op_kwargs['right_on']}\", how=\"{op_kwargs['how']}\")"
#             return code
        
#         _build_op(last_node_in_plan)
#         code = "\n".join(logical_plan)
#         return code, plan_stats
    
#     def _extract_op_name_and_args(self, code: str):
#         match = re.search(r'(\w+)\.semantic_(\w+)\((.*)\)', code, flags=re.DOTALL)
#         if match:
#             data_name = match.group(1)
#             op_name = match.group(2)
#             args = match.group(3)
#             if op_name == "map" or op_name == "filter" or op_name == "reduce":
#                 user_instruction = re.search(r'user_instruction="([^"]*)"', args)
#                 input_column = re.search(r'input_column="([^"]*)"', args)
#                 func = re.search(r'func=([^,]+)', args)
#                 output_column = re.search(r'output_column="([^"]*)"', args)
#             else:
#                 user_instruction = re.search(r'user_instruction="([^"]*)"', args)
#                 other = re.search(r'other=([^,]+)', args)
#                 left_on = re.search(r'left_on="([^"]*)"', args)
#                 right_on = re.search(r'right_on="([^"]*)"', args)
#                 how = re.search(r'how="([^"]*)"', args)
#                 func = re.search(r'func=([^,]+)', args)
#             try:
#                 return {
#                     "data_name": data_name,
#                     "op_name": op_name,
#                     "user_instruction": user_instruction.group(1) if user_instruction else None,
#                     "input_column": input_column.group(1) if input_column else None,
#                     "func": eval(func.group(1)) if func else None,
#                     "output_column": output_column.group(1) if output_column else None,
#                 }
#             except Exception as e:
#                 logger.debug(f"the `func` cannot be used: {e}")
            
#             return {
#                 "data_name": data_name,
#                 "op_name": op_name,
#                 "user_instruction": user_instruction.group(1) if user_instruction else None,
#                 "input_column": input_column.group(1) if input_column else None,
#                 "func": None,
#                 "output_column": output_column.group(1) if output_column else None,
#             }
#         return None
    
#     def _build_plan_from_code(self, code: str, valid_sets: deque, **kwargs):
#         sub_lineages = deque()
#         plan_stats = []
#         operations = code.split("\n")
#         data_name = set()
#         for operation in operations:
#             # define the regex pattern and match the operation string
#             op_kwargs = self._extract_op_name_and_args(operation)
#             if op_kwargs is None:
#                 continue
#             op_node = LineageNode(op_name=op_kwargs["op_name"], op_metadata=op_kwargs)
#             plan_stats.append(collect_op_metadata(op_node, print_info=False))
            
#             if op_kwargs["op_name"] == "join":
#                 if op_kwargs["data_name"] not in data_name and op_kwargs["other"] not in data_name:
#                     data_name.update([op_kwargs["data_name"], op_kwargs["other"]])
#                     left_data = valid_sets.popleft()
#                     right_data = valid_sets.popleft()
#                     left_node = LineageNode(op_name="scan", data_metadata={"columns": left_data.columns}, datasource=left_data)
#                     right_node = LineageNode(op_name="scan", data_metadata={"columns": right_data.columns}, datasource=right_data)
#                     data_metadata = {
#                         "input_left_fields": left_data.columns,
#                         "input_right_fields": right_data.columns,
#                         "output_fields": list(set(left_data.columns).union(set(right_data.columns))),
#                     }
#                     node = LineageNode(op_name="join", op_metadata=op_kwargs, data_metadata=data_metadata)
#                     node.set_left_child(left_node)
#                     node.set_right_child(right_node)
#                 elif op_kwargs["other"] not in data_name:
#                     data_name.add(op_kwargs["other"])
#                     left_node = sub_lineages.popleft()
#                     right_node = valid_sets.popleft()
#                     data_metadata = {
#                         "input_left_fields": left_node.data_metadata["columns"],
#                         "input_right_fields": right_data.columns,
#                         "output_fields": list(set(left_node.data_metadata["columns"]).union(set(right_data.columns))),
#                     }
#                     node = LineageNode(op_name="join", op_metadata=op_kwargs, data_metadata=data_metadata)
#                     node.set_left_child(left_node)
#                     node.set_right_child(right_node)
#                 else:
#                     left_node = sub_lineages.popleft()
#                     right_node = sub_lineages.popleft()
#                     data_metadata = {
#                         "input_left_fields": left_node.data_metadata["columns"],
#                         "input_right_fields": right_node.data_metadata["columns"],
#                         "output_fields": list(set(left_node.data_metadata["columns"]).union(set(right_node.data_metadata["columns"]))),
#                     }
#                     node = LineageNode(op_name="join", op_metadata=op_kwargs, data_metadata=data_metadata)
#                     node.set_left_child(left_node)
#                     node.set_right_child(right_node)
#                 sub_lineages.append(node)
#             elif op_kwargs["op_name"] == "map" or op_kwargs["op_name"] == "filter":
#                 if op_kwargs["data_name"] not in data_name:
#                     data_name.add(op_kwargs["data_name"])
#                     dataset = valid_sets.popleft()
#                     data = LineageNode(op_name="scan", data_metadata={"columns": dataset.columns}, datasource=dataset)
#                     data_metadata = {
#                         "input_fields": dataset.columns,
#                         "output_fields": dataset.columns + [op_kwargs["output_column"]] if op_kwargs["output_column"] else dataset.columns,
#                     }
#                     op_node = LineageNode(op_name=op_kwargs["op_name"], op_metadata=op_kwargs, data_metadata=data_metadata)
#                     op_node.set_left_child(data)
#                 else:
#                     last_node = sub_lineages.popleft()
#                     data_metadata = {
#                         "input_fields": last_node.data_metadata["columns"],
#                         "output_fields": last_node.data_metadata["columns"] + [op_kwargs["output_column"]] if op_kwargs["output_column"] else last_node.data_metadata["columns"],
#                     }
#                     op_node = LineageNode(op_name=op_kwargs["op_name"], op_metadata=op_kwargs, data_metadata=data_metadata)
#                     op_node.set_left_child(last_node)
#                 sub_lineages.append(op_node)
#             elif op_kwargs["op_name"] == "reduce":
#                 if op_kwargs["data_name"] not in data_name:
#                     data_name.add(op_kwargs["data_name"])
#                     dataset = valid_sets.popleft()
#                     data = LineageNode(op_name="scan", data_metadata={"columns": dataset.columns}, datasource=dataset)
#                     data_metadata = {
#                         "input_fields": dataset.columns,
#                         "output_fields": None
#                     }
#                     op_node = LineageNode(op_name=op_kwargs["op_name"], op_metadata=op_kwargs, data_metadata=data_metadata)
#                     op_node.set_left_child(data)
#                 else:
#                     last_node = sub_lineages.popleft()
#                     data_metadata = {
#                         "input_fields": last_node.data_metadata["columns"],
#                         "output_fields": None
#                     }
#                     op_node = LineageNode(op_name=op_kwargs["op_name"], op_metadata=op_kwargs, data_metadata=data_metadata)
#                     op_node.set_left_child(last_node)
#                 sub_lineages.append(op_node)
#         assert len(sub_lineages) == 1
#         return sub_lineages.pop(), plan_stats
    
#     def _naive_estimate_plan_cost(self, init_plan_stats: list, new_plan_stats: list = None):
#         accuracy_score, token_cost, selectivity = 1.0, 0.0, 1.0
#         # only estimate the cost of initial plan when new plan is None
#         if new_plan_stats is None:
#             for op_name, user_instr, input_col, output_col, has_func in init_plan_stats:
#                 if op_name == "map":
#                     token_cost += selectivity * len(user_instr)
#                 elif op_name == "filter":
#                     token_cost += selectivity * len(user_instr)
#                     selectivity *= 0.5
#                 elif op_name == "reduce":
#                     token_cost += len(user_instr) # TODO: need to consider the size of reducer input
#             return accuracy_score, token_cost, 0.0
        
#         ops_on_columns_init, ops_on_columns_new = defaultdict(Counter), defaultdict(Counter)
#         for op_info in init_plan_stats:
#             ops_on_columns_init[op_info[2]].update([op_info[0]])
#         for op_info in new_plan_stats:
#             ops_on_columns_new[op_info[2]].update([op_info[0]])

#         if ops_on_columns_new.keys() != ops_on_columns_init.keys():
#             return 0.0, 0.0
#         for op_name, user_instr, input_col, output_col, has_func in new_plan_stats:
#             if op_name == "map":
#                 token_cost += 0.0 if has_func else selectivity * len(user_instr)
#             elif op_name == "filter":
#                 num_diff = ops_on_columns_init[input_col]["filter"] - ops_on_columns_new[input_col]["filter"] + 1
#                 token_cost += 0.0 if has_func else selectivity * len(user_instr)
#                 selectivity *= 0.5 ** num_diff
#             elif op_name == "reduce":
#                 token_cost += 0.0 if has_func else len(user_instr)
#         return accuracy_score, token_cost, 0.0
    
#     def _estimate_plan_cost(self, plan: LineageNode, ground_truth: pd.DataFrame = None):
#         results, token_cost, exec_time = execute_along_lineage(plan)
#         if ground_truth is None:
#             accuracy_score = 1.0
#         else:
#             accuracy_score = Evaluator.evaluate(ground_truth, results, self.agent)
#         return results, accuracy_score, token_cost, exec_time

#     def optimize(self, initial_plan: LineageNode, valid_set: List[pd.DataFrame]):
#         round = 0
#         optimize_cost = 0.0
#         optimize_start_time = time.time()
#         # 0. prepare the valid set
#         columns = []
#         for dataset in valid_set:
#             columns.extend(dataset.columns.to_list())
#         valid_set = deque(valid_set)
#         # 1. get the data processing ground truth by executing the initial plan on the validation set
#         init_code, init_plan_stats = self._build_code_from_plan(initial_plan)
#         # Bug: need to load valid sets in initial plan
#         groundtruth, accuracy_score, token_cost, exec_time = self._estimate_plan_cost(initial_plan)
#         init_plan_cost = PlanCost(initial_plan, init_code, accuracy_score, token_cost, exec_time)
#         self.plan_candidates.append(init_plan_cost)

#         # 2. optimize the plan
#         while round < self.max_round:
#             # 2.1 sample a plan from the candidate list
#             cand_plan_cost = self._sample_from_candidates()

#             # 2.2 optimize the plan
#             optimize_prompt = [{
#                 "role": "user",
#                 "content": PLAN_OPIMIZE_PROMPT.format(columns=columns, logical_plan=cand_plan_cost.code)
#             }]
#             agent_output = asyncio.run(self.agent(messages=optimize_prompt, parse_code=True, lang="python"))
#             optimized_code, cost_per_optim = agent_output["output"], agent_output["cost"]
#             optimize_cost += cost_per_optim
#             if optimized_code == "":
#                 round += 1
#                 continue
#             optimized_plan, plan_stats = self._build_plan_from_code(optimized_code, columns)
#             if optimized_plan is None:
#                 round += 1
#                 continue

#             # 2.3. compare the results with the ground truth
#             _, accuracy_score, token_cost, exec_time = self._estimate_plan_cost(optimized_plan, groundtruth)
#             plan_cost = PlanCost(optimized_plan, optimized_code, accuracy_score, token_cost, exec_time)
#             self.plan_candidates.append(plan_cost)
#             round += 1

#         # 3. select the best plan from the candidate list
#         best_plan = sorted(self.plan_candidates)[0]
#         optimize_end_time = time.time()
#         optimize_time = optimize_end_time - optimize_start_time
#         logger.info(f"Plan optimization is finished, taking {optimize_time:.4f} seconds and ${optimize_cost:.4f}. Here are some statistics:")
#         logger.info(f"initial plan cost: {init_plan_cost.cost} -> optimized plan cost: {best_plan.cost}")
#         logger.info(f"initial plan runtime: {init_plan_cost.runtime} -> optimized plan runtime: {best_plan.runtime}")
#         logger.info(f"initial plan accuracy: {init_plan_cost.accuracy} -> optimized plan accuracy: {best_plan.accuracy}")
#         return best_plan.plan
