import logging
from typing import List
from collections import defaultdict, Counter
import re
import time
import numpy as np

from mahjong.models.llm_backbone import LLMClient
from mahjong.lineage.abstractions import LineageNode, LineageOpNode, LineageDataNode
from mahjong.lineage.utils import collect_op_metadata
from mahjong.optim.optimize_prompt import PLAN_OPIMIZE_PROMPT

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
    def __init__(self, max_round: int = 5, agent: LLMClient = None):
        self.agent = agent
        self.max_round = max_round
        self.candidate_logical_plan = []

    def clear(self):
        self.candidate_logical_plan = []

    def _sample_from_candidates(self, lambda_=0.2):
        token_costs = np.array([plan.cost for plan in self.candidate_logical_plan])
        token_costs = (np.max(token_costs) - token_costs) / (np.max(token_costs) - np.min(token_costs) + 1e-6)
        costs_prob = np.exp(token_costs) / np.exp(token_costs).sum()
        uniform_prob = np.ones_like(token_costs, dtype=float) / len(token_costs)
        total_prob = lambda_ * uniform_prob + (1 - lambda_) * costs_prob

        selected_index = np.random.choice(len(token_costs), p=total_prob)
        return self.candidate_logical_plan[selected_index]

    def _build_code_from_plan(self, last_node_in_plan: LineageNode, input_dataset_name: str):
        logical_plan = []
        plan_stats = []
        
        def _build_op(node: LineageNode):
            if len(node.parent) == 0:
                node: LineageOpNode = node
                plan_stats.append(collect_op_metadata(node, print_info=False))
                if node.output_column:
                    code = f"{input_dataset_name}.semantic_{node.op_name}(user_instruction=\"{node.user_instruction}\", input_column=\"{node.input_column}\", output_column=\"{node.output_column}\")"
                else:
                    code = f"{input_dataset_name}.semantic_{node.op_name}(user_instruction=\"{node.user_instruction}\", input_column=\"{node.input_column}\")"
                return code

            code_for_parents = ""
            for parent_node in node.parent:
                code_for_parents += _build_op(parent_node)
            if code_for_parents:
                logical_plan.append(code_for_parents)

            if isinstance(node, LineageDataNode):
                return ""
            
            plan_stats.append(collect_op_metadata(node, print_info=False))
            if node.output_column:
                code = f"{input_dataset_name}.semantic_{node.op_name}(user_instruction=\"{node.user_instruction}\", input_column=\"{node.input_column}\", output_column=\"{node.output_column}\")"
            else:
                code = f"{input_dataset_name}.semantic_{node.op_name}(user_instruction=\"{node.user_instruction}\", input_column=\"{node.input_column}\")"
            return code
        
        _build_op(last_node_in_plan)
        code = "\n".join(logical_plan)
        return code, plan_stats
    
    def _extract_op_name_and_args(self, code: str):
        match = re.search(r'\w+\.semantic_(\w+)\((.*)\)', code, flags=re.DOTALL)
        if match:
            op_name = match.group(1)
            args = match.group(2)
            user_instruction = re.search(r'user_instruction="([^"]*)"', args)
            input_column = re.search(r'input_column="([^"]*)"', args)
            func = re.search(r'func=([^,]+)', args)
            output_column = re.search(r'output_column="([^"]*)"', args)
            try:
                return {
                    "op_name": op_name,
                    "user_instruction": user_instruction.group(1) if user_instruction else None,
                    "input_column": input_column.group(1) if input_column else None,
                    "func": eval(func.group(1)) if func else None,
                    "output_column": output_column.group(1) if output_column else None,
                }
            except Exception as e:
                logger.debug(f"the `func` cannot be used: {e}")
            
            return {
                "op_name": op_name,
                "user_instruction": user_instruction.group(1) if user_instruction else None,
                "input_column": input_column.group(1) if input_column else None,
                "func": None,
                "output_column": output_column.group(1) if output_column else None,
            }
        return None
    
    def _build_plan_from_code(self, code: str, columns: List[str]):
        last_node_in_plan = None
        plan_stats = []
        operations = code.split("\n")
        for operation in operations:
            # define the regex pattern and match the operation string
            op_kwargs = self._extract_op_name_and_args(operation)
            if op_kwargs is None:
                continue
            op_node = LineageOpNode(**op_kwargs)
            plan_stats.append(collect_op_metadata(op_node, print_info=False))
            
            if last_node_in_plan is None:
                data_node = LineageDataNode(columns=columns, new_field=op_kwargs["output_column"])
                op_node.add_child(data_node)
                data_node.add_parent(op_node)
                last_node_in_plan = data_node
            else:
                columns_from_last_node = (
                    last_node_in_plan.columns 
                    if last_node_in_plan.new_field is None else 
                    last_node_in_plan.columns + [last_node_in_plan.new_field]
                )
                if op_kwargs["input_column"] not in columns_from_last_node:
                    return None, None
                data_node = LineageDataNode(columns=columns_from_last_node, new_field=op_kwargs["output_column"])
                op_node.add_child(data_node)
                data_node.add_parent(op_node)

                op_node.add_parent(last_node_in_plan)
                last_node_in_plan.add_child(op_node)
                last_node_in_plan = data_node
        # if last_node_in_plan is None:
        #     raise RuntimeError("The operation string does not match the expected format.")
        return last_node_in_plan, plan_stats
    
    def _naive_estimate_plan_cost(self, init_plan_stats: list, new_plan_stats: list = None):
        accuracy_score, token_cost, selectivity = 1.0, 0.0, 1.0
        # only estimate the cost of initial plan when new plan is None
        if new_plan_stats is None:
            for op_name, user_instr, input_col, output_col, has_func in init_plan_stats:
                if op_name == "map":
                    token_cost += selectivity * len(user_instr)
                elif op_name == "filter":
                    token_cost += selectivity * len(user_instr)
                    selectivity *= 0.5
                elif op_name == "reduce":
                    token_cost += len(user_instr) # TODO: need to consider the size of reducer input
            return accuracy_score, token_cost
        
        ops_on_columns_init, ops_on_columns_new = defaultdict(Counter), defaultdict(Counter)
        for op_info in init_plan_stats:
            ops_on_columns_init[op_info[2]].update([op_info[0]])
        for op_info in new_plan_stats:
            ops_on_columns_new[op_info[2]].update([op_info[0]])

        if ops_on_columns_new.keys() != ops_on_columns_init.keys():
            return 0.0, 0.0
        for op_name, user_instr, input_col, output_col, has_func in new_plan_stats:
            if op_name == "map":
                token_cost += 0.0 if has_func else selectivity * len(user_instr)
            elif op_name == "filter":
                num_diff = ops_on_columns_init[input_col]["filter"] - ops_on_columns_new[input_col]["filter"] + 1
                token_cost += 0.0 if has_func else selectivity * len(user_instr)
                selectivity *= 0.5 ** num_diff
            elif op_name == "reduce":
                token_cost += 0.0 if has_func else len(user_instr)
        return accuracy_score, token_cost

    def optimize(self, initial_plan: LineageNode, input_dataset_name: str, columns: List[str]):
        round = 0
        optimize_cost = 0.0
        optimize_start_time = time.time()
        # 1. get the data processing ground truth by executing the initial plan on the validation set
        init_code, init_plan_stats = self._build_code_from_plan(initial_plan, input_dataset_name=input_dataset_name)
        accuracy_score, token_cost = self._naive_estimate_plan_cost(init_plan_stats)
        init_plan_cost = PlanCost(initial_plan, init_code, accuracy_score, token_cost, 0.0)
        self.candidate_logical_plan.append(init_plan_cost)

        # 2. optimize the plan
        while round < self.max_round:
            # 2.1 sample a plan from the candidate list
            cand_plan_cost = self._sample_from_candidates()

            # 2.2 optimize the plan
            optimize_prompt = [{
                "role": "user",
                "content": PLAN_OPIMIZE_PROMPT.format(columns=columns, logical_plan=cand_plan_cost.code)
            }]
            agent_output = self.agent(messages=optimize_prompt, parse_code=True, lang="python")
            optimized_code, cost_per_optim = agent_output["output"], agent_output["cost"]
            optimize_cost += cost_per_optim
            if optimized_code == "":
                round += 1
                continue
            optimized_plan, plan_stats = self._build_plan_from_code(optimized_code, columns)
            if optimized_plan is None:
                round += 1
                continue

            # 2.3. compare the results with the ground truth
            accuracy_score, token_cost = self._naive_estimate_plan_cost(init_plan_stats, plan_stats)
            # accuracy_score = Evaluator.evaluate(ground_truth_on_valid_set, results_on_valid_set, self.agent)
            plan_cost = PlanCost(optimized_plan, optimized_code, accuracy_score, token_cost, 0.0)
            self.candidate_logical_plan.append(plan_cost)
            round += 1

        # 3. select the best plan from the candidate list
        best_plan = sorted(self.candidate_logical_plan)[0]
        optimize_end_time = time.time()
        optimize_time = optimize_end_time - optimize_start_time
        logger.info(f"Plan optimization is finished, taking {optimize_time:.4f} seconds and ${optimize_cost:.4f}. Here are some statistics:")
        logger.info(f"initial plan cost: {init_plan_cost.cost} -> optimized plan cost: {best_plan.cost}")
        logger.info(f"initial plan runtime: {init_plan_cost.runtime} -> optimized plan runtime: {best_plan.runtime}")
        logger.info(f"initial plan accuracy: {init_plan_cost.accuracy} -> optimized plan accuracy: {best_plan.accuracy}")
        return best_plan.plan
