import logging
from typing import List
import re
import time
import numpy as np

from mahjong.models.llm_backbone import LLMClient
from mahjong.lineage.abstractions import LineageNode, LineageOpNode, LineageDataNode
from mahjong.optimization.optimize_prompt import PLAN_OPIMIZE_PROMPT

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
        if self.accuracy / other.accuracy >= self.error_tolerance:
            return self.cost < other.cost
        return False
    
    def __le__(self, other: "PlanCost") -> bool:
        if self.accuracy / other.accuracy >= self.error_tolerance:
            return self.cost <= other.cost
        return False
    
    def __gt__(self, other: "PlanCost") -> bool:
        if self.accuracy / other.accuracy >= self.error_tolerance:
            return self.cost > other.cost
        return True
    
    def __ge__(self, other: "PlanCost") -> bool:
        if self.accuracy / other.accuracy >= self.error_tolerance:
            return self.cost >= other.cost
        return True
    
    def __eq__(self, other: "PlanCost") -> bool:
        if self.accuracy / other.accuracy >= self.error_tolerance:
            return self.cost == other.cost
        return False


class Optimizer:
    agent: LLMClient = None

    def __init__(self, valid_set, max_round: int = 5):
        self.valid_set = valid_set
        self.max_round = max_round
        self.candidate_logical_plan = []

    @classmethod
    def set_agent(cls, client: LLMClient):
        cls.agent = client

    def _sample_from_candidates(self, lambda_=0.4):
        token_costs = np.array([plan.cost for plan in self.candidate_logical_plan])
        token_costs = (np.max(token_costs) - token_costs) / (np.max(token_costs) - np.min(token_costs) + 1e-6)
        costs_prob = np.exp(token_costs) / np.exp(token_costs).sum()
        uniform_prob = np.ones_like(token_costs, dtype=float) / len(token_costs)
        total_prob = lambda_ * uniform_prob + (1 - lambda_) * costs_prob

        selected_index = np.random.choice(len(token_costs), p=total_prob)
        return self.candidate_logical_plan[selected_index]

    def _build_code_from_plan(self, last_node_in_plan: LineageNode, input_dataset_name: str):
        logical_plan = []
        plan_statistics = []
        def _collect_op_statistics(node: LineageOpNode):
            op_name = node.op_name
            user_instruction = node.user_instruction
            input_column = node.input_column
            plan_statistics.append((op_name, user_instruction, input_column))
            
        def _build_op(node: LineageNode):
            if len(node.parent) == 0:
                node: LineageOpNode = node
                _collect_op_statistics(node)
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
            
            _collect_op_statistics(node)
            if node.output_column:
                code = f"{input_dataset_name}.semantic_{node.op_name}(user_instruction=\"{node.user_instruction}\", input_column=\"{node.input_column}\", output_column=\"{node.output_column}\")"
            else:
                code = f"{input_dataset_name}.semantic_{node.op_name}(user_instruction=\"{node.user_instruction}\", input_column=\"{node.input_column}\")"
            return code
        
        _build_op(last_node_in_plan)
        code = "\n".join(logical_plan)
        return code, plan_statistics
    
    def _build_plan_from_code(self, code: str, columns: List[str]) -> LineageDataNode:
        last_node_in_plan = None
        operations = code.split("\n")
        for operation in operations:
            # define the regex pattern and match the operation string
            pattern = (
                r"\w+\.semantic_(?P<op_name>\w+)\(user_instruction=\"(?P<user_instruction>[^\"]+)\", input_column=\"(?P<input_column>[^\"]+)\"(?:, output_column=\"(?P<output_column>[^\"]+)\")?\)"
            )
            match = re.match(pattern, operation)
            if match is None:
                continue

            op_name = match.group("op_name")
            user_instruction = match.group("user_instruction")
            input_column = match.group("input_column")
            output_column = match.group("output_column")

            op_node = LineageOpNode(op_name, user_instruction, input_column, output_column)
            
            if last_node_in_plan is None:
                data_node = LineageDataNode(columns=columns, new_field=output_column)
                op_node.add_child(data_node)
                data_node.add_parent(op_node)
                last_node_in_plan = data_node
            else:
                columns_from_last_node = (
                    last_node_in_plan.columns 
                    if last_node_in_plan.new_field is None else 
                    last_node_in_plan.columns + [last_node_in_plan.new_field]
                )
                data_node = LineageDataNode(columns=columns_from_last_node, new_field=output_column)
                op_node.add_child(data_node)
                data_node.add_parent(op_node)

                op_node.add_parent(last_node_in_plan)
                last_node_in_plan.add_child(op_node)
                last_node_in_plan = data_node
        if last_node_in_plan is None:
            raise RuntimeError("The operation string does not match the expected format.")
        return last_node_in_plan
    
    def _naive_estimate_plan_cost(self, init_plan_stats: list, new_plan_stats: list = None):
        accuracy_score, token_cost, selectivity = 1.0, 0.0, 1.0
        # only estimate the cost of initial plan when new plan is None
        if new_plan_stats is None:
            for op_name, user_instruction, input_column in init_plan_stats:
                if op_name == "map":
                    token_cost += selectivity * len(user_instruction)
                elif op_name == "filter":
                    token_cost += selectivity * len(user_instruction)
                    selectivity *= 0.5
                elif op_name == "reduce":
                    token_cost += len(user_instruction) # TODO: need to consider the size of reducer input
            return accuracy_score, token_cost
        
        columns_in_init_plan = set([op[2] for op in init_plan_stats])
        columns_in_new_plan = set([op[2] for op in new_plan_stats])
        if columns_in_init_plan != columns_in_new_plan:
            return 0.0, 0.0
        for op_name, user_instruction, input_column in new_plan_stats:
            if op_name == "map":
                token_cost += selectivity * len(user_instruction)
            elif op_name == "filter":
                replaced_filters = [op for op in init_plan_stats if op[0] == "filter" and op[2] == input_column]
                token_cost += selectivity * len(user_instruction)
                selectivity *= 0.5 ** len(replaced_filters)
            elif op_name == "reduce":
                token_cost += len(user_instruction)
        return accuracy_score, token_cost

    def optimize(self, initial_plan: LineageNode, input_dataset_name: str, columns: List[str]):
        round = 0
        optimize_start_time = time.time()
        # 1. get the data processing ground truth by executing the initial plan on the validation set
        # ground_truth_on_valid_set, token_cost, runtime = execute_plan(initial_plan, self.valid_set)
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
            optimized_code = self.agent(messages=optimize_prompt, parse_code=True, lang="python")["output"]
            if optimized_code == "":
                round += 1
                continue
            optimized_plan = self._build_plan_from_code(optimized_code, columns)
            optimized_code, plan_stats = self._build_code_from_plan(optimized_plan, input_dataset_name=input_dataset_name)

            # 2.3. compare the results with the ground truth
            # results_on_valid_set, token_cost, runtime = execute_plan(optimized_plan, self.valid_set)
            accuracy_score, token_cost = self._naive_estimate_plan_cost(init_plan_stats, plan_stats)
            # accuracy_score = Evaluator.evaluate(ground_truth_on_valid_set, results_on_valid_set, self.agent)
            plan_cost = PlanCost(optimized_plan, optimized_code, accuracy_score, token_cost, 0.0)
            self.candidate_logical_plan.append(plan_cost)
            round += 1

        # 3. select the best plan from the candidate list
        best_plan = sorted(self.candidate_logical_plan)[0]
        optimize_end_time = time.time()
        optimize_time = optimize_end_time - optimize_start_time
        logger.info(f"Plan optimization is finished, taking {optimize_time:.2f} seconds. Here are some statistics:")
        logger.info(f"initial plan cost: {init_plan_cost.cost} -> optimized plan cost: {best_plan.cost}")
        logger.info(f"initial plan runtime: {init_plan_cost.runtime} -> optimized plan runtime: {best_plan.runtime}")
        logger.info(f"initial plan accuracy: {init_plan_cost.accuracy} -> optimized plan accuracy: {best_plan.accuracy}")
        return best_plan.plan
