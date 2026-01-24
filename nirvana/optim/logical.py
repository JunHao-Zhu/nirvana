import logging
import asyncio
from functools import partial
import time
import numpy as np
import pandas as pd

from nirvana.executors.llm_backbone import LLMClient
from nirvana.lineage.abstractions import LineageNode, execute_node
from nirvana.optim.rules import (
    FilterPushdown,
    FilterPullup,
    MapPullup,
    NonLLMPushdown,
    NonLLMReplace,
    OperatorFusion,
)
from nirvana.optim.evaluator import Evaluator

logger = logging.getLogger(__name__)


class PlanCost:
    error_tolerance = 0.8

    def __init__(self, plan: LineageNode, rules: set, accuracy: float, cost: float, runtime: float):
        self.plan = plan
        self.rules = rules
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
        max_round: int = 5,
        agent: LLMClient = None,
        filter_pullup: bool = True,
        filter_pushdown: bool = True,
        map_pullup: bool = True,
        non_llm_pushdown: bool = True,
        non_llm_replace: bool = True,
        operator_fusion: bool = True,
    ):
        self.agent = agent
        self.max_round = max_round
        if non_llm_replace or operator_fusion:
            assert agent is not None, "LLM must be provided for semantic-aware plan rewrites, e.g., for Non-LLM Replace and Operator Fusion."
        self.rules = set()
        if filter_pullup:
            self.rules.add(FilterPullup.transform)
        if filter_pushdown:
            self.rules.add(FilterPushdown.transform)
        if map_pullup:
            self.rules.add(MapPullup.transform)
        if non_llm_pushdown:
            self.rules.add(NonLLMPushdown.transform)
        if non_llm_replace:
            self.rules.add(partial(NonLLMReplace.transform, rewriter=self.agent))
        if operator_fusion:
            self.rules.add(partial(OperatorFusion.transform, rewriter=self.agent))
        self.plan_candidates = []

    def clear(self):
        self.rules = set()
        self.plan_candidates = []

    def sample_from_candidates(self, lambda_=0.2):
        token_costs = np.array([plan.cost for plan in self.plan_candidates])
        token_costs = (np.max(token_costs) - token_costs) / (np.max(token_costs) - np.min(token_costs) + 1e-6)
        costs_prob = np.exp(token_costs) / np.exp(token_costs).sum()
        uniform_prob = np.ones_like(token_costs, dtype=float) / len(token_costs)
        total_prob = lambda_ * uniform_prob + (1 - lambda_) * costs_prob

        selected_index = np.random.choice(len(token_costs), p=total_prob)
        return self.plan_candidates[selected_index]

    def prepare_for_logical_optimization(
        self,
        plan: LineageNode,
        dev_datasets: list | None,
        num_samples: int | None,
    ):
        initial_state = []
        def prepare_dev_datasets(node: LineageNode):
            if node.left_child:
                prepare_dev_datasets(node.left_child)
            if node.right_child:
                prepare_dev_datasets(node.right_child)
            
            if node.op_name == "scan":
                initial_state.append(node.datasource)
                if dev_datasets:
                    node.datasource = dev_datasets.pop(0)
                elif num_samples is not None:
                    node.operator.set_sample_size(num_samples)
                else:
                    raise ValueError(
                        "Either `dev_datasets` or `num_samples` must be provided to prepare the development datasets for logical optimization."
                    )
                return
        prepare_dev_datasets(plan)
        return initial_state

    def recover_plan_state(self, initial_state: list, plan: LineageNode):
        def reset_datasources(node: LineageNode):
            if node.left_child:
                reset_datasources(node.left_child)
            if node.right_child:
                reset_datasources(node.right_child)
            
            if node.op_name == "scan":
                node.datasource = initial_state.pop(0)
                node.operator.set_sample_size(None)
                return
        reset_datasources(plan)
    
    async def estimate_plan_cost(self, plan: LineageNode, ground_truth: pd.DataFrame = None):
        try:
            results, token_cost, exec_time = await execute_node(plan)
        except Exception as e:
            return None, 0.0, 0.0, 0.0
        if ground_truth is None:
            accuracy_score = 1.0
        else:
            accuracy_score = await Evaluator.evaluate(ground_truth, results, self.agent)
        return results, accuracy_score, token_cost, exec_time
    
    async def _optimize(self, initial_plan: LineageNode, dev_datasets: list | None = None, num_samples: int = 10):
        round = 0
        optimize_cost = 0.0
        optimize_start_time = time.time()
        # 0. prepare for logical optimization
        initial_state = self.prepare_for_logical_optimization(
            initial_plan,
            dev_datasets=dev_datasets,
            num_samples=num_samples,
        )
        
        # 1. get the data processing ground truth by executing the initial plan on the validation set
        groundtruth, accuracy_score, token_cost, exec_time = await self.estimate_plan_cost(initial_plan)
        init_plan_cost = PlanCost(initial_plan, set(), accuracy_score, token_cost, exec_time)
        self.plan_candidates.append(init_plan_cost)

        # 2. optimize the plan
        while round < self.max_round:
            logger.info(f"Round {round + 1} of logical optimization is started.")
            # 2.1. sample a plan from the candidate list
            cand_plan_cost = self.sample_from_candidates()

            # 2.2. select a transformation rule that has not been applied to the sampled plan
            applicable_rules = self.rules - cand_plan_cost.rules
            if len(applicable_rules) == 0:
                round += 1
                continue

            selected_rule = np.random.choice(list(applicable_rules))
            optimized_plan, cost_per_rewrite = await selected_rule(cand_plan_cost.plan)
            optimize_cost += cost_per_rewrite

            # 2.3. compare the results with the ground truth
            _, accuracy_score, token_cost, exec_time = await self.estimate_plan_cost(optimized_plan, groundtruth)
            applied_rules = cand_plan_cost.rules.union({selected_rule})
            plan_cost = PlanCost(optimized_plan, applied_rules, accuracy_score, token_cost, exec_time)
            self.plan_candidates.append(plan_cost)
            round += 1

        # 3. select the best plan from the candidate list
        best_plan = sorted(self.plan_candidates)[0]
        optimize_end_time = time.time()
        optimize_time = optimize_end_time - optimize_start_time

        # 4. restore the state for the final best plan
        self.recover_plan_state(initial_state, best_plan.plan)

        logger.info(f"Plan optimization is finished, taking {optimize_time:.4f} seconds and ${optimize_cost:.4f}. Here are some statistics:")
        logger.info(f"initial plan cost: {init_plan_cost.cost} -> optimized plan cost: {best_plan.cost}")
        logger.info(f"initial plan runtime: {init_plan_cost.runtime} -> optimized plan runtime: {best_plan.runtime}")
        logger.info(f"initial plan accuracy: {init_plan_cost.accuracy} -> optimized plan accuracy: {best_plan.accuracy}")        
        return best_plan.plan

    def optimize(
        self,
        plan: LineageNode,
        dev_datasets: list | None = None,
        num_samples: int = 10,
    ):
        return asyncio.run(self._optimize(plan, dev_datasets, num_samples))
