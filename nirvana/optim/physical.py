import time
import asyncio
import logging
import functools
import numpy as np
import pandas as pd

from nirvana.executors.llm_backbone import LLMClient
from nirvana.lineage.abstractions import LineageNode, OpOutputsType

logger = logging.getLogger(__name__)


ORDERED_MODELS= [  # llm list ordered by speed and intelligence (weak but fast -> strong but slow)
    "gpt-4.1-nano-2025-04-14",
    "gpt-4o-mini-2024-07-18",
    "gpt-4.1-mini-2025-04-14",
    "gpt-4.1-2025-04-14",
]


def concate_output(output1: OpOutputsType, output2: OpOutputsType):
    return output1 + output2


def cosine_similarity(x: np.array, y: np.array):
    x_norms = np.linalg.norm(x, axis=1)
    y_norms = np.linalg.norm(y, axis=1)
    dot_products = np.sum(x * y, axis=1)
    return dot_products / (x_norms * y_norms)


class PhysicalOptimizer:
    def __init__(
            self,
            agent: LLMClient = None,
            available_models: list[str] = ORDERED_MODELS,
    ):
        self.agent = agent
        self.available_models = available_models if available_models else ORDERED_MODELS

    def should_optimize(self, op_name: str, input_data: pd.DataFrame | list[pd.DataFrame], num_samples: int) -> bool:
        if op_name == "join":
            # and len(input_data[0]) * len(input_data[1]) < 2 * num_samples:
            # we don't optimize join yet
            return False
        elif op_name in ["filter", "map"] and len(input_data[0]) < 2 * num_samples:
            return False
        elif op_name in ["reduce", "rank"]:
            return False
        else:
            return True

    def split_input_data(self, op_name: str, input_data: pd.DataFrame | list[pd.DataFrame], num_samples: int):
        if op_name == "join":
            left_length, right_length = len(input_data[0]), len(input_data[1])
            if left_length > right_length:
                return [input_data[0].iloc[:num_samples], input_data[1]], [input_data[0].iloc[num_samples:], input_data[1]]
            else:
                return [input_data[0], input_data[1].iloc[:num_samples]], [input_data[0], input_data[1].iloc[num_samples:]]
        else:
            return input_data.iloc[:num_samples], input_data.iloc[num_samples:]

    async def optimize_exec_model(
            self, 
            node: LineageNode, 
            input_data: pd.DataFrame | list[pd.DataFrame], 
            num_samples: int, 
            improve_margin: float = 0.2
    ):
        if not self.should_optimize(node.op_name, input_data, num_samples):
            node.operator.model = self.available_models[0] if node.op_name in ["join", "rank"] else self.available_models[-1]
            node_output = await node.execute_operation(input_data)
            return node_output

        optimize_start_time = time.time()
        train_set, test_set = self.split_input_data(node.op_name, input_data, num_samples)
        node.operator.model = self.available_models[0] 
        node_output = None

        if node.op_name == "filter":
            worst_output = await node.execute_operation(train_set)
            node_output = worst_output
            
            improve_score_list = [0.0]
            mismatch_idx_worst2subworst = None
            mismatch_idx_subworst2subbest = None
            for model_id in range(1, len(ORDERED_MODELS)):
                if model_id == 1:
                    better_model = ORDERED_MODELS[model_id]
                    node.operator.model = better_model
                    better_output = await node.execute_operation(train_set)
                    node_output.cost += better_output.cost
                    subworst_output = np.array(better_output.output)
                    mismatch_idx_worst2subworst = np.array(worst_output.output) != subworst_output
                    mismatch_record_ratio = sum(mismatch_idx_worst2subworst) / len(worst_output.output)
                    improve_score_list.append(mismatch_record_ratio)

                elif model_id == 2:
                    better_model = ORDERED_MODELS[model_id]
                    node.operator.model = better_model
                    better_output = await node.execute_operation(train_set)
                    node_output.cost += better_output.cost
                    subbest_output = np.array(better_output.output)
                    mismatch_idx_subworst2subbest = subworst_output != subbest_output
                    if mismatch_idx_worst2subworst.all():
                        mismatch_ratio = 0.0
                    else:
                        mismatch_ratio = (
                            (mismatch_idx_subworst2subbest & ~mismatch_idx_worst2subworst).sum() / (~mismatch_idx_worst2subworst).sum()
                        )
                    improve_score = improve_score_list[-1] + mismatch_ratio * (1 - improve_score_list[-1])
                    improve_score_list.append(improve_score)

                elif model_id == 3:
                    better_model = ORDERED_MODELS[model_id]
                    node.operator.model = better_model
                    all_matched_idx = ~mismatch_idx_subworst2subbest & ~mismatch_idx_worst2subworst
                    if all_matched_idx.any():
                        better_output = await node.execute_operation(train_set.iloc[all_matched_idx])
                        node_output.cost += better_output.cost
                        best_output = np.array(better_output.output)
                        mismatch_idx_subbest2best = subbest_output[all_matched_idx] != best_output
                        mismatch_ratio = mismatch_idx_subbest2best.sum() / all_matched_idx.sum()
                        subbest_output[all_matched_idx] = best_output
                        better_output.output = subbest_output.tolist()
                    else:
                        mismatch_ratio = 0.0
                    if mismatch_idx_worst2subworst.any():
                        matched_ratio = (
                            (~mismatch_idx_subworst2subbest & mismatch_idx_worst2subworst).sum() / mismatch_idx_worst2subworst.sum()
                        )
                    else:
                        matched_ratio = 0.0
                    improve_score = mismatch_ratio * (1 - improve_score_list[-1]) + improve_score_list[-1] - improve_score_list[-2] + matched_ratio
                    improve_score_list.append(improve_score)

                if improve_score_list[-1] - improve_score_list[-2] > improve_margin:
                    exec_model = better_model
                    node_output.output = better_output.output

        elif node.op_name == "map":
            worst_output = await node.execute_operation(train_set)
            node_output = worst_output
            worst_output_embeds, embed_cost = await self.agent.create_embedding(worst_output.output)
            node_output.cost += embed_cost

            improve_score_list = [0.0]
            mismatch_idx_worst2subworst = None
            mismatch_idx_subworst2subbest = None
            for model_id in range(1, len(ORDERED_MODELS)):
                if model_id == 1:
                    better_model = ORDERED_MODELS[model_id]
                    node.operator.model = better_model
                    better_output = await node.execute_operation(train_set)
                    node_output.cost += better_output.cost
                    subworst_output = np.array(better_output.output)
                    subworst_output_embeds, embed_cost = await self.agent.create_embedding(better_output.output)
                    node_output.cost += embed_cost
                    cos_sim = cosine_similarity(worst_output_embeds, subworst_output_embeds)
                    mismatch_idx_worst2subworst = cos_sim < 0.5
                    mismatch_ratio = sum(mismatch_idx_worst2subworst) / len(worst_output.output)
                    improve_score_list.append(mismatch_ratio)
                
                elif model_id == 2:
                    better_model = ORDERED_MODELS[model_id]
                    node.operator.model = better_model
                    better_output = await node.execute_operation(train_set)
                    node_output.cost += better_output.cost
                    subbest_output = np.array(better_output.output)
                    subbest_output_embeds, embed_cost = await self.agent.create_embedding(better_output.output)
                    node_output.cost += embed_cost
                    cos_sim = cosine_similarity(subworst_output_embeds, subbest_output_embeds)
                    mismatch_idx_subworst2subbest = cos_sim < 0.5
                    if mismatch_idx_worst2subworst.all():
                        mismatch_ratio = 0.0
                    else:
                        mismatch_ratio = (
                            (mismatch_idx_subworst2subbest & ~mismatch_idx_worst2subworst).sum() / (~mismatch_idx_worst2subworst).sum()
                        )
                    improve_score = improve_score_list[-1] + mismatch_ratio * (1 - improve_score_list[-1])
                    improve_score_list.append(improve_score)
                
                elif model_id == 3:
                    better_model = ORDERED_MODELS[model_id]
                    node.operator.model = better_model
                    all_matched_idx = ~mismatch_idx_subworst2subbest & ~mismatch_idx_worst2subworst
                    if all_matched_idx.any():
                        better_output = await node.execute_operation(train_set.iloc[all_matched_idx])
                        node_output.cost += better_output.cost
                        best_output = np.array(better_output.output)
                        best_output_embeds, embed_cost = await self.agent.create_embedding(better_output.output)
                        node_output.cost += embed_cost
                        cos_sim = cosine_similarity(subbest_output_embeds[all_matched_idx], best_output_embeds)
                        mismatch_idx_subbest2best = cos_sim < 0.5
                        mismatch_ratio = mismatch_idx_subbest2best.sum() / all_matched_idx.sum()
                        subbest_output[all_matched_idx] = best_output
                        better_output.output = subbest_output.tolist()
                    else:
                        mismatch_ratio = 0.0
                    if mismatch_idx_worst2subworst.any():
                        matched_ratio = (
                            (~mismatch_idx_subworst2subbest & mismatch_idx_worst2subworst).sum() / mismatch_idx_worst2subworst.sum()
                        )
                    else:
                        matched_ratio = 0.0
                    improve_score = mismatch_ratio * (1 - improve_score_list[-1]) + improve_score_list[-1] - improve_score_list[-2] + matched_ratio
                    improve_score_list.append(improve_score)

                if improve_score_list[-1] - improve_score_list[-2] > improve_margin:
                    exec_model = better_model
                    node_output.output = better_output.output
        
        optimize_end_time = time.time()
        logger.info(f"Physical Plan Optimization Time: {optimize_end_time - optimize_start_time:.4f} sec")

        node.set_exec_model(exec_model)
        rest_output = await node.execute_operation(test_set)
        node_output = concate_output(node_output, rest_output)
        return node_output

    def optimize(
            self, 
            plan: LineageNode,
            num_samples: int = 10,
            improve_margin: float = 0.2,
    ):
        optimize_output = {
            "total_token_cost": 0.0
        }
        optimize_func = functools.partial(self.optimize_exec_model, num_samples=num_samples, improve_margin=improve_margin)
        
        def _optimize_node(node: LineageNode) -> pd.DataFrame:
            if node.left_child:
                dataframe_from_left_node = _optimize_node(node.left_child)
            if node.right_child:
                dataframe_from_right_node = _optimize_node(node.right_child)

            if node.op_name == "scan":
                output_from_node = asyncio.run(node.run())
                dataframe_from_node = output_from_node.output
                optimize_output["total_token_cost"] += output_from_node.cost
                return dataframe_from_node
            
            elif node.op_name == "join":
                output_from_node = asyncio.run(optimize_func(node, [dataframe_from_left_node, dataframe_from_right_node]))
                dataframe_from_node = asyncio.run(node.collate_dataframe([dataframe_from_left_node, dataframe_from_right_node], output_from_node))
                optimize_output["total_token_cost"] += output_from_node.cost
                return dataframe_from_node
            
            else:
                output_from_node = asyncio.run(optimize_func(node, dataframe_from_left_node))
                dataframe_from_node = asyncio.run(node.collate_dataframe(dataframe_from_left_node, output_from_node))
                optimize_output["total_token_cost"] += output_from_node.cost
                return dataframe_from_node
        
        execution_start_time = time.time()
        dataframe_from_node = _optimize_node(plan)
        execution_end_time = time.time()
        execution_time = execution_end_time - execution_start_time
        return dataframe_from_node, optimize_output["total_token_cost"], execution_time
