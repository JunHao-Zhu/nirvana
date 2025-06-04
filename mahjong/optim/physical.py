import time
import functools
import numpy as np
import pandas as pd

from mahjong.models.llm_backbone import LLMClient
from mahjong.lineage.abstractions import LineageNode, LineageDataNode, LineageOpNode, OpOutputsType


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
    def __init__(self, agent: LLMClient = None):
        self.agent = agent

    def optimize_exec_model_approx(self, node: LineageOpNode, input_data: pd.DataFrame, num_samples: int, improve_margin: float = 0.2):
        if node.func is not None:
            return node.run(input_data)
        if len(input_data) < 2 * num_samples:
            node.set_exec_model(ORDERED_MODELS[-1])
            return node.run(input_data)

        data_samples = input_data.iloc[:num_samples]
        exec_model = ORDERED_MODELS[0]
        node.set_exec_model(exec_model)
        node_output = None

        if node.op_name == "reduce":
            exec_model = ORDERED_MODELS[-1]
            node.set_exec_model(exec_model)
            node_output = node.run(input_data)
            return node_output

        elif node.op_name == "filter":
            worst_output = node.run(data_samples)
            node_output = worst_output
            
            improve_score_list = [0.0]
            mismatch_idx_worst2subworst = None
            mismatch_idx_subworst2subbest = None
            for model_id in range(1, len(ORDERED_MODELS)):
                if model_id == 1:
                    better_model = ORDERED_MODELS[model_id]
                    node.set_exec_model(better_model)
                    better_output = node.run(data_samples)
                    node_output.cost += better_output.cost
                    subworst_output = np.array(better_output.output)
                    mismatch_idx_worst2subworst = np.array(worst_output.output) != subworst_output
                    mismatch_record_ratio = sum(mismatch_idx_worst2subworst) / len(worst_output.output)
                    improve_score_list.append(mismatch_record_ratio)

                elif model_id == 2:
                    better_model = ORDERED_MODELS[model_id]
                    node.set_exec_model(better_model)
                    better_output = node.run(data_samples)
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
                    node.set_exec_model(better_model)
                    all_matched_idx = ~mismatch_idx_subworst2subbest & ~mismatch_idx_worst2subworst
                    if all_matched_idx.any():
                        better_output = node.run(data_samples.iloc[all_matched_idx])
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

        else:
            worst_output = node.run(data_samples)
            node_output = worst_output
            worst_output_embeds, embed_cost = self.agent.create_embedding(worst_output.output)
            node_output.cost += embed_cost

            improve_score_list = [0.0]
            mismatch_idx_worst2subworst = None
            mismatch_idx_subworst2subbest = None
            for model_id in range(1, len(ORDERED_MODELS)):
                if model_id == 1:
                    better_model = ORDERED_MODELS[model_id]
                    node.set_exec_model(better_model)
                    better_output = node.run(data_samples)
                    node_output.cost += better_output.cost
                    subworst_output = np.array(better_output.output)
                    subworst_output_embeds, embed_cost = self.agent.create_embedding(better_output.output)
                    node_output.cost += embed_cost
                    cos_sim = cosine_similarity(worst_output_embeds, subworst_output_embeds)
                    mismatch_idx_worst2subworst = cos_sim < 0.5
                    mismatch_ratio = sum(mismatch_idx_worst2subworst) / len(worst_output.output)
                    improve_score_list.append(mismatch_ratio)
                
                elif model_id == 2:
                    better_model = ORDERED_MODELS[model_id]
                    node.set_exec_model(better_model)
                    better_output = node.run(data_samples)
                    node_output.cost += better_output.cost
                    subbest_output = np.array(better_output.output)
                    subbest_output_embeds, embed_cost = self.agent.create_embedding(better_output.output)
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
                    node.set_exec_model(better_model)
                    all_matched_idx = ~mismatch_idx_subworst2subbest & ~mismatch_idx_worst2subworst
                    if all_matched_idx.any():
                        better_output = node.run(data_samples.iloc[all_matched_idx])
                        node_output.cost += better_output.cost
                        best_output = np.array(better_output.output)
                        best_output_embeds, embed_cost = self.agent.create_embedding(better_output.output)
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

        node.set_exec_model(exec_model)
        rest_output = node.run(input_data.iloc[num_samples:])
        node_output = concate_output(node_output, rest_output)
        return node_output

    def optimize_exec_model(self, node: LineageOpNode, input_data: pd.DataFrame, num_samples: int, improve_margin: float = 0.2):
        if len(input_data) < 2 * num_samples:
            node.set_exec_model(ORDERED_MODELS[-1])
            return node.run(input_data)
        
        data_samples = input_data.iloc[:num_samples]
        best_model = ORDERED_MODELS[-1]
        exec_model = ORDERED_MODELS[0]
        node.set_exec_model(exec_model)
        node_output = None

        if node.op_name == "reduce":
            exec_model = ORDERED_MODELS[-1]
            node.set_exec_model(exec_model)
            node_output = node.run(input_data)
            return node_output

        elif node.op_name == "filter":
            worst_output = node.run(data_samples)
            node_output = worst_output
            curr_improve_score = 0.0
            for model_id in range(1, len(ORDERED_MODELS)):
                better_model = ORDERED_MODELS[model_id]
                node.set_exec_model(better_model)
                better_output = node.run(data_samples)
                node_output.cost += better_output.cost
                mismatch_record_idx = np.array(worst_output.output) != np.array(better_output.output)
                mismatch_record_ratio = sum(mismatch_record_idx) / len(worst_output.output)

                if better_model == best_model:
                    improve_score = mismatch_record_ratio
                else:
                    node.set_exec_model(best_model)
                    best_output = node.run(data_samples.iloc[mismatch_record_idx])
                    node_output.cost += best_output.cost
                    matched_record_idx = np.array(better_output.output) == np.array(best_output.output)
                    matched_record_ratio = sum(matched_record_idx) / len(best_output.output)
                    improve_score = matched_record_ratio * mismatch_record_ratio
                
                if (improve_score - curr_improve_score) > improve_margin:
                    exec_model = better_model
                    node_output.output = better_output.output
                    curr_improve_score = improve_score

        else:
            worst_output = node.run(data_samples)
            node_output = worst_output
            worst_output_embeds, embed_cost = self.agent.create_embedding(worst_output.output)
            node_output.cost += embed_cost
            curr_improve_score = 0.0
            for model_id in range(1, len(ORDERED_MODELS)):
                better_model = ORDERED_MODELS[model_id]
                node.set_exec_model(better_model)
                better_output = node.run(data_samples)
                node_output.cost += better_output.cost
                better_output_embeds, embed_cost = self.agent.create_embedding(better_output.output)
                node_output.cost += embed_cost
                cos_sim = cosine_similarity(worst_output_embeds, better_output_embeds)
                mismatch_record_idx = cos_sim < 0.5
                mismatch_record_ratio = sum(mismatch_record_idx) / len(worst_output_embeds)

                if better_model == best_model:
                    improve_score = mismatch_record_ratio
                else:
                    node.set_exec_model(best_model)
                    best_output = node.run(data_samples.iloc[mismatch_record_idx])
                    node_output.cost += best_output.cost
                    best_output_embeds, embed_cost = self.agent.create_embedding(best_output.output)
                    node_output.cost += embed_cost
                    cos_sim = cosine_similarity(worst_output_embeds[mismatch_record_idx], best_output_embeds)
                    matched_record_idx = cos_sim > 0.5
                    matched_record_ratio = sum(matched_record_idx) / len(mismatch_record_idx)
                    improve_score = matched_record_ratio * mismatch_record_ratio
                
                if (improve_score - curr_improve_score) > improve_margin:
                    exec_model = better_model
                    node_output.output = better_output.output
                    curr_improve_score = improve_score

        node.set_exec_model(exec_model)
        rest_output = node.run(input_data.iloc[num_samples:])
        node_output = concate_output(node_output, rest_output)
        return node_output

    def optimize(
            self, 
            plan: LineageNode, 
            input_data: pd.DataFrame, 
            num_samples: int = 10, 
            improve_margin: float = 0.2, 
            approx_mode: bool = True
    ):
        node_output = {
            "dataframe_from_node": input_data.copy(),
            "output_from_node": None,
            "total_token_cost": 0,
        }
        if approx_mode:
            optimize_func = functools.partial(self.optimize_exec_model_approx, num_samples=num_samples, improve_margin=improve_margin)
        else:
            optimize_func = functools.partial(self.optimize_exec_model, num_samples=num_samples, improve_margin=improve_margin)
        
        def _optimize_node(node: LineageNode):
            if len(node.parent) == 0:
                assert isinstance(node, LineageOpNode), "The first node should be an operator."
                output_from_node = optimize_func(node, node_output["dataframe_from_node"])
                node_output["output_from_node"] = output_from_node
                node_output["total_token_cost"] += output_from_node.cost
                return
            for parent_node in node.parent:
                _optimize_node(parent_node)

            if isinstance(node, LineageDataNode):
                dataframe_from_node = node.run(node_output["dataframe_from_node"], node_output["output_from_node"])
                node_output["dataframe_from_node"] = dataframe_from_node
                return

            if isinstance(node, LineageOpNode):
                output_from_node = optimize_func(node, node_output["dataframe_from_node"])
                node_output["output_from_node"] = output_from_node
                node_output["total_token_cost"] += output_from_node.cost
                return
        
        execution_start_time = time.time()
        _optimize_node(plan)
        execution_end_time = time.time()
        execution_time = execution_end_time - execution_start_time
        return node_output["dataframe_from_node"], node_output["total_token_cost"], execution_time
