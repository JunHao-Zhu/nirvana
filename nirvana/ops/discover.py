import os
import asyncio
from enum import Enum
from typing import Any, List, Iterable, Union
from dataclasses import dataclass
import bm25s
import Stemmer
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine

from nirvana.ops.base import BaseOpOutputs, BaseOperation
from nirvana.ops.prompt_templates.discover_prompter import SummarizePrompter, RerankPrompter


def summarize_wrapper(**kwargs):
    pass


def discover_wrapper(**kwargs):
    pass


class DiscoveryType(Enum):
    CONTEXT = 1
    COLUMN = 2
    RECORD = 3


@dataclass
class SummarizeOpOutputs(BaseOpOutputs):
    column_summary: str = None
    record_summary: str = None


@dataclass
class VectorizeOpOutputs(BaseOpOutputs):
    doc_ids: List[str] = None
    semantic_vectors: List[List[float]] = None
    lexical_vectors: Union[List[List[str]], bm25s.tokenization.Tokenized] = None
    documents: List[str] = None


@dataclass
class DiscoverOpOutputs(BaseOpOutputs):
    table_ids: List[int] = None
    scores: List[float] = None
    types: List[DiscoveryType] = None


class SummarizeOperation(BaseOperation):
    """ TODO: apply data summary on a single data (table) 
    Operation for data summary that summarize the core concept of a table from its columns and rows.
    Currently, the main purpose of this operation is serving data discovery.
    
    The algorithm is adopted from [Pneuma: Leveraging LLMs for Tabular Data Representation and Retrieval in an End-to-End System](https://arxiv.org/abs/2504.09207)
    """
    
    def __init__(
            self,
            *args,
            **kwargs,
    ):
        super().__init__("summarize", *args, **kwargs)
        self.prompter = SummarizePrompter()
        rate_limit = kwargs.get("rate_limit", 16)
        self.semaphore = asyncio.Semaphore(rate_limit)

    async def _summarize_column_by_llm(self, target_column: Any, all_columns: Iterable[str], **kwargs):
        async with self.semaphore:
            columns = " | ".join(all_columns)
            full_prompt = self.prompter.generate_prompt(target_column, columns)
            output = await self.llm(full_prompt, parse_tags=False, **kwargs)
            return output["output"], output["cost"]
        
    def _summarize_record(self, sampled_rows: pd.DataFrame, **kwargs):
        rows = []
        for row_idx, row in sampled_rows.iterrows():
            row = " | ".join([f"{col}: {val}" for col, val in row.items()])
            rows.append(row.strip())
        return rows

    def merge_summarizes(self, summarizes: List[str], strategy=None, max_length=None):
        merged_summary = ""
        if strategy == "truncate":
            for desc_id, desc in enumerate(summarizes):
                desc = f"{desc} || " if desc_id < len(summarizes) - 1 else desc
                if len(merged_summary) + len(desc) <= max_length:
                    merged_summary += desc
                else:
                    break
        else:
            for desc_id, desc in enumerate(summarizes):
                desc = f"{desc} || " if desc_id < len(summarizes) - 1 else desc
                merged_summary += desc
        return merged_summary

    async def execute(
            self,
            data_from_lake: pd.DataFrame,
            column_level: bool = True,
            record_level: bool = True,
            sample_size: int = 5,
            strategy: str = None,
            max_length: int = 4096,
            **kwargs
    ):
        token_cost = 0.0
        column_summary, record_summary = None, None
        if column_level:
            tasks = []
            for col in data_from_lake.columns:
                tasks.append(
                    asyncio.create_task(self._summarize_column_by_llm(col, data_from_lake.columns, **kwargs))
                )
            results = await asyncio.gather(*tasks)
            column_summarizes = [f"{col}: {result['output'].strip()}" for col, result in zip(data_from_lake.columns, results)]
            token_cost += sum([result["cost"] for result in results])
            column_summary = self.merge_summarizes(column_summarizes, strategy, max_length)

        if record_level:
            sample_size = min(sample_size, len(data_from_lake))
            sampled_rows = data_from_lake.sample(n=sample_size).reset_index(drop=True)
            record_summarizes = self._summarize_record(sampled_rows, **kwargs)
            record_summary = self.merge_summarizes(record_summarizes, strategy, max_length)
        
        return SummarizeOpOutputs(
            column_summary=column_summary, 
            record_summary=record_summary,
            cost=token_cost
        )
    

class VectorizeOperation(BaseOperation):
    def __init__(self, *args, **kwargs):
        super().__init__("index", *args, **kwargs)
        rate_limit = kwargs.get("rate_limit", 16)
        self.semaphore = asyncio.Semaphore(rate_limit)

    async def _prepare_documents(self, lake_manager):
        doc_idx, documents = [], []
        for idx, metadata in enumerate(lake_manager):
            if not metadata.is_summarized:
                continue
            if metadata.column_summary is not None:
                documents.append(metadata.column_summary)
                doc_idx.append(f"{idx}_column")
            if metadata.record_summary is not None:
                documents.append(metadata.record_summary)
                doc_idx.append(f"{idx}_record")
            if metadata.context is not None:
                documents.append(metadata.context)
                doc_idx.append(f"{idx}_context")
            metadata.is_vector_indexed = True
            metadata.is_text_indexed = True
        return doc_idx, documents

    async def _embed_documents(self, documents: List[str], embedding_model: str):
        with self.semaphore:
            return await self.llm.create_embedding(documents, embedding_model)

    async def execute(
            self, 
            lake_manager,
            embedding_model: str,
            *args, 
            **kwargs
    ):
        doc_idx, documents = await self._prepare_documents(lake_manager)
        tasks = []
        for i in range(0, len(documents), 30000):
            tasks.append(
                asyncio.create_task(self._embed_documents(documents[i:i+30000], embedding_model))
            )
        results = await asyncio.gather(*tasks)
        embeddings = np.concat([result[0] for result in results], axis=0)
        token_cost = sum([result[1] for result in results])

        tokenized_docs = bm25s.tokenize(
            documents,
            stopwords="en",
            stemmer=Stemmer.Stemmer("english"),
            show_progress=False
        )
        return VectorizeOpOutputs(
            doc_ids=doc_idx, 
            documents=documents,
            semantic_vectors=embeddings.tolist(),
            lexical_vectors=tokenized_docs,
            cost=token_cost
        )


class DiscoverOperation(BaseOperation):
    """ TODO: apply data discovery over a data lake
    Operation for data discovery that identify relevant data(tables) from data lakes for a data analytics task (in NL).
    
    The algorithm is adopted from [Pneuma: Leveraging LLMs for Tabular Data Representation and Retrieval in an End-to-End System](https://arxiv.org/abs/2504.09207)
    """
    
    def __init__(
            self,
            *args,
            **kwargs,
    ):
        super().__init__("discover", *args, **kwargs)
        self.prompter = RerankPrompter()
        rate_limit = kwargs.get("rate_limit", 16)
        self.semaphore = asyncio.Semaphore(rate_limit)

    def _adjust_index_search(self, search_results, misaligned_ids, reverse_vocab_dict, text_index, query):
        docs = [docs for docs in search_results[0][0]]
        scores = [score for score in search_results[1][0]]

        misaligned_docs = [
            text_index.corpus[reverse_vocab_dict[doc_id]] for doc_id in misaligned_ids
        ]
        misaligned_scores = [
            text_index.get_scores(
                bm25s.tokenization.convert_tokenized_to_string_list(query)[0]
            )[reverse_vocab_dict[doc_id]] for doc_id in misaligned_ids
        ]
        docs.extend(misaligned_docs)
        scores.extend(misaligned_scores)

        max_score, min_score = max(scores), min(scores)
        processed_results = {
            result["id"]: (
                1 if min_score == max_score else (scores[i] - min_score) / (max_score - min_score),
                result["text"]
            )
            for i, result in enumerate(docs)
        }
        return processed_results
    
    def _adjust_vector_search(self, search_results, misaligned_ids, vector_index, query):
        misaligned_results = vector_index.get_fast(
            ids=misaligned_ids, limit=len(misaligned_ids), include=["documents", "embeddings"]
        )
        search_results["ids"][0].extend(misaligned_results["ids"])
        search_results["documents"][0].extend(misaligned_results["documents"])
        search_results["distances"][0].extend(
            cosine(query, misaligned_results["embeddings"][i])
            for i in range(len(misaligned_ids))
        )
        scores = [1 - dist for dist in search_results["distances"][0]]
        dosc = search_results["documents"][0]
        ids = search_results["ids"][0]

        max_score, min_score = max(scores), min(scores)
        processed_results = {
            ids[i]: (
                1 if min_score == max_score else (scores[i] - min_score) / (max_score - min_score),
                dosc[i]
            )
            for i in range(len(scores))
        }
        return processed_results
    
    async def _judge_relevance_by_llm(self, query, doc_id, doc=None, **kwargs):
        with self.semaphore:
            if "context" in doc_id:
                full_prompt = self.prompter.generate_prompt(query, context_desc=doc)
            else:
                full_prompt = self.prompter.generate_prompt(query, columns_desc=doc)
            outputs = await self.llm(full_prompt, parse_tags=True, tags=["output"], **kwargs)
            return outputs["output"], outputs["cost"]
    
    async def _rerank_search_results(self, search_results, query, **kwargs):
        tasks = []
        for doc_id, _, doc in search_results:
            tasks.append(
                asyncio.create_task(self._judge_relevance_by_llm(query, doc_id, doc, **kwargs))
            )
        results = await asyncio.gather(*tasks)
        table_relevance = ["true" in result["output"].lower() for result in results]
        token_cost = sum([result["cost"] for result in results])
        rerank_results = (
            [result for idx, result in enumerate(search_results) if table_relevance[idx]] + 
            [result for idx, result in enumerate(search_results) if not table_relevance[idx]]
        )
        return rerank_results, token_cost
    
    def postprocess_discover_output(self, search_results, **kwargs):
        table_ids, scores, types = [], [], []
        for doc_id, score, doc in search_results:
            if "context" in doc_id:
                types.append(DiscoveryType.CONTEXT)
            elif "column" in doc_id:
                types.append(DiscoveryType.COLUMN)
            else:
                types.append(DiscoveryType.RECORD)
            table_ids.append(int(doc_id.split("_")[0]))
            scores.append(score)
        return table_ids, scores, types

    async def execute(
            self, 
            query: str,
            vector_index,
            text_index,
            embedding_model: str,
            topk: int = 1,
            pool_factor: int = 5,
            alpha: float = 0.5,
            **kwargs
    ):
        pool_size = pool_factor * topk
        token_cost = 0.0
        # process the query
        tokenized_query = bm25s.tokenize(
            query,
            stopwords="en",
            stemmer=Stemmer.Stemmer("english"),
            show_progress=False
        )
        embedded_query, embedding_query_cost = await self.llm.create_embedding(query, embedding_model)
        token_cost += embedding_query_cost

        # search relevant data with vector index and text index
        text_search_results = text_index.retrieve(
            tokenized_query,
            k=pool_size,
            show_progress=False
        )
        vector_search_results = vector_index.query(
            query_embeddings=embedded_query.tolist(),
            n_results=pool_size
        )

        # align search results between vector index and text index
        doc_ids_from_vector_search = set([doc_id for doc_id in vector_search_results["ids"][0]])
        doc_ids_from_text_search = set([result["id"] for result in text_search_results[0][0]])
        reverse_vocab_dict = {
            docs["id"]: idx for idx, docs in enumerate(text_index.corpus)
        }
        adjusted_text_search_results = self._adjust_index_search(
            text_search_results, 
            list(doc_ids_from_vector_search - doc_ids_from_text_search),
            reverse_vocab_dict,
            text_index,
            tokenized_query
        )
        adjusted_vector_search_results = self._adjust_vector_search(
            vector_search_results, 
            list(doc_ids_from_text_search - doc_ids_from_vector_search),
            vector_index,
            embedded_query
        )

        # rerank the search results and generate hybrid search results
        hybrid_search_results = []
        for doc_id in sorted(doc_ids_from_vector_search | doc_ids_from_text_search):
            result_from_text_search = adjusted_text_search_results.get(doc_id)
            result_from_vector_search = adjusted_vector_search_results.get(doc_id)
            combined_score = alpha * result_from_text_search[0] + (1 - alpha) * result_from_vector_search[0]
            if result_from_text_search is None:
                doc = result_from_vector_search[1]
            else:
                doc = result_from_text_search[1]
            hybrid_search_results.append((doc_id, combined_score, doc))
        
        sorted_hybrid_results = sorted(hybrid_search_results, key=lambda x: (-x[1], x[0]))[:topk]
        reranked_search_results, rerank_cost = await self._rerank_search_results(
            sorted_hybrid_results, 
            query, 
            **kwargs
        )
        token_cost += rerank_cost

        table_ids, scores, types = self.postprocess_discover_output(reranked_search_results)
        return DiscoverOpOutputs(table_ids=table_ids, scores=scores, types=types, cost=token_cost)
