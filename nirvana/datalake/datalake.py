import os
import asyncio
import pandas as pd

from nirvana.datalake.lake_manager import LakeMetadata, LakeManager
from nirvana.ops.discover import SummarizeOperation, VectorizeOperation, DiscoverOperation


class DataLake:
    """
    An interface over a data lake that serves data discovery (now) and more operations (in the future) over a data lake.
    """

    def __init__(self, embedding_model):
        self.mgr = LakeManager()
        self.embedding_model = embedding_model

    def register(self, lake_dir, storage: str = "local", **kwargs):
        self.lake_dir = lake_dir
        if storage == "local":
            for path in os.listdir(lake_dir):
                if os.path.isdir(os.path.join(lake_dir, path)):
                    self.register(os.path.join(lake_dir, path), storage, **kwargs)
                elif os.path.isfile(os.path.join(lake_dir, path)):
                    self.register_data(os.path.join(lake_dir, path))
                else:
                    ValueError(f"There is no such file or directory: {os.path.join(lake_dir, path)}")
        elif storage == "s3":
            raise NotImplementedError("S3 storage will be implemented in the near future.")
        else:
            raise ValueError(f"Unsupported storage: {storage}")
        
    def _read_context(self, context_path: str):
        file_ext = os.path.splitext(context_path)[-1][1:]
        if file_ext == "txt":
            with open(context_path, "r") as f:
                return f.read()
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")            

    def register_data(self, data_path, context_path: str = None):
        if context_path is not None:
            context = self._read_context(context_path)
        else:
            context = None
        metadata = LakeMetadata(data_path, context)
        self.mgr.append(metadata)

    def summarize_data(
            self,
            column_level: bool = True, 
            record_level: bool = True, 
            sample_size: int = 5,
            strategy: str = None,
            max_length: int = 4096,
            **kwargs
    ):
        summarize_op = SummarizeOperation(**kwargs)
        total_cost = 0.0
        for metadata in self.mgr:
            data_to_summarize = pd.read_csv(metadata.data_path)
            summarize_output = asyncio.run(
                summarize_op.execute(
                    data_to_summarize, 
                    column_level=column_level, 
                    record_level=record_level,
                    sample_size=sample_size,
                    strategy=strategy,
                    max_length=max_length,
                )
            )
            metadata.column_summary, metadata.record_summary = summarize_output.column_summary, summarize_output.record_summary
            metadata.is_summarized = True
            total_cost += summarize_output.cost
        return total_cost

    def generate_index(self, **kwargs):
        vectorize_op = VectorizeOperation(**kwargs)
        vectorize_output = asyncio.run(self.vectorize_op.execute(
            self.mgr,
            self.embedding_model,
        ))
        self.mgr.text_index.index(vectorize_output.lexical_vectors, show_progress=False)
        corpus = [{"id": idx, "text": doc} for idx, doc in zip(vectorize_output.doc_ids, vectorize_output.documents)]
        self.mgr.text_index.corpus = self.mgr.text_index.corpus + corpus
        self.mgr.text_index.save(
            os.path.join(self.mgr.text_index_path, "bm25s_index"),
            corpus=self.mgr.text_index.corpus
        )
        self.mgr.vector_index.add(
            embeddings=vectorize_output.semantic_vectors,
            documents=vectorize_output.documents,
            ids=vectorize_output.doc_ids
        )
        return vectorize_output.cost

    def discover_data(
        self, 
        query: str,
        topk: int = 1,
        pool_factor: int = 5,
        alpha: float = 0.5,
        **kwargs
    ):
        discover_op = DiscoverOperation(**kwargs)
        discover_output = asyncio.run(
            discover_op.execute(
                query,
                vector_index=self.mgr.vector_index,
                text_index=self.mgr.text_index,
                embedding_model=self.embedding_model,
                topk=topk,
                pool_factor=pool_factor,
                alpha=alpha
            )
        )
        pass
