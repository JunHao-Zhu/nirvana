import os
from typing import Optional
from collections.abc import MutableSequence
import bm25s
import chromadb


class LakeMetadata:
    path: str
    context: Optional[str] = None
    column_summary: Optional[str] = None
    record_summary: Optional[str] = None
    is_summarized: bool = False
    is_vector_indexed: bool = False
    is_text_indexed: bool = False


class LakeManager(MutableSequence):
    """
    Core internal intermediator for the data lake (e.g., local storage, S3, and more).

    Manage metadata for a data lake, for example, tracking the size of the data lake,
    providing accesses to data files in the data lake, storing data summaries, etc.
    """

    def __init__(self, vector_index_path: str = None, text_index_path: str = None):
        super().__init__()
        self.vector_index_path = vector_index_path
        self.text_index_path = text_index_path
        self.vector_index = self.__init_vector_index__()
        self.text_index = self.__init_text_index__()

    def __init_vector_index__(self):
        client = chromadb.PersistentClient(path=self.vector_base_path)
        vector_db = client.create_collection(
            name="vector_index", 
            metadata={
                "hnsw:space": "cosine",
                "hnsw:random_seed": 42,
                "hnsw:M": 48,
            }
        )
        return vector_db
    
    def __init_text_index__(self):
        indexer = bm25s.BM25(corpus=[])
        corpus_tokens = bm25s.tokenize([])
        indexer.index(corpus_tokens, show_progress=False)
        indexer.save(
            os.path.join(self.text_index_path, "bm25s_index")
        )
        return indexer

    def __setitem__(
            self, 
            index: int, 
            value: LakeMetadata
    ):
        super().__setitem__(index, value)

    def __getitem__(self, index: int) -> LakeMetadata:
        return super().__getitem__(index)
    
    def __len__(self) -> int:
        return super().__len__()
    
    def __delitem__(self, index: int):
        super().__delitem__(index)

    def insert(self, index, value: LakeMetadata):
        return super().insert(index, value)
