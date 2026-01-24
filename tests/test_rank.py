import pytest
import pandas as pd
from unittest.mock import AsyncMock
import warnings

import nirvana as nv
from nirvana.ops.rank import RankOperation, RankOpOutputs
from nirvana.executors.llm_backbone import LLMClient
from nirvana.executors.tools import FunctionCallTool


@pytest.fixture
def sample_dataframe():
    """Create a sample dataframe for testing."""
    return pd.DataFrame({
        "Courses": ["Probability and Random Processes", "History", "Database", "Convex Optimization"]
    })


@pytest.fixture
def empty_dataframe():
    """Create an empty dataframe for testing."""
    return pd.DataFrame({"Courses": []})


@pytest.fixture
def dataframe_with_nan():
    """Create a dataframe with NaN values for testing."""
    return pd.DataFrame({
        "Courses": ["Probability and Random Processes", None, "Convex Optimization"]
    })


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client for testing."""
    
    async def mock_call(messages, parse_tags=False, parse_code=False, **kwargs):
        """Mock LLM call that returns structured output for rank operations."""
        # For ranking, the LLM should return "1" or "2" indicating which item is more relevant
        # For the test case "The course requires the least math":
        # - "Probability and Random Processes" likely requires more math
        # - "Convex Optimization" likely requires less math (or vice versa)
        # We'll return "2" to indicate item 2 is more relevant (requires less math)
        result = {
            "raw_output": "<output>2</output>",
            "cost": 0.01
        }
        
        if parse_tags:
            tags = kwargs.get("tags", [])
            for tag in tags:
                if tag == "output":
                    # Return "2" by default, meaning item 2 is more relevant
                    result[tag] = "2"
        
        return result
    
    # Create a proper async mock
    mock_client = AsyncMock(spec=LLMClient, side_effect=mock_call)
    mock_client.default_model = "gpt-4"
    return mock_client


class TestRankOperation:
    """Test suite for RankOperation class."""
    
    def test_rank_operation_initialization(self, mock_llm_client):
        """Test that RankOperation initializes correctly."""
        RankOperation.set_llm(mock_llm_client)
        op = RankOperation(
            user_instruction="The course requires the least math",
            input_columns=["Courses"],
            descend=True
        )
        
        assert op.op_name == "rank"
        assert op.input_columns == ["Courses"]
        assert op.descend == True
        assert op.dependencies == ["Courses"]
        assert op.generated_fields == []
    
    def test_op_kwargs_property(self, mock_llm_client):
        """Test that op_kwargs includes input_columns and descend."""
        RankOperation.set_llm(mock_llm_client)
        op = RankOperation(
            user_instruction="The course requires the least math",
            input_columns=["Courses"],
            descend=True
        )
        
        kwargs = op.op_kwargs
        assert "input_columns" in kwargs
        assert "descend" in kwargs
        assert kwargs["input_columns"] == ["Courses"]
        assert kwargs["descend"] == True
    
    @pytest.mark.asyncio
    async def test_execute_with_empty_dataframe(self, mock_llm_client):
        """Test that execute returns empty results for empty dataframe."""
        RankOperation.set_llm(mock_llm_client)
        
        op = RankOperation(
            user_instruction="The course requires the least math",
            input_columns=["Courses"]
        )
        
        empty_df = pd.DataFrame({"Courses": []})
        result = await op.execute(input_data=empty_df)
        
        assert isinstance(result, RankOpOutputs)
        assert result.ranked_indices == []
        assert result.cost == 0.0
    
    @pytest.mark.asyncio
    async def test_execute_basic_ranking(self, sample_dataframe, mock_llm_client):
        """Test execute with basic ranking."""
        RankOperation.set_llm(mock_llm_client)
        
        op = RankOperation(
            user_instruction="The course requires the least math",
            input_columns=["Courses"],
            descend=True
        )
        cache = {
            (0, 1): "2", (0, 2): "2", (0, 3): "1", (1, 2): "1", (1, 3): "1", (2, 3): "1",
            (1, 0): "1", (2, 0): "1", (3, 0): "2", (2, 1): "2", (3, 1): "2", (3, 2): "2"
        }
        result = await op.execute(input_data=sample_dataframe, cache=cache)
        
        assert isinstance(result, RankOpOutputs)
        assert result.ranked_indices == [1, 2, 0, 3]
        assert result.cost == 0.0
    
    @pytest.mark.asyncio
    async def test_execute_by_ascending_order(self, sample_dataframe, mock_llm_client):
        """Test rank by ascending order."""
        RankOperation.set_llm(mock_llm_client)
        
        op = RankOperation(
            user_instruction="The course requires the least math",
            input_columns=["Courses"],
            descend=False
        )
        cache = {
            (0, 1): "2", (0, 2): "2", (0, 3): "1", (1, 2): "1", (1, 3): "1", (2, 3): "1",
            (1, 0): "1", (2, 0): "1", (3, 0): "2", (2, 1): "2", (3, 1): "2", (3, 2): "2"
        }
        result = await op.execute(input_data=sample_dataframe, cache=cache)
        
        assert isinstance(result, RankOpOutputs)
        assert result.ranked_indices == [3, 0, 2, 1]
        assert result.cost == 0.0
    
    @pytest.mark.asyncio
    async def test_execute_with_udf_warning(self, sample_dataframe, mock_llm_client):
        """Test that UDF triggers a warning."""
        RankOperation.set_llm(mock_llm_client)
        
        def compare_func(item1, item2):
            """Function that compares two items."""
            return "1" if len(item1) > len(item2) else "2"
        
        tool = FunctionCallTool.from_function(func=compare_func)
        
        op = RankOperation(
            user_instruction="The course requires the least math",
            input_columns=["Courses"],
            tool=tool
        )
        cache = {
            (0, 1): "2", (0, 2): "2", (0, 3): "1", (1, 2): "1", (1, 3): "1", (2, 3): "1",
            (1, 0): "1", (2, 0): "1", (3, 0): "2", (2, 1): "2", (3, 1): "2", (3, 2): "2"
        }
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = await op.execute(input_data=sample_dataframe, cache=cache)
            
            # Check that warning was issued
            assert len(w) > 0
            assert "udf is not supported" in str(w[0].message).lower()
            # Should still return a valid result (falls back to LLM)
            assert isinstance(result, RankOpOutputs)
    
    @pytest.mark.asyncio
    async def test_execute_missing_user_instruction_and_func(self, sample_dataframe, mock_llm_client):
        """Test that execute raises ValueError when both user_instruction and func are missing."""
        RankOperation.set_llm(mock_llm_client)
        
        op = RankOperation(
            user_instruction=None,
            input_columns=["Courses"]
        )
        
        with pytest.raises(ValueError, match="Neither `user_instruction` nor `func` is given"):
            await op.execute(input_data=sample_dataframe)


class TestRankOpOutputs:
    """Test suite for RankOpOutputs class."""
    
    def test_rank_op_outputs_initialization(self):
        """Test that RankOpOutputs initializes correctly."""
        outputs = RankOpOutputs(
            ranking=[1, 2],
            ranked_indices=[0, 1],
            cost=0.05
        )
        
        assert outputs.ranking == [1, 2]
        assert outputs.ranked_indices == [0, 1]
        assert outputs.cost == 0.05
    
    def test_rank_op_outputs_default_values(self):
        """Test that RankOpOutputs has correct default values."""
        outputs = RankOpOutputs()
        
        assert outputs.ranking == []
        assert outputs.ranked_indices == []
        assert outputs.cost == 0.0


class TestRankWrapper:
    """Test suite for rank_wrapper function."""
    
    def test_rank_wrapper_basic(self, sample_dataframe, mock_llm_client):
        """Test the rank_wrapper function with basic usage."""
        RankOperation.set_llm(mock_llm_client)
        cache = {
            (0, 1): "2", (0, 2): "2", (0, 3): "1", (1, 2): "1", (1, 3): "1", (2, 3): "1",
            (1, 0): "1", (2, 0): "1", (3, 0): "2", (2, 1): "2", (3, 1): "2", (3, 2): "2"
        }
        result = nv.ops.rank(
            sample_dataframe,
            "The course requires the least math",
            input_column="Courses",
            descend=True,
            cache=cache
        )
        
        assert isinstance(result, RankOpOutputs)
        assert result.ranked_indices == [1, 2, 0, 3]
        assert result.ranking == [3, 1, 2, 4]
        assert result.cost == 0.0
    
    def test_rank_wrapper_by_ascending_order(self, sample_dataframe, mock_llm_client):
        """Test rank_wrapper by ascending order."""
        RankOperation.set_llm(mock_llm_client)
        cache = {
            (0, 1): "2", (0, 2): "2", (0, 3): "1", (1, 2): "1", (1, 3): "1", (2, 3): "1",
            (1, 0): "1", (2, 0): "1", (3, 0): "2", (2, 1): "2", (3, 1): "2", (3, 2): "2"
        }
        result = nv.ops.rank(
            sample_dataframe,
            "The course requires the least math",
            input_column="Courses",
            descend=False,
            cache=cache
        )
        
        assert isinstance(result, RankOpOutputs)
        assert result.ranked_indices == [3, 0, 2, 1]
        assert result.ranking == [2, 4, 3, 1]
        assert result.cost == 0.0


class TestRankOperationIntegration:
    """Integration tests for rank operation with real-like scenarios."""
    
    @pytest.mark.asyncio
    async def test_compare_by_plain_llm(self, mock_llm_client):
        """Test that _compare_by_plain_llm returns correct format."""
        RankOperation.set_llm(mock_llm_client)
        
        op = RankOperation(
            user_instruction="The course requires the least math",
            input_columns=["Courses"]
        )
        
        data1 = "Probability and Random Processes"
        data2 = "Convex Optimization"
        
        output, cost = await op._compare_by_plain_llm(
            data1, data2, "The course requires the least math", "str"
        )
        
        # Output should be "1" or "2"
        assert output in ["1", "2"]
        assert cost > 0
    
    @pytest.mark.asyncio
    async def test_partition_function(self, sample_dataframe, mock_llm_client):
        """Test that partition function works correctly."""
        RankOperation.set_llm(mock_llm_client)
        cache = {
            (0, 1): "2", (0, 2): "2", (0, 3): "1", (1, 2): "1", (1, 3): "1", (2, 3): "1",
            (1, 0): "1", (2, 0): "1", (3, 0): "2", (2, 1): "2", (3, 1): "2", (3, 2): "2"
        }
        
        op = RankOperation(
            user_instruction="The course requires the least math",
            input_columns=["Courses"]
        )
        
        data = sample_dataframe["Courses"]
        ranking = list(range(len(data)))
        low, high = 0, len(data) - 1
        partition_pos, cost = await op.partition(
            data, ranking, low, high, "The course requires the least math", "str", cache=cache
        )
        
        assert isinstance(partition_pos, int)
        assert partition_pos == 0
        assert cost == 0.0
    
    @pytest.mark.asyncio
    async def test_quick_sort_function(self, sample_dataframe, mock_llm_client):
        """Test that quick_sort function works correctly."""
        RankOperation.set_llm(mock_llm_client)
        cache = {
            (0, 1): "2", (0, 2): "2", (0, 3): "1", (1, 2): "1", (1, 3): "1", (2, 3): "1",
            (1, 0): "1", (2, 0): "1", (3, 0): "2", (2, 1): "2", (3, 1): "2", (3, 2): "2"
        }
        
        op = RankOperation(
            user_instruction="The course requires the least math",
            input_columns=["Courses"]
        )
        
        data = sample_dataframe["Courses"]
        ranking = list(range(len(data)))
        low, high = 0, len(data) - 1
        sorted_ranking, cost = await op.quick_sort(
            data, ranking, low, high, "The course requires the least math", "str", cache=cache
        )
        
        assert sorted_ranking == [1, 2, 0, 3]
        assert cost == 0.0
