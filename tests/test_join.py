import pytest
import pandas as pd
from unittest.mock import AsyncMock

import nirvana as nv
from nirvana.ops.join import JoinOperation, JoinOpOutputs
from nirvana.executors.llm_backbone import LLMClient
from nirvana.executors.tools import FunctionCallTool


@pytest.fixture
def left_dataframe():
    """Create a sample left dataframe for testing."""
    return pd.DataFrame(
        data={"symptom": ["headache", "cough", "fever"]},
        index=[1, 2, 3]
    )


@pytest.fixture
def right_dataframe():
    """Create a sample right dataframe for testing."""
    return pd.DataFrame(
        data={
            "medical_use": [
                "treat mild to moderate pain, painful menstruation, osteoarthritis, dental pain, headaches, and pain from kidney stones",
                "treat bronchospasm, as well as chronic obstructive pulmonary disease",
                "reduce fever and inflammation"
            ],
        },
        index=[1, 4, 7]
    )


@pytest.fixture
def left_empty_dataframe():
    """Create an empty left dataframe for testing."""
    return pd.DataFrame({"symptom": []})


@pytest.fixture
def right_empty_dataframe():
    """Create an empty right dataframe for testing."""
    return pd.DataFrame({"medical_use": []})


@pytest.fixture
def dataframe_with_nan():
    """Create a dataframe with NaN values for testing."""
    return pd.DataFrame(
        data={"symptom": [None, "cough", "fever"]},
        index=[1, 2, 3]
    )


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client for testing."""
    
    async def mock_call(messages, parse_tags=False, parse_code=False, **kwargs):
        """Mock LLM call that returns structured output for join operations."""
        result = {
            "raw_output": "<output>False</output>",
            "cost": 0.01
        }
        
        if parse_tags:
            tags = kwargs.get("tags", [])
            # When parse_tags is True, the LLM client extracts XML tags
            for tag in tags:
                if tag == "output":
                    # Return False for join operations by default
                    result[tag] = "False"
        return result
    
    # Create a proper async mock
    mock_client = AsyncMock(spec=LLMClient, side_effect=mock_call)
    mock_client.default_model = "gpt-4"
    return mock_client


@pytest.fixture
def mock_block_join_llm_client():
    """ Mock LLM to return batch join results."""
    async def mock_call_block(messages, parse_tags=False, **kwargs):
        result = {
            "raw_output": "<output></output>",
            "cost": 0.01
        }
        if parse_tags:
            tags = kwargs.get("tags", [])
            for tag in tags:
                if tag == "output":
                    result[tag] = ""
        return result
    
    # Create a proper async mock
    mock_client = AsyncMock(spec=LLMClient, side_effect=mock_call_block)
    mock_client.default_model = "gpt-4"
    return mock_client


class TestJoinOperation:
    """Test suite for JoinOperation class."""
    
    def test_join_operation_initialization(self, mock_llm_client):
        """Test that JoinOperation initializes correctly."""
        JoinOperation.set_llm(mock_llm_client)
        op = JoinOperation(
            user_instruction="Does the drug cure the symptom?",
            left_on=["symptom"],
            right_on=["medical_use"],
            how="inner",
            strategy="nest"
        )
        
        assert op.op_name == "join"
        assert op.left_on == ["symptom"]
        assert op.right_on == ["medical_use"]
        assert op.how == "inner"
        assert op.dependencies == ["symptom", "medical_use"]
        assert op.generated_fields == []
    
    def test_op_kwargs_property(self, mock_llm_client):
        """Test that op_kwargs includes left_on, right_on, and how."""
        JoinOperation.set_llm(mock_llm_client)
        op = JoinOperation(
            user_instruction="Does the drug cure the symptom?",
            left_on=["symptom"],
            right_on=["medical_use"],
            how="inner"
        )
        
        kwargs = op.op_kwargs
        assert "left_on" in kwargs
        assert "right_on" in kwargs
        assert "how" in kwargs
        assert kwargs["left_on"] == ["symptom"]
        assert kwargs["right_on"] == ["medical_use"]
        assert kwargs["how"] == "inner"
    
    @pytest.mark.asyncio
    async def test_execute_with_empty_dataframes(self, left_empty_dataframe, right_empty_dataframe, mock_llm_client):
        """Test that execute returns empty results for empty dataframes."""
        JoinOperation.set_llm(mock_llm_client)
        
        op = JoinOperation(
            user_instruction="Does the drug cure the symptom?",
            left_on=["symptom"],
            right_on=["medical_use"],
            how="inner"
        )
        
        result = await op.execute(left_data=left_empty_dataframe, right_data=right_empty_dataframe)
        
        assert isinstance(result, JoinOpOutputs)
        # JoinOpOutputs inherits 'output' from BaseOpOutputs
        # The code returns output=[] for empty case
        assert result.output == []
        assert result.left_join_keys == []
        assert result.right_join_keys == []
        assert result.cost == 0.0
    
    @pytest.mark.asyncio
    async def test_execute_with_nest_strategy(self, left_dataframe, right_dataframe, mock_llm_client):
        """Test execute with nested join strategy."""
        JoinOperation.set_llm(mock_llm_client)
        
        op = JoinOperation(
            user_instruction="Does the drug cure the symptom?",
            left_on=["symptom"],
            right_on=["medical_use"],
            how="inner",
            strategy="nest"
        )
        cache = {(1, 1): "True", (3, 7): "True"}
        result = await op.execute(left_data=left_dataframe, right_data=right_dataframe, cache=cache)
        
        assert isinstance(result, JoinOpOutputs)
        # The code returns output=joined_pairs, so result.output contains the joined pairs
        assert result.output == [(1, 1), (3, 7)]
        assert result.left_join_keys == [1, 3]
        assert result.right_join_keys == [1, 3]
        assert result.cost == 0.07
    
    @pytest.mark.asyncio
    async def test_execute_with_block_strategy(self, left_dataframe, right_dataframe, mock_block_join_llm_client):
        """Test execute with block join strategy."""
        JoinOperation.set_llm(mock_block_join_llm_client)
        
        op = JoinOperation(
            user_instruction="Does the drug cure the symptom?",
            left_on=["symptom"],
            right_on=["medical_use"],
            how="inner",
            strategy="block",
        )
        cache = {(0, 0): "L0-R0", (1, 1): "L0-R0"}
        result = await op.execute(left_data=left_dataframe, right_data=right_dataframe, batch_size=2, cache=cache)
        
        assert result.output == [(1, 1), (3, 7)]
        assert result.left_join_keys == [1, 3]
        assert result.right_join_keys == [1, 3]
        assert result.cost == 0.02
    
    @pytest.mark.asyncio
    async def test_execute_with_nan_values(self, dataframe_with_nan, right_dataframe, mock_llm_client):
        """Test that execute handles NaN values correctly."""
        JoinOperation.set_llm(mock_llm_client)
        
        op = JoinOperation(
            user_instruction="Does the drug cure the symptom?",
            left_on=["symptom"],
            right_on=["medical_use"],
            how="inner",
            strategy="nest"
        )
        cache = {(3, 7): "True"}
        result = await op.execute(left_data=dataframe_with_nan, right_data=right_dataframe, cache=cache)
        
        assert result.output == [(3, 7)]
        assert result.left_join_keys == [3]
        assert result.right_join_keys == [3]
    
    @pytest.mark.asyncio
    async def test_execute_with_udf(self, left_dataframe, right_dataframe, mock_llm_client):
        """Test execute with user-defined function."""
        JoinOperation.set_llm(mock_llm_client)
        
        def check_match(symptom, medical_use):
            """Simple function to check if symptom matches medical use."""
            if symptom is None or medical_use is None:
                return False
            return symptom.lower() in medical_use.lower()
        
        tool = FunctionCallTool.from_function(func=check_match)
        op = JoinOperation(
            user_instruction="Does the drug cure the symptom?",
            left_on=["symptom"],
            right_on=["medical_use"],
            how="inner",
            strategy="nest",
            tool=tool
        )
        result = await op.execute(left_data=left_dataframe, right_data=right_dataframe)
        
        assert result.output == [(1, 1), (3, 7)]
        assert result.left_join_keys == [1, 3]
        assert result.right_join_keys == [1, 3]
        assert result.cost == 0.0
    
    @pytest.mark.asyncio
    async def test_execute_with_udf_exception_fallback(self, left_dataframe, right_dataframe, mock_llm_client):
        """Test that UDF exceptions fall back to LLM."""
        JoinOperation.set_llm(mock_llm_client)
        
        def failing_func(left, right):
            """Function that raises an exception."""
            raise ValueError("Function failed")
        
        tool = FunctionCallTool.from_function(func=failing_func)
        op = JoinOperation(
            user_instruction="Does the drug cure the symptom?",
            left_on=["symptom"],
            right_on=["medical_use"],
            how="inner",
            strategy="nest",
            tool=tool
        )
        
        cache = {(1, 1): "True", (3, 7): "True"}
        result = await op.execute(left_data=left_dataframe, right_data=right_dataframe, cache=cache)
        
        assert isinstance(result, JoinOpOutputs)
        # The code returns output=joined_pairs, so result.output contains the joined pairs
        assert result.output == [(1, 1), (3, 7)]
        assert result.left_join_keys == [1, 3]
        assert result.right_join_keys == [1, 3]
        assert result.cost == 0.07
    
    @pytest.mark.asyncio
    async def test_execute_with_different_join_types(self, left_dataframe, right_dataframe, mock_llm_client):
        """Test execute with different join types (inner, left, right)."""
        JoinOperation.set_llm(mock_llm_client)
        
        for join_type in ["inner", "left", "right"]:
            op = JoinOperation(
                user_instruction="Does the drug cure the symptom?",
                left_on=["symptom"],
                right_on=["medical_use"],
                how=join_type,
                strategy="nest"
            )
            cache = {(1, 1): "True", (3, 7): "True"}
            result = await op.execute(left_data=left_dataframe, right_data=right_dataframe, cache=cache)
            
            if join_type in ["inner", "left"]:
                assert result.output == [(1, 1), (3, 7)]
                assert result.left_join_keys == [1, 3]
                assert result.right_join_keys == [1, 3]
            else:
                assert result.output == [(1, 1), (3, 7)]
                assert result.left_join_keys == [1, 7]
                assert result.right_join_keys == [1, 7]
    
    @pytest.mark.asyncio
    async def test_execute_unsupported_strategy(self, left_dataframe, right_dataframe, mock_llm_client):
        """Test that execute raises ValueError for unsupported strategy."""
        JoinOperation.set_llm(mock_llm_client)
        
        op = JoinOperation(
            user_instruction="Does the drug cure the symptom?",
            left_on=["symptom"],
            right_on=["medical_use"],
            how="inner",
            strategy="unsupported"
        )
        with pytest.raises(ValueError, match="Strategy unsupported is not supported"):
            await op.execute(left_data=left_dataframe, right_data=right_dataframe)
    
    @pytest.mark.asyncio
    async def test_block_join_with_udf_warning(self, left_dataframe, right_dataframe, mock_block_join_llm_client):
        """Test that block join warns when UDF is provided."""
        import warnings
        
        JoinOperation.set_llm(mock_block_join_llm_client)
        
        def check_match(symptom, medical_use):
            return True
        
        tool = FunctionCallTool.from_function(func=check_match)
        
        op = JoinOperation(
            user_instruction="Does the drug cure the symptom?",
            left_on=["symptom"],
            right_on=["medical_use"],
            how="inner",
            strategy="block",
            tool=tool
        )
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = await op.execute(left_data=left_dataframe, right_data=right_dataframe)
            
            # Check that warning was issued
            assert len(w) > 0
            assert "block semantic join does not support user-defined functions" in str(w[0].message).lower()


class TestJoinOpOutputs:
    """Test suite for JoinOpOutputs class."""
    
    def test_join_op_outputs_initialization(self):
        """Test that JoinOpOutputs initializes correctly."""
        outputs = JoinOpOutputs(
            joined_pairs=[(0, 1), (1, 2)],
            left_join_keys=[0, 1],
            right_join_keys=[1, 2],
            cost=0.05
        )
        
        assert outputs.joined_pairs == [(0, 1), (1, 2)]
        assert outputs.left_join_keys == [0, 1]
        assert outputs.right_join_keys == [1, 2]
        assert outputs.cost == 0.05
    
    def test_join_op_outputs_default_values(self):
        """Test that JoinOpOutputs has correct default values."""
        outputs = JoinOpOutputs()
        
        assert outputs.joined_pairs == []
        assert outputs.left_join_keys == []
        assert outputs.right_join_keys == []
        assert outputs.cost == 0.0


class TestJoinWrapper:
    """Test suite for join_wrapper function."""
    
    def test_join_wrapper_basic(self, left_dataframe, right_dataframe, mock_llm_client):
        """Test the join_wrapper function with basic usage."""
        JoinOperation.set_llm(mock_llm_client)
        
        cache = {(1, 1): "True", (3, 7): "True"}
        result = nv.ops.join(
            left_data=left_dataframe,
            right_data=right_dataframe,
            user_instruction="Does the drug cure the symptom?",
            left_on="symptom",
            right_on="medical_use",
            how="inner",
            strategy="nest",
            cache=cache
        )
        
        assert isinstance(result, JoinOpOutputs)
        # The code returns output=joined_pairs, so result.output contains the joined pairsassert result.output == [(1, 1), (3, 7)]
        assert result.output == [(1, 1), (3, 7)]
        assert result.left_join_keys == [1, 3]
        assert result.right_join_keys == [1, 3]
        assert result.cost == 0.07
    
    def test_join_wrapper_with_different_join_types(self, left_dataframe, right_dataframe, mock_llm_client):
        """Test join_wrapper with different join types."""
        JoinOperation.set_llm(mock_llm_client)
        
        cache = {(1, 1): "True", (3, 7): "True"}
        for join_type in ["inner", "left", "right"]:
            result = nv.ops.join(
                left_data=left_dataframe,
                right_data=right_dataframe,
                user_instruction="Does the drug cure the symptom?",
                left_on="symptom",
                right_on="medical_use",
                how=join_type,
                strategy="nest",
                cache=cache
            )

            if join_type in ["inner", "left"]:
                assert result.output == [(1, 1), (3, 7)]
                assert result.left_join_keys == [1, 3]
                assert result.right_join_keys == [1, 3]
            else:
                assert result.output == [(1, 1), (3, 7)]
                assert result.left_join_keys == [1, 7]
                assert result.right_join_keys == [1, 7]


class TestJoinOperationIntegration:
    """Integration tests for join operation with real-like scenarios."""
    
    def test_nested_join_postprocess_outputs(self, mock_llm_client):
        """Test that _postprocess_nested_join_outputs handles different output formats."""
        JoinOperation.set_llm(mock_llm_client)
        
        op = JoinOperation(
            user_instruction="Does the drug cure the symptom?",
            left_on=["symptom"],
            right_on=["medical_use"],
            how="inner",
            strategy="nest"
        )
        
        # Simulate results with different output formats
        data_id_pairs = [(0, 0), (0, 2), (1, 0), (1, 2)]
        results = [
            {"output": "True", "cost": 0.01},
            {"output": "False", "cost": 0.01},
            {"output": True, "cost": 0.01},
            {"output": "", "cost": 0.01}
        ]
        joined_pairs, left_keys, right_keys, cost = op._postprocess_nested_join_outputs(
            data_id_pairs, results, "inner"
        )
        
        assert joined_pairs == [(0, 0), (1, 0)]
        assert left_keys == [0, 1]
        assert right_keys == [0, 1]
        assert cost == 0.04
    
    def test_block_join_postprocess_outputs(self, mock_block_join_llm_client):
        """Test that _postprocess_block_join_outputs handles batch results."""
        JoinOperation.set_llm(mock_block_join_llm_client)
        
        op = JoinOperation(
            user_instruction="Does the drug cure the symptom?",
            left_on=["symptom"],
            right_on=["medical_use"],
            how="inner",
            strategy="block"
        )
        
        # Simulate batch results
        results = [
            {"output": "L0-R0,L1-R1", "cost": 0.01},
            {"output": "", "cost": 0.01},
            {"output": "", "cost": 0.01},
            {"output": "L0-R0", "cost": 0.01}
        ]
        left_keys_in_batches = [[0, 1], [3]]
        right_keys_in_batches = [[0, 1], [2]]
        batch_ids_pairs = [(0, 0), (0, 1), (1, 0), (1, 1)]
        joined_pairs, left_keys, right_keys, cost = op._postprocess_block_join_outputs(
            results, "inner", batch_ids_pairs, left_keys_in_batches, right_keys_in_batches
        )
        
        assert joined_pairs == [(0, 0), (1, 1), (3, 2)]
        assert left_keys == [0, 1, 3]
        assert right_keys == [0, 1, 3]
        assert cost == 0.04
    
    def test_prepare_nested_join_pairs(self, mock_llm_client):
        """Test that _prepare_nested_join_pairs creates correct pairs."""
        JoinOperation.set_llm(mock_llm_client)
        
        op = JoinOperation(
            user_instruction="Does the drug cure the symptom?",
            left_on=["symptom"],
            right_on=["medical_use"],
            how="inner"
        )
        
        left_values = pd.Series(["a", "b"], index=[0, 1])
        right_values = pd.Series(["x", "y"], index=[2, 3])
        
        data_id_pairs = op._prepare_nested_join_pairs(left_values, right_values)
        
        assert len(data_id_pairs) == 4  # 2x2 = 4 pairs
        assert data_id_pairs == [(0, 2), (0, 3), (1, 2), (1, 3)]
    
    def test_prepare_join_batches(self, mock_block_join_llm_client):
        """Test that _prepare_join_batches creates correct batches."""
        JoinOperation.set_llm(mock_block_join_llm_client)
        
        op = JoinOperation(
            user_instruction="Does the drug cure the symptom?",
            left_on=["symptom"],
            right_on=["medical_use"],
            how="inner",
            strategy="block"
        )
        left_values = pd.Series(["a", "b", "c", "d"], index=[0, 1, 2, 3])
        right_values = pd.Series(["x", "y"], index=[4, 5])
        left_batches, left_keys, right_batches, right_keys = op._prepare_join_batches(
            left_values, right_values, batch_size=2
        )
        
        assert left_batches == [["a", "b"], ["c", "d"]]  # 4 items / 2 batch_size = 2 batches
        assert right_batches == [["x", "y"]]  # 2 items / 2 batch_size = 1 batch
        assert left_keys == [[0, 1], [2, 3]]
        assert right_keys == [[4, 5]]
