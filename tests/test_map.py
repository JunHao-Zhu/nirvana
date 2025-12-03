import pytest
import pandas as pd
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio

import nirvana as nv
from nirvana.ops.map import MapOperation, MapOpOutputs
from nirvana.executors.llm_backbone import LLMClient
from nirvana.executors.tools import FunctionCallTool


@pytest.fixture
def sample_dataframe():
    """Create a sample dataframe for testing."""
    return pd.DataFrame({
        "title": ["The Godfather", "The Dark Knight", "Inception"],
        "overview": [
            "An organized crime dynasty's aging patriarch transfers control of his clandestine empire to his reluctant son.",
            "When the menace known as the Joker wreaks havoc and chaos on the people of Gotham, Batman must accept one of the greatest psychological and physical tests of his ability to fight injustice.",
            "A thief who steals corporate secrets through the use of dream-sharing technology is given the inverse task of planting an idea into the mind of a C.E.O."
        ]
    })


@pytest.fixture
def empty_dataframe():
    """Create an empty dataframe for testing."""
    return pd.DataFrame({"title": [], "overview": []})


@pytest.fixture
def dataframe_with_nan():
    """Create a dataframe with NaN values for testing."""
    return pd.DataFrame({
        "title": ["The Godfather", None, "Inception"],
        "overview": [
            "An organized crime dynasty's aging patriarch transfers control of his clandestine empire to his reluctant son.",
            None,
            "A thief who steals corporate secrets through the use of dream-sharing technology is given the inverse task of planting an idea into the mind of a C.E.O."
        ]
    })


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client for testing."""
    mock_client = MagicMock(spec=LLMClient)
    mock_client.default_model = "gpt-4"
    
    async def mock_call(messages, parse_tags=False, parse_code=False, **kwargs):
        """Mock LLM call that returns structured output."""
        result = {
            "raw_output": "<genre>crime, drama</genre>",
            "cost": 0.01
        }
        
        if parse_tags:
            tags = kwargs.get("tags", [])
            # When parse_tags is True, the LLM client extracts XML tags
            # and puts them in the result dict with tag names as keys
            for tag in tags:
                if tag == "genre":
                    result[tag] = "crime, drama"
                elif tag == "evaluation":
                    result[tag] = "PASS"
                elif tag == "feedback":
                    result[tag] = "No feedback needed"
        
        return result
    
    mock_client.__call__ = AsyncMock(side_effect=mock_call)
    return mock_client


class TestMapOperation:
    """Test suite for MapOperation class."""
    
    def test_map_operation_initialization(self, mock_llm_client):
        """Test that MapOperation initializes correctly."""
        MapOperation.set_llm(mock_llm_client)
        op = MapOperation(
            user_instruction="Extract genre from the movie overview",
            input_columns=["overview"],
            output_columns=["genre"],
            implementation="plain"
        )
        
        assert op.op_name == "map"
        assert op.dependencies == ["overview"]
        assert op.generated_fields == ["genre"]
    
    def test_op_kwargs_property(self, mock_llm_client):
        """Test that op_kwargs includes input and output columns."""
        MapOperation.set_llm(mock_llm_client)
        op = MapOperation(
            user_instruction="Extract genre from the movie overview",
            input_columns=["overview"],
            output_columns=["genre"],
            implementation="plain"
        )
        
        kwargs = op.op_kwargs
        assert "input_columns" in kwargs
        assert "output_columns" in kwargs
        assert kwargs["input_columns"] == ["overview"]
        assert kwargs["output_columns"] == ["genre"]
    
    @pytest.mark.asyncio
    async def test_execute_with_empty_dataframe(self, mock_llm_client):
        """Test that execute returns None output for empty dataframe."""
        MapOperation.set_llm(mock_llm_client)
        
        op = MapOperation(
            user_instruction="Extract genre from the movie overview",
            input_columns=["overview"],
            output_columns=["genre"]
        )
        
        empty_df = pd.DataFrame({"overview": []})
        result = await op.execute(input_data=empty_df)
        
        assert result.field_name == ["genre"]
        assert result.output == []
        assert result.cost == 0.0
    
    @pytest.mark.asyncio
    async def test_execute_with_plain_strategy(self, sample_dataframe, mock_llm_client):
        """Test execute with plain LLM strategy."""
        MapOperation.set_llm(mock_llm_client)
        
        op = MapOperation(
            user_instruction="Extract the genre of each movie from the movie overview.",
            input_columns=["overview"],
            output_columns=["genre"],
            implementation="plain"
        )
        
        result = await op.execute(input_data=sample_dataframe)
        
        assert result.field_name == ["genre"]
        assert isinstance(result.output, dict)
        assert "genre" in result.output
        assert len(result.output["genre"]) == 3
        assert result.cost > 0
    
    @pytest.mark.asyncio
    async def test_execute_with_nan_values(self, dataframe_with_nan, mock_llm_client):
        """Test that execute handles NaN values correctly."""
        MapOperation.set_llm(mock_llm_client)
        
        op = MapOperation(
            user_instruction="Extract the genre of each movie.",
            input_columns=["overview"],
            output_columns=["genre"],
            implementation="plain"
        )
        
        result = await op.execute(input_data=dataframe_with_nan)
        
        assert result.field_name == ["genre"]
        assert isinstance(result.output, dict)
        assert "genre" in result.output
        assert len(result.output["genre"]) == 3
        # NaN values should be None in the output
        assert None in result.output["genre"]
    
    @pytest.mark.asyncio
    async def test_execute_with_self_refine_strategy(self, sample_dataframe, mock_llm_client):
        """Test execute with self-refine strategy."""
        # Mock the evaluation response
        async def mock_call_with_eval(messages, parse_tags=False, **kwargs):
            result = {
                "raw_output": "<genre>crime, drama</genre>",
                "cost": 0.01
            }
            
            if parse_tags:
                tags = kwargs.get("tags", [])
                for tag in tags:
                    if tag == "evaluation":
                        result[tag] = "PASS"
                    elif tag == "feedback":
                        result[tag] = "No feedback needed"
                    elif tag == "genre":
                        result[tag] = "crime, drama"
            
            return result
        
        mock_llm_client.__call__ = AsyncMock(side_effect=mock_call_with_eval)
        MapOperation.set_llm(mock_llm_client)
        
        op = MapOperation(
            user_instruction="Extract the genre of each movie.",
            input_columns=["overview"],
            output_columns=["genre"],
            implementation="self_refine"
        )
        
        result = await op.execute(input_data=sample_dataframe)
        
        assert result.field_name == ["genre"]
        assert isinstance(result.output, dict)
        assert "genre" in result.output
        assert len(result.output["genre"]) == 3
    
    @pytest.mark.asyncio
    async def test_execute_with_fewshot_strategy(self, sample_dataframe, mock_llm_client):
        """Test execute with fewshot strategy."""
        MapOperation.set_llm(mock_llm_client)
        
        demos = [
            {"data": "overview: A crime story", "answer": "<genre>crime</genre>"},
            {"data": "overview: A comedy film", "answer": "<genre>comedy</genre>"}
        ]
        
        op = MapOperation(
            user_instruction="Extract the genre of each movie.",
            input_columns=["overview"],
            output_columns=["genre"],
            implementation="fewshot",
            context=demos
        )
        
        result = await op.execute(input_data=sample_dataframe)
        
        assert result.field_name == ["genre"]
        assert isinstance(result.output, dict)
        assert "genre" in result.output
        assert len(result.output["genre"]) == 3
    
    @pytest.mark.asyncio
    async def test_execute_with_udf(self, sample_dataframe, mock_llm_client):
        """Test execute with user-defined function."""
        MapOperation.set_llm(mock_llm_client)
        
        def extract_genre(text):
            """Simple function to extract genre."""
            if "crime" in text.lower():
                return "crime"
            elif "action" in text.lower():
                return "action"
            return "unknown"
        
        tool = FunctionCallTool.from_function(func=extract_genre)
        
        op = MapOperation(
            user_instruction="Extract the genre of each movie.",
            input_columns=["overview"],
            output_columns=["genre"],
            tool=tool
        )
        
        result = await op.execute(input_data=sample_dataframe)
        
        assert result.field_name == ["genre"]
        assert isinstance(result.output, dict)
        assert "genre" in result.output
        assert len(result.output["genre"]) == 3
        # UDF should return the function results directly
        assert "crime" in result.output["genre"]
    
    @pytest.mark.asyncio
    async def test_execute_with_udf_exception_fallback(self, sample_dataframe, mock_llm_client):
        """Test that UDF exceptions fall back to LLM."""
        MapOperation.set_llm(mock_llm_client)
        
        def failing_func(text):
            """Function that raises an exception."""
            raise ValueError("Function failed")
        
        tool = FunctionCallTool.from_function(func=failing_func)
        
        op = MapOperation(
            user_instruction="Extract the genre of each movie.",
            input_columns=["overview"],
            output_columns=["genre"],
            tool=tool
        )
        
        result = await op.execute(input_data=sample_dataframe)
        
        # Should fall back to LLM, so result should still be valid
        assert result.field_name == ["genre"]
        assert isinstance(result.output, dict)
        assert "genre" in result.output
        assert len(result.output["genre"]) == 3
    
    @pytest.mark.asyncio
    async def test_execute_missing_user_instruction_and_func(self, sample_dataframe, mock_llm_client):
        """Test that execute raises ValueError when both user_instruction and func are missing."""
        MapOperation.set_llm(mock_llm_client)
        
        op = MapOperation(
            user_instruction=None,
            input_columns=["overview"],
            output_columns=["genre"]
        )
        
        with pytest.raises(ValueError, match="Neither `user_instruction` nor `func` is given."):
            await op.execute(input_data=sample_dataframe)
    
    @pytest.mark.asyncio
    async def test_execute_unsupported_strategy(self, sample_dataframe, mock_llm_client):
        """Test that execute raises NotImplementedError for unsupported strategy."""
        MapOperation.set_llm(mock_llm_client)
        
        op = MapOperation(
            user_instruction="Extract genre",
            input_columns=["overview"],
            output_columns=["genre"],
            implementation="unsupported_strategy"
        )
        
        with pytest.raises(NotImplementedError, match="Strategy unsupported_strategy is not implemented."):
            await op.execute(input_data=sample_dataframe)
    
    @pytest.mark.asyncio
    async def test_execute_fewshot_without_context(self, sample_dataframe, mock_llm_client):
        """Test that execute raises AssertionError for fewshot without context."""
        MapOperation.set_llm(mock_llm_client)
        
        op = MapOperation(
            user_instruction="Extract genre",
            input_columns=["overview"],
            output_columns=["genre"],
            implementation="fewshot"
        )
        
        with pytest.raises(AssertionError, match="Few-shot examples must be provided"):
            await op.execute(input_data=sample_dataframe)


class TestMapOpOutputs:
    """Test suite for MapOpOutputs class."""
    
    def test_map_op_outputs_initialization(self):
        """Test that MapOpOutputs initializes correctly with list field_name."""
        outputs = MapOpOutputs(
            field_name=["genre", "rating"],
            output={"genre": ["crime", "drama"], "rating": [8.5, 9.0]},
            cost=0.05
        )
        
        assert outputs.field_name == ["genre", "rating"]
        assert outputs.output == {"genre": ["crime", "drama"], "rating": [8.5, 9.0]}
        assert outputs.cost == 0.05
    
    def test_map_op_outputs_default_values(self):
        """Test that MapOpOutputs has correct default values."""
        outputs = MapOpOutputs()
        
        assert outputs.field_name is None
        assert outputs.output == []
        assert outputs.cost == 0.0
    
    def test_map_op_outputs_addition_field_name(self):
        """Test that MapOpOutputs can be added together with list field_name."""
        outputs1 = MapOpOutputs(
            field_name=["genre", "rating"],
            output={"genre": ["crime"], "rating": [8.5]},
            cost=0.02
        )
        
        outputs2 = MapOpOutputs(
            field_name=["genre", "rating"],
            output={"genre": ["drama"], "rating": [9.0]},
            cost=0.03
        )
        
        combined = outputs1 + outputs2
        
        assert combined.field_name == ["genre", "rating"]
        assert combined.output == {"genre": ["crime", "drama"], "rating": [8.5, 9.0]}
        assert combined.cost == 0.05
    
    def test_map_op_outputs_addition_different_field_names(self):
        """Test that adding MapOpOutputs with different field names raises AssertionError."""
        outputs1 = MapOpOutputs(field_name=["genre"], output={"genre": ["crime"]}, cost=0.01)
        outputs2 = MapOpOutputs(field_name=["rating"], output={"rating": ["8.5"]}, cost=0.01)
        
        with pytest.raises(AssertionError, match="Cannot merge MapOpOutputs with different field names"):
            _ = outputs1 + outputs2


class TestMapWrapper:
    """Test suite for map_wrapper function."""
    
    @pytest.mark.asyncio
    async def test_map_wrapper_basic(self, sample_dataframe, mock_llm_client):
        """Test the map_wrapper function with basic usage."""
        MapOperation.set_llm(mock_llm_client)
        
        result = nv.ops.map(
            sample_dataframe,
            "Extract the genre of each movie.",
            input_column="overview",
            output_columns=["genre"],
            strategy="plain"
        )
        
        assert isinstance(result, MapOpOutputs)
        assert result.field_name == ["genre"]
        assert isinstance(result.output, dict)
        assert "genre" in result.output
        assert len(result.output["genre"]) == 3
    
    @pytest.mark.asyncio
    async def test_map_wrapper_with_udf(self, sample_dataframe, mock_llm_client):
        """Test the map_wrapper function with user-defined function."""
        MapOperation.set_llm(mock_llm_client)
        
        def extract_genre(text):
            if "crime" in text.lower():
                return "crime"
            return "other"
        
        result = nv.ops.map(
            sample_dataframe,
            input_column="overview",
            output_columns=["genre"],
            func=extract_genre
        )
        
        assert isinstance(result, MapOpOutputs)
        assert result.field_name == ["genre"]
        assert isinstance(result.output, dict)
        assert "genre" in result.output
        assert len(result.output["genre"]) == 3


class TestMapOperationIntegration:
    """Integration tests for map operation with real-like scenarios."""
    
    @pytest.mark.asyncio
    async def test_map_operation_with_string_dtype(self, mock_llm_client):
        """Test map operation with string data type."""
        MapOperation.set_llm(mock_llm_client)
        
        df = pd.DataFrame({
            "text": ["Hello world", "Python programming", "Data science"]
        })
        
        op = MapOperation(
            user_instruction="Extract the main topic.",
            input_columns=["text"],
            output_columns=["topic"],
            implementation="plain"
        )
        
        result = await op.execute(input_data=df)
        
        assert result.field_name == ["topic"]
        assert isinstance(result.output, dict)
        assert "topic" in result.output
        assert len(result.output["topic"]) == 3
    
    @pytest.mark.asyncio
    async def test_map_operation_postprocess_outputs(self, mock_llm_client):
        """Test that _postprocess_map_outputs handles None values correctly."""
        MapOperation.set_llm(mock_llm_client)
        
        op = MapOperation(
            user_instruction="Extract genre",
            input_columns=["overview"],
            output_columns=["genre"]
        )
        
        # Simulate results with None values and dict structure
        results = [
            {"genre": "crime", "cost": 0.01},
            None,  # None response
            {"genre": "drama", "cost": 0.01}
        ]
        outputs, costs = op._postprocess_map_outputs(results, ["genre"])
        
        assert isinstance(outputs, dict)
        assert "genre" in outputs
        assert len(outputs["genre"]) == 3
        assert None in outputs["genre"]
        assert costs == 0.02
    
    @pytest.mark.asyncio
    async def test_map_operation_multiple_output_columns(self, sample_dataframe, mock_llm_client):
        """Test map operation with multiple output columns."""
        # Mock LLM to return multiple fields
        async def mock_call_multi(messages, parse_tags=False, **kwargs):
            result = {
                "raw_output": "<genre>crime</genre><rating>9.2</rating>",
                "cost": 0.01
            }
            if parse_tags:
                tags = kwargs.get("tags", [])
                for tag in tags:
                    if tag == "genre":
                        result[tag] = "crime"
                    elif tag == "rating":
                        result[tag] = "9.2"
            return result
        
        mock_llm_client.__call__ = AsyncMock(side_effect=mock_call_multi)
        MapOperation.set_llm(mock_llm_client)
        
        op = MapOperation(
            user_instruction="Extract genre and rating.",
            input_columns=["overview"],
            output_columns=["genre", "rating"],
            implementation="plain"
        )
        
        result = await op.execute(input_data=sample_dataframe)
        
        assert result.field_name == ["genre", "rating"]
        assert isinstance(result.output, dict)
        assert "genre" in result.output
        assert "rating" in result.output
        assert len(result.output["genre"]) == 3
        assert len(result.output["rating"]) == 3
