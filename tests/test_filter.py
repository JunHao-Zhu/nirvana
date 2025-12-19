import pytest
import pandas as pd
from unittest.mock import AsyncMock

import nirvana as nv
from nirvana.ops.filter import FilterOperation, FilterOpOutputs
from nirvana.executors.llm_backbone import LLMClient
from nirvana.executors.tools import FunctionCallTool


@pytest.fixture
def sample_dataframe():
    """Create a sample dataframe for testing."""
    return pd.DataFrame({
        "Courses": [
            "Probability and Random Processes",
            "History",
            "Database",
            "Convex Optimization",
        ]
    })


@pytest.fixture
def empty_dataframe():
    """Create an empty dataframe for testing."""
    return pd.DataFrame({"Courses": []})


@pytest.fixture
def dataframe_with_nan():
    """Create a dataframe with NaN values for testing."""
    return pd.DataFrame({
        "Courses": [
            "Probability and Random Processes",
            None,
            "Convex Optimization",
        ]
    })


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client for testing filter operations."""

    async def mock_call(messages, parse_tags=False, parse_code=False, **kwargs):
        """Mock LLM call that returns structured boolean output for filter."""
        text = str(messages)
        # Courses that require a lot of math:
        # - "Probability and Random Processes"
        # - "Convex Optimization"
        if "Probability and Random Processes" in text or "Convex Optimization" in text:
            value = "True"
        else:
            value = "False"

        result = {
            "raw_output": f"<output>{value}</output>",
            "cost": 0.01,
        }

        if parse_tags:
            tags = kwargs.get("tags", [])
            for tag in tags:
                if tag == "output":
                    result[tag] = value

        return result

    mock_client = AsyncMock(spec=LLMClient, side_effect=mock_call)
    mock_client.default_model = "gpt-4"
    return mock_client


class TestFilterOperation:
    """Test suite for FilterOperation class."""

    def test_filter_operation_initialization(self, mock_llm_client):
        """Test that FilterOperation initializes correctly."""
        FilterOperation.set_llm(mock_llm_client)
        op = FilterOperation(
            user_instruction="The course requires a lot of math",
            input_columns=["Courses"],
            strategy="plain",
        )

        assert op.op_name == "filter"
        assert op.input_columns == ["Courses"]
        assert op.dependencies == ["Courses"]
        assert op.generated_fields == []

    def test_op_kwargs_property(self, mock_llm_client):
        """Test that op_kwargs includes input_columns."""
        FilterOperation.set_llm(mock_llm_client)
        op = FilterOperation(
            user_instruction="The course requires a lot of math",
            input_columns=["Courses"],
            strategy="plain",
        )

        kwargs = op.op_kwargs
        assert "input_columns" in kwargs
        assert kwargs["input_columns"] == ["Courses"]

    @pytest.mark.asyncio
    async def test_execute_with_empty_dataframe(self, mock_llm_client):
        """Test that execute returns default outputs for empty dataframe."""
        FilterOperation.set_llm(mock_llm_client)

        op = FilterOperation(
            user_instruction="The course requires a lot of math",
            input_columns=["Courses"],
        )

        empty_df = pd.DataFrame({"Courses": []})
        result = await op.execute(input_data=empty_df)

        assert isinstance(result, FilterOpOutputs)
        assert result.output is None or result.output == []
        assert result.cost == 0.0

    @pytest.mark.asyncio
    async def test_execute_with_plain_strategy(self, sample_dataframe, mock_llm_client):
        """Test execute with plain LLM strategy."""
        FilterOperation.set_llm(mock_llm_client)

        op = FilterOperation(
            user_instruction="The course requires a lot of math",
            input_columns=["Courses"],
            strategy="plain",
        )

        result = await op.execute(input_data=sample_dataframe)

        assert isinstance(result, FilterOpOutputs)
        assert result.output == [True, False, False, True]
        assert result.cost > 0

    @pytest.mark.asyncio
    async def test_execute_with_nan_values(self, dataframe_with_nan, mock_llm_client):
        """Test that execute handles NaN values correctly."""
        FilterOperation.set_llm(mock_llm_client)

        op = FilterOperation(
            user_instruction="The course requires a lot of math",
            input_columns=["Courses"],
            strategy="plain",
        )

        result = await op.execute(input_data=dataframe_with_nan)

        assert isinstance(result, FilterOpOutputs)
        # NaN rows should be treated as False
        assert result.output == [True, False, True]
        assert result.cost > 0

    @pytest.mark.asyncio
    async def test_execute_with_udf(self, sample_dataframe, mock_llm_client):
        """Test execute with user-defined function."""
        FilterOperation.set_llm(mock_llm_client)

        def requires_lot_of_math(course: str) -> bool:
            """Simple predicate to identify math-heavy courses."""
            return course in {
                "Probability and Random Processes",
                "Convex Optimization",
            }

        tool = FunctionCallTool.from_function(func=requires_lot_of_math)

        op = FilterOperation(
            user_instruction="The course requires a lot of math",
            input_columns=["Courses"],
            tool=tool,
        )

        result = await op.execute(input_data=sample_dataframe)

        assert isinstance(result, FilterOpOutputs)
        assert result.output == [True, False, False, True]
        assert result.cost == 0.0

    @pytest.mark.asyncio
    async def test_execute_with_udf_exception_fallback(self, sample_dataframe, mock_llm_client):
        """Test that UDF exceptions fall back to LLM."""
        FilterOperation.set_llm(mock_llm_client)

        def failing_func(course: str) -> bool:
            """Function that always raises an exception."""
            raise ValueError("Function failed")

        tool = FunctionCallTool.from_function(func=failing_func)

        op = FilterOperation(
            user_instruction="The course requires a lot of math",
            input_columns=["Courses"],
            tool=tool,
            strategy="plain",
        )

        result = await op.execute(input_data=sample_dataframe)

        # Should fall back to LLM, so result should still be valid
        assert isinstance(result, FilterOpOutputs)
        assert result.output == [True, False, False, True]
        assert result.cost > 0

    @pytest.mark.asyncio
    async def test_execute_missing_user_instruction_and_func(self, sample_dataframe, mock_llm_client):
        """Test that execute raises ValueError when both user_instruction and func are missing."""
        FilterOperation.set_llm(mock_llm_client)

        op = FilterOperation(
            user_instruction=None,
            input_columns=["Courses"],
        )

        with pytest.raises(ValueError, match="Neither `user_instruction` nor `func` is given."):
            await op.execute(input_data=sample_dataframe)

    @pytest.mark.asyncio
    async def test_execute_fewshot_without_context(self, sample_dataframe, mock_llm_client):
        """Test that execute raises AssertionError for fewshot without context."""
        FilterOperation.set_llm(mock_llm_client)

        op = FilterOperation(
            user_instruction="The course requires a lot of math",
            input_columns=["Courses"],
            strategy="fewshot",
        )

        with pytest.raises(AssertionError, match="Few-shot examples must be provided"):
            await op.execute(input_data=sample_dataframe)


class TestFilterOpOutputs:
    """Test suite for FilterOpOutputs class."""

    def test_filter_op_outputs_initialization(self):
        """Test that FilterOpOutputs initializes correctly."""
        outputs = FilterOpOutputs(
            output=[True, False, True],
            cost=0.05,
        )

        assert outputs.output == [True, False, True]
        assert outputs.cost == 0.05

    def test_filter_op_outputs_default_values(self):
        """Test that FilterOpOutputs has correct default values."""
        outputs = FilterOpOutputs()

        assert outputs.output is None
        assert outputs.cost == 0.0

    def test_filter_op_outputs_addition(self):
        """Test that FilterOpOutputs can be added together."""
        outputs1 = FilterOpOutputs(output=[True, False], cost=0.02)
        outputs2 = FilterOpOutputs(output=[False, True], cost=0.03)

        combined = outputs1 + outputs2

        assert combined.output == [True, False, False, True]
        assert combined.cost == 0.05


class TestFilterWrapper:
    """Test suite for filter_wrapper function."""

    def test_filter_wrapper_basic(self, sample_dataframe, mock_llm_client):
        """Test the filter_wrapper function with basic usage."""
        FilterOperation.set_llm(mock_llm_client)

        result = nv.ops.filter(
            sample_dataframe,
            "The course requires a lot of math",
            input_columns=["Courses"],
            strategy="plain",
        )

        assert isinstance(result, FilterOpOutputs)
        assert result.output == [True, False, False, True]
        assert result.cost > 0


