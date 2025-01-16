import os
import sys
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from aiohttp import ClientSession

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tool.tavily_search import (  # noqa: E402
    TavilySearchTool,
    TavilySearchToolInputSchema,
    TavilySearchToolConfig,
)


@pytest.fixture
def mock_aiohttp_session():
    # Patch the ClientSession in the module where it's used
    with patch("aiohttp.ClientSession") as mock_session_class:
        mock_session = MagicMock(spec=ClientSession)
        # Mock the async context manager __aenter__ to return the mock session
        mock_session_class.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        yield mock_session


@pytest.mark.asyncio
async def test_searxng_search_tool_missing_fields(mock_aiohttp_session):
    mock_api_key = "KEY"
    mock_query = "query with missing fields"
    mock_response_data = {
        "results": [
            {"title": "Result Missing Content", "url": "https://example.com/1", "query": mock_query},
            {"content": "Result Missing Title", "url": "https://example.com/2", "query": mock_query},
            {"title": "Result Missing URL", "content": "Some content", "query": mock_query},
            {"title": "Result Missing Query", "url": "https://example.com/4", "content": "Some content"},
            {"title": "Valid Result", "url": "https://example.com/5", "content": "Valid content", "query": mock_query},
        ]
    }

    # Create a mock response object
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json.return_value = mock_response_data

    # Configure the mock
    mock_aiohttp_session.get.return_value.__aenter__.return_value = mock_response

    # Initialize the tool
    tavily_tool = TavilySearchTool(TavilySearchToolConfig(api_key=mock_api_key))
    input_schema = TavilySearchToolInputSchema(queries=[mock_query])

    # Run the tool
    result = await tavily_tool.run_async(input_schema)

    # The last two results should be included
    assert len(result.results) == 2
    assert result.results[0].title == "Result Missing Query"
    assert result.results[1].title == "Valid Result"


def test_searxng_search_tool_sync_run_method(mock_aiohttp_session):
    mock_api_key = "KEY"
    mock_query = "sync query"
    mock_response_data = {
        "results": [
            {
                "title": "Sync Test Result",
                "url": "https://example.com/sync",
                "content": "Sync content",
                "query": mock_query,
            }
        ]
    }

    # Create a mock response object
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json.return_value = mock_response_data

    # Configure the mock
    mock_aiohttp_session.get.return_value.__aenter__.return_value = mock_response

    # Initialize the tool
    tavily_tool = TavilySearchTool(TavilySearchToolConfig(api_key=mock_api_key))
    input_schema = TavilySearchToolInputSchema(queries=[mock_query])

    # Run the tool synchronously
    result = tavily_tool.run(input_schema)

    # Assertions
    assert len(result.results) == 1
    assert result.results[0].title == "Sync Test Result"


@pytest.mark.asyncio
async def test_searxng_search_tool_with_max_results(mock_aiohttp_session):
    mock_api_key = "KEY"
    mock_query = "test query with max results"
    mock_response_data = {
        "results": [
            {"title": f"Result {i}", "url": f"https://example.com/{i}", "content": f"Content {i}", "query": mock_query}
            for i in range(10)
        ]
    }

    # Create a mock response object
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json.return_value = mock_response_data

    # Configure the mock
    mock_aiohttp_session.get.return_value.__aenter__.return_value = mock_response

    # Initialize the tool
    tavily_tool = TavilySearchTool(TavilySearchToolConfig(api_key=mock_api_key, max_results=10))
    input_schema = TavilySearchToolInputSchema(queries=[mock_query])

    # Run the tool with max_results=5
    result = await tavily_tool.run_async(input_schema, max_results=5)

    # Assertions
    assert len(result.results) == 5


@pytest.mark.asyncio
async def test_searxng_search_tool_no_results(mock_aiohttp_session):
    # Existing test case remains the same
    mock_api_key = "KEY"
    mock_query = "no results query"
    mock_response_data = {"results": []}

    # Create a mock response object
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json.return_value = mock_response_data

    # Configure the mock
    mock_aiohttp_session.get.return_value.__aenter__.return_value = mock_response

    # Initialize the tool
    tavily_tool = TavilySearchTool(TavilySearchToolConfig(api_key=mock_api_key))
    input_schema = TavilySearchToolInputSchema(queries=[mock_query])

    # Run the tool
    result = await tavily_tool.run_async(input_schema)

    # Assertions
    assert len(result.results) == 0


@pytest.mark.asyncio
async def test_searxng_search_tool_error(mock_aiohttp_session):
    # Existing test case remains the same
    mock_api_key = "KEY"
    mock_query = "error query"

    # Create a mock response object with a 500 status
    mock_response = AsyncMock()
    mock_response.status = 500
    mock_response.reason = "Internal Server Error"

    # Configure the mock
    mock_aiohttp_session.get.return_value.__aenter__.return_value = mock_response

    # Initialize the tool
    tavily_tool = TavilySearchTool(TavilySearchToolConfig(api_key=mock_api_key))
    input_schema = TavilySearchToolInputSchema(queries=[mock_query])

    # Run the tool and expect an exception
    with pytest.raises(Exception) as excinfo:
        await tavily_tool.run_async(input_schema)

    # Assertion on the exception message
    assert "Failed to fetch search results" in str(excinfo.value)


if __name__ == "__main__":
    pytest.main([__file__])
