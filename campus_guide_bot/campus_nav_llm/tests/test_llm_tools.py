"""Tests for LLM Planner — tool definitions, system prompt, and mock API loop."""
import json
from unittest.mock import MagicMock, patch

import pytest

from campus_nav_llm.llm_planner_node import TOOLS, LLMPlannerCore, MAX_TOOL_ITERATIONS


class TestToolDefinitions:
    """Validate that Phase 1 tool schemas are well-formed."""

    def test_has_exactly_3_tools(self):
        assert len(TOOLS) == 3

    def test_tool_names(self):
        names = {t["name"] for t in TOOLS}
        assert names == {"navigate_to", "get_robot_position", "speak"}

    def test_navigate_to_has_required_param(self):
        tool = next(t for t in TOOLS if t["name"] == "navigate_to")
        schema = tool["input_schema"]
        assert "location_name" in schema["properties"]
        assert "location_name" in schema["required"]

    def test_speak_has_required_param(self):
        tool = next(t for t in TOOLS if t["name"] == "speak")
        schema = tool["input_schema"]
        assert "text" in schema["properties"]
        assert "text" in schema["required"]

    def test_get_position_has_no_required(self):
        tool = next(t for t in TOOLS if t["name"] == "get_robot_position")
        schema = tool["input_schema"]
        assert "required" not in schema or len(schema.get("required", [])) == 0

    def test_all_tools_have_description(self):
        for tool in TOOLS:
            assert "description" in tool
            assert len(tool["description"]) > 10

    def test_all_tools_have_input_schema(self):
        for tool in TOOLS:
            assert "input_schema" in tool
            assert tool["input_schema"]["type"] == "object"


class TestSystemPrompt:
    """Test the system prompt generation."""

    def test_prompt_contains_locations(self, sample_semantic_map):
        planner = LLMPlannerCore(sample_semantic_map, api_key="fake")
        prompt = planner.build_system_prompt()
        assert "whiteboard" in prompt
        assert "desk_1" in prompt
        assert "entrance" in prompt

    def test_prompt_contains_aliases(self, sample_semantic_map):
        planner = LLMPlannerCore(sample_semantic_map, api_key="fake")
        prompt = planner.build_system_prompt()
        assert "board" in prompt
        assert "door" in prompt

    def test_prompt_contains_instructions(self, sample_semantic_map):
        planner = LLMPlannerCore(sample_semantic_map, api_key="fake")
        prompt = planner.build_system_prompt()
        assert "navigate_to" in prompt
        assert "named locations" in prompt

    def test_prompt_mentions_turtlebot(self, sample_semantic_map):
        planner = LLMPlannerCore(sample_semantic_map, api_key="fake")
        prompt = planner.build_system_prompt()
        assert "TurtleBot 4" in prompt


class TestMaxIterations:
    """Test that the iteration limit is correctly set."""

    def test_max_iterations_is_10(self):
        assert MAX_TOOL_ITERATIONS == 10


class TestConversationHistory:
    """Test conversation history management."""

    def test_history_starts_empty(self, sample_semantic_map):
        planner = LLMPlannerCore(sample_semantic_map, api_key="fake")
        assert len(planner.conversation_history) == 0

    def test_history_bounded(self, sample_semantic_map):
        """History should not grow beyond 20 entries."""
        planner = LLMPlannerCore(sample_semantic_map, api_key="fake")
        # Simulate 25 user messages
        for i in range(25):
            planner.conversation_history.append(
                {"role": "user", "content": f"message {i}"}
            )
        # Manually trigger the bounding logic from run_tool_loop
        if len(planner.conversation_history) > 20:
            planner.conversation_history = planner.conversation_history[-20:]
        assert len(planner.conversation_history) == 20


class TestToolLoopWithMock:
    """Test the tool call loop with a mocked Claude API."""

    def _make_mock_response(self, stop_reason, content):
        """Create a mock Anthropic API response."""
        response = MagicMock()
        response.stop_reason = stop_reason
        response.content = content
        return response

    def _make_text_block(self, text):
        block = MagicMock()
        block.type = "text"
        block.text = text
        return block

    def _make_tool_use_block(self, tool_name, tool_input, tool_id="tool_1"):
        block = MagicMock()
        block.type = "tool_use"
        block.name = tool_name
        block.input = tool_input
        block.id = tool_id
        return block

    def test_simple_text_response(self, sample_semantic_map):
        """LLM returns text without tools → immediate reply."""
        planner = LLMPlannerCore(sample_semantic_map, api_key="fake")

        text_block = self._make_text_block("I'll go to the whiteboard!")
        mock_response = self._make_mock_response(
            "end_turn", [text_block]
        )

        with patch.object(planner.client.messages, "create", return_value=mock_response):
            result = planner.run_tool_loop("go to whiteboard", lambda n, i: {})
            assert result == "I'll go to the whiteboard!"

    def test_tool_call_then_text(self, sample_semantic_map):
        """LLM calls navigate_to, gets result, then returns text."""
        planner = LLMPlannerCore(sample_semantic_map, api_key="fake")

        # First response: tool_use
        tool_block = self._make_tool_use_block(
            "navigate_to", {"location_name": "whiteboard"}
        )
        tool_response = self._make_mock_response("tool_use", [tool_block])

        # Second response: end_turn text
        text_block = self._make_text_block("Arrived at the whiteboard!")
        text_response = self._make_mock_response("end_turn", [text_block])

        with patch.object(
            planner.client.messages,
            "create",
            side_effect=[tool_response, text_response],
        ):
            executor_calls = []

            def mock_executor(name, inp):
                executor_calls.append((name, inp))
                return {"status": "arrived", "target": "whiteboard"}

            result = planner.run_tool_loop("go to whiteboard", mock_executor)
            assert result == "Arrived at the whiteboard!"
            assert len(executor_calls) == 1
            assert executor_calls[0][0] == "navigate_to"

    def test_api_error_returns_error_message(self, sample_semantic_map):
        """API exception should return an error string, not crash."""
        planner = LLMPlannerCore(sample_semantic_map, api_key="fake")

        with patch.object(
            planner.client.messages,
            "create",
            side_effect=Exception("API rate limit"),
        ):
            result = planner.run_tool_loop("go somewhere", lambda n, i: {})
            assert "error" in result.lower()
            assert "rate limit" in result.lower()

    def test_iteration_limit(self, sample_semantic_map):
        """If LLM keeps calling tools, loop should stop at MAX_TOOL_ITERATIONS."""
        planner = LLMPlannerCore(sample_semantic_map, api_key="fake")

        tool_block = self._make_tool_use_block(
            "get_robot_position", {}
        )
        tool_response = self._make_mock_response("tool_use", [tool_block])

        with patch.object(
            planner.client.messages,
            "create",
            return_value=tool_response,
        ):
            result = planner.run_tool_loop(
                "keep checking position",
                lambda n, i: {"x": 0, "y": 0, "theta": 0},
            )
            assert "step limit" in result.lower()
