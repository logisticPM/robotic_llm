"""Integration test — simulates the full Phase 1 pipeline.

No ROS, no real Claude API. Tests that LLMPlannerCore + TaskExecutorCore
work together correctly via mock API responses.
"""
import json
from unittest.mock import MagicMock, patch

import pytest

from campus_nav_llm.llm_planner_node import LLMPlannerCore
from campus_nav_llm.task_executor_node import TaskExecutorCore


@pytest.fixture
def system(sample_semantic_map):
    """Set up a connected Planner + Executor system."""
    planner = LLMPlannerCore(sample_semantic_map, api_key="fake")
    executor = TaskExecutorCore(sample_semantic_map, navigator=None)
    return planner, executor


def _make_tool_block(name, inp, tool_id="t1"):
    block = MagicMock()
    block.type = "tool_use"
    block.name = name
    block.input = inp
    block.id = tool_id
    return block


def _make_text_block(text):
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


def _make_response(stop_reason, content):
    resp = MagicMock()
    resp.stop_reason = stop_reason
    resp.content = content
    return resp


class TestEndToEndNavigation:
    """User says 'go to whiteboard' → LLM calls navigate_to → executor navigates → LLM replies."""

    def test_single_navigation(self, system):
        planner, executor = system

        # Step 1: LLM decides to call navigate_to
        nav_block = _make_tool_block("navigate_to", {"location_name": "whiteboard"})
        tool_resp = _make_response("tool_use", [nav_block])

        # Step 2: LLM sees the result and speaks
        speak_block = _make_tool_block("speak", {"text": "I've arrived at the whiteboard!"}, "t2")
        speak_resp = _make_response("tool_use", [speak_block])

        # Step 3: LLM finishes
        text_block = _make_text_block("Done! I'm at the whiteboard now.")
        final_resp = _make_response("end_turn", [text_block])

        with patch.object(
            planner.client.messages,
            "create",
            side_effect=[tool_resp, speak_resp, final_resp],
        ):
            result = planner.run_tool_loop(
                "go to the whiteboard",
                lambda name, inp: executor.execute(name, inp),
            )
            assert "whiteboard" in result.lower()


class TestMultiStepNavigation:
    """User says 'go to desk 1 then entrance' → 2 navigations."""

    def test_two_step_navigation(self, system):
        planner, executor = system

        # Step 1: navigate to desk_1
        nav1 = _make_tool_block("navigate_to", {"location_name": "desk_1"}, "t1")
        resp1 = _make_response("tool_use", [nav1])

        # Step 2: navigate to entrance
        nav2 = _make_tool_block("navigate_to", {"location_name": "entrance"}, "t2")
        resp2 = _make_response("tool_use", [nav2])

        # Step 3: done
        text = _make_text_block("Visited desk 1 and entrance.")
        resp3 = _make_response("end_turn", [text])

        call_log = []

        def logging_executor(name, inp):
            result = executor.execute(name, inp)
            call_log.append((name, inp, result))
            return result

        with patch.object(
            planner.client.messages,
            "create",
            side_effect=[resp1, resp2, resp3],
        ):
            result = planner.run_tool_loop(
                "go to desk 1 then the entrance",
                logging_executor,
            )
            assert len(call_log) == 2
            assert call_log[0][0] == "navigate_to"
            assert call_log[0][2]["target"] == "desk_1"
            assert call_log[1][0] == "navigate_to"
            assert call_log[1][2]["target"] == "entrance"


class TestPositionQuery:
    """User asks 'where am I?' → LLM calls get_robot_position."""

    def test_position_query(self, system):
        planner, executor = system
        executor.update_pose(2.0, 3.5, 1.57)

        pos_block = _make_tool_block("get_robot_position", {})
        pos_resp = _make_response("tool_use", [pos_block])

        text = _make_text_block("You are near the whiteboard at (2.0, 3.5).")
        final_resp = _make_response("end_turn", [text])

        with patch.object(
            planner.client.messages,
            "create",
            side_effect=[pos_resp, final_resp],
        ):
            result = planner.run_tool_loop(
                "where am I?",
                lambda name, inp: executor.execute(name, inp),
            )
            assert "whiteboard" in result.lower()


class TestUnknownLocationHandling:
    """User asks to go to a non-existent location → error, LLM handles it."""

    def test_unknown_location_error_recovery(self, system):
        planner, executor = system

        # LLM tries to navigate to 'library' (doesn't exist)
        nav_block = _make_tool_block("navigate_to", {"location_name": "library"})
        nav_resp = _make_response("tool_use", [nav_block])

        # LLM sees the error and responds with text
        text = _make_text_block(
            "I don't know where the library is. Available locations are: "
            "whiteboard, desk_1, entrance."
        )
        final_resp = _make_response("end_turn", [text])

        with patch.object(
            planner.client.messages,
            "create",
            side_effect=[nav_resp, final_resp],
        ):
            result = planner.run_tool_loop(
                "go to the library",
                lambda name, inp: executor.execute(name, inp),
            )
            # LLM should report the error gracefully
            assert "library" in result.lower() or "don't know" in result.lower()


class TestTier1Commands:
    """Simulate Tier 1 evaluation commands (direct navigation)."""

    TIER1_COMMANDS = [
        ("go to the whiteboard", "whiteboard"),
        ("navigate to desk 1", "desk_1"),
        ("go to the entrance", "entrance"),
        ("take me to the board", "whiteboard"),  # alias
        ("go to the door", "entrance"),  # alias
    ]

    @pytest.mark.parametrize("command,expected_target", TIER1_COMMANDS)
    def test_tier1_command(self, system, command, expected_target):
        planner, executor = system

        nav_block = _make_tool_block(
            "navigate_to", {"location_name": expected_target}
        )
        nav_resp = _make_response("tool_use", [nav_block])

        text = _make_text_block(f"Arrived at {expected_target}.")
        final_resp = _make_response("end_turn", [text])

        results = []

        def track_executor(name, inp):
            result = executor.execute(name, inp)
            results.append(result)
            return result

        with patch.object(
            planner.client.messages,
            "create",
            side_effect=[nav_resp, final_resp],
        ):
            planner.conversation_history.clear()
            reply = planner.run_tool_loop(command, track_executor)

        # Navigation should succeed
        assert len(results) >= 1
        nav_result = results[0]
        assert nav_result["status"] == "arrived"
        assert nav_result["target"] == expected_target
