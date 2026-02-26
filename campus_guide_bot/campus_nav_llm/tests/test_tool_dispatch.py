"""Tests for TaskExecutorCore — tool dispatch and execution logic."""
import pytest

from campus_nav_llm.task_executor_node import TaskExecutorCore


@pytest.fixture
def executor(sample_semantic_map):
    """Create a TaskExecutorCore without a real navigator (testing mode)."""
    return TaskExecutorCore(sample_semantic_map, navigator=None)


class TestNavigateTo:
    """Test the navigate_to tool."""

    def test_navigate_exact_name(self, executor):
        result = executor.execute("navigate_to", {"location_name": "whiteboard"})
        assert result["status"] == "arrived"
        assert result["target"] == "whiteboard"
        assert result["position"]["x"] == 2.00
        assert result["position"]["y"] == 3.50

    def test_navigate_alias(self, executor):
        result = executor.execute("navigate_to", {"location_name": "door"})
        assert result["status"] == "arrived"
        assert result["target"] == "entrance"

    def test_navigate_unknown_location(self, executor):
        result = executor.execute("navigate_to", {"location_name": "cafeteria"})
        assert "error" in result
        assert "cafeteria" in result["error"]
        assert "available" in result

    def test_navigate_empty_name(self, executor):
        result = executor.execute("navigate_to", {"location_name": ""})
        assert "error" in result

    def test_navigate_partial_match(self, executor):
        result = executor.execute("navigate_to", {"location_name": "desk"})
        assert result["status"] == "arrived"
        assert result["target"] == "desk_1"

    def test_navigate_case_insensitive(self, executor):
        result = executor.execute("navigate_to", {"location_name": "WHITEBOARD"})
        assert result["status"] == "arrived"
        assert result["target"] == "whiteboard"


class TestGetRobotPosition:
    """Test the get_robot_position tool."""

    def test_default_position(self, executor):
        result = executor.execute("get_robot_position", {})
        assert "x" in result
        assert "y" in result
        assert "theta" in result
        assert result["x"] == 0.0
        assert result["y"] == 0.0

    def test_updated_position(self, executor):
        executor.update_pose(1.5, 2.5, 0.78)
        result = executor.execute("get_robot_position", {})
        assert result["x"] == 1.5
        assert result["y"] == 2.5
        assert result["theta"] == 0.78

    def test_position_includes_nearby(self, executor):
        executor.update_pose(2.0, 3.5, 0.0)  # at whiteboard
        result = executor.execute("get_robot_position", {})
        assert "nearby_locations" in result
        names = [loc["name"] for loc in result["nearby_locations"]]
        assert "whiteboard" in names


class TestSpeak:
    """Test the speak tool."""

    def test_speak_returns_text(self, executor):
        result = executor.execute("speak", {"text": "Hello!"})
        assert result["status"] == "spoken"
        assert result["text"] == "Hello!"

    def test_speak_empty_text(self, executor):
        result = executor.execute("speak", {"text": ""})
        assert result["status"] == "spoken"
        assert result["text"] == ""

    def test_speak_missing_text(self, executor):
        result = executor.execute("speak", {})
        assert result["status"] == "spoken"


class TestUnknownTool:
    """Test handling of undefined tools."""

    def test_unknown_tool(self, executor):
        result = executor.execute("fly_to_moon", {"destination": "moon"})
        assert "error" in result
        assert "Unknown tool" in result["error"]

    def test_phase2_tools_not_available(self, executor):
        """Phase 2 tools should not exist in Phase 1."""
        result = executor.execute("detect_persons", {})
        assert "error" in result

        result = executor.execute("ask_user", {"question": "Where?"})
        assert "error" in result

        result = executor.execute("recall_observations", {})
        assert "error" in result

        result = executor.execute("get_battery_status", {})
        assert "error" in result


class TestPoseThreadSafety:
    """Test that pose updates are thread-safe."""

    def test_concurrent_pose_updates(self, executor):
        import threading

        def update_loop(start_x):
            for i in range(100):
                executor.update_pose(start_x + i * 0.01, 0.0, 0.0)

        threads = [
            threading.Thread(target=update_loop, args=(0.0,)),
            threading.Thread(target=update_loop, args=(10.0,)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Just verify no crash; final value is nondeterministic
        pose = executor.get_pose()
        assert isinstance(pose["x"], float)
