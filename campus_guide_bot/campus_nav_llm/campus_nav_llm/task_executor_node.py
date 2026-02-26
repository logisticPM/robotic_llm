"""Task Executor Node — Phase 1 MVP.

Receives tool commands from LLM Planner, executes them against ROS
subsystems, and returns structured results.

Phase 1: navigate_to, get_robot_position, speak only.
No person detection, no spatial memory, no battery monitoring.
"""
import json
import math
import time
import threading
import logging

from campus_nav_llm.location_resolver import LocationResolver

logger = logging.getLogger(__name__)


class TaskExecutorCore:
    """Core tool dispatch logic — no ROS dependency.

    Navigation is delegated via the `navigator` interface.
    For testing, pass a mock navigator.
    """

    def __init__(self, semantic_map: dict, navigator=None):
        self.resolver = LocationResolver(semantic_map)
        self.navigator = navigator
        self._robot_pose = {"x": 0.0, "y": 0.0, "theta": 0.0}
        self._pose_lock = threading.Lock()

        self._dispatch = {
            "navigate_to": self._navigate,
            "get_robot_position": self._get_position,
            "speak": self._speak,
        }

    def update_pose(self, x: float, y: float, theta: float):
        """Update the robot's current pose (called by AMCL subscriber)."""
        with self._pose_lock:
            self._robot_pose = {
                "x": round(x, 3),
                "y": round(y, 3),
                "theta": round(theta, 3),
            }

    def get_pose(self) -> dict:
        with self._pose_lock:
            return dict(self._robot_pose)

    def execute(self, tool_name: str, tool_input: dict) -> dict:
        """Execute a tool call and return the result dict."""
        handler = self._dispatch.get(tool_name)
        if not handler:
            return {"error": f"Unknown tool: {tool_name}"}
        try:
            return handler(tool_input)
        except Exception as e:
            logger.error("Tool %s failed: %s", tool_name, e)
            return {"error": str(e)}

    def _navigate(self, inp: dict) -> dict:
        loc_name = inp.get("location_name", "")
        resolved = self.resolver.resolve(loc_name)
        if not resolved:
            return {
                "error": f"Unknown location: '{loc_name}'",
                "available": self.resolver.location_names,
            }

        name, info = resolved
        target_x, target_y = info["x"], info["y"]
        facing_deg = info.get("facing_deg", 0)

        if self.navigator is None:
            # No navigator available (testing mode)
            return {
                "status": "arrived",
                "target": name,
                "position": {"x": target_x, "y": target_y},
                "note": "simulated (no navigator)",
            }

        # Real navigation via TurtleBot4Navigator
        facing_rad = math.radians(facing_deg)
        try:
            goal = self.navigator.getPoseStamped(
                [target_x, target_y], _deg_to_direction(facing_deg)
            )
            self.navigator.startToPose(goal)

            # Wait for navigation to complete (poll loop)
            while not self.navigator.isTaskComplete():
                time.sleep(0.5)

            nav_result = self.navigator.getResult()
            arrived = nav_result == 0
        except Exception as e:
            return {
                "status": "failed",
                "target": name,
                "error": str(e),
            }

        return {
            "status": "arrived" if arrived else "failed",
            "target": name,
            "position": {"x": target_x, "y": target_y},
        }

    def _get_position(self, inp: dict) -> dict:
        pose = self.get_pose()
        nearby = self.resolver.get_nearby(pose["x"], pose["y"], radius=3.0)
        return {**pose, "nearby_locations": nearby}

    def _speak(self, inp: dict) -> dict:
        text = inp.get("text", "")
        logger.info("Robot says: %s", text)
        return {"status": "spoken", "text": text}


def _deg_to_direction(deg: float):
    """Convert facing angle to TurtleBot4Navigator direction constant.

    Only imported when running on real robot.
    """
    from turtlebot4_navigator import TurtleBot4Directions

    mapping = {
        0: TurtleBot4Directions.EAST,
        90: TurtleBot4Directions.NORTH,
        180: TurtleBot4Directions.WEST,
        270: TurtleBot4Directions.SOUTH,
    }
    normalized = deg % 360
    closest = min(mapping.keys(), key=lambda k: min(abs(k - normalized), 360 - abs(k - normalized)))
    return mapping[closest]


# ── ROS 2 Node wrapper ──

def create_ros_node():
    """Factory function to create the ROS 2 node."""
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
    from geometry_msgs.msg import PoseWithCovarianceStamped
    from turtlebot4_navigator import TurtleBot4Navigator

    class TaskExecutorNode(Node):
        def __init__(self):
            super().__init__("task_executor")

            # Load semantic map
            self.declare_parameter("semantic_map_path", "semantic_map.json")
            map_path = self.get_parameter("semantic_map_path").value
            from campus_nav_llm.location_resolver import load_semantic_map
            semantic_map = load_semantic_map(map_path)

            # Initialize navigator
            navigator = TurtleBot4Navigator()
            navigator.waitUntilNav2Active()
            self.get_logger().info("Nav2 active")

            self.executor_core = TaskExecutorCore(semantic_map, navigator)

            # AMCL pose subscription
            self.create_subscription(
                PoseWithCovarianceStamped,
                "/amcl_pose",
                self._on_amcl,
                10,
            )

            # Tool command / result pub-sub
            self.sub_cmd = self.create_subscription(
                String, "/tool_cmd", self._on_tool_cmd, 10
            )
            self.pub_result = self.create_publisher(
                String, "/tool_result", 10
            )
            self.pub_reply = self.create_publisher(
                String, "/robot_reply", 10
            )

            self.get_logger().info("Task Executor ready (Phase 1)")

        def _on_amcl(self, msg):
            p = msg.pose.pose
            q = p.orientation
            yaw = math.atan2(
                2 * (q.w * q.z + q.x * q.y),
                1 - 2 * (q.y * q.y + q.z * q.z),
            )
            self.executor_core.update_pose(
                p.position.x, p.position.y, yaw
            )

        def _on_tool_cmd(self, msg):
            threading.Thread(
                target=self._execute,
                args=(msg.data,),
                daemon=True,
            ).start()

        def _execute(self, cmd_json):
            cmd = json.loads(cmd_json)
            name = cmd["tool_name"]
            inp = cmd["tool_input"]

            result = self.executor_core.execute(name, inp)

            # If speak, also publish to /robot_reply
            if name == "speak" and "text" in result:
                self.pub_reply.publish(String(data=result["text"]))

            self.pub_result.publish(String(data=json.dumps(result)))

    return TaskExecutorNode


def main():
    import rclpy

    rclpy.init()
    NodeClass = create_ros_node()
    node = NodeClass()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
