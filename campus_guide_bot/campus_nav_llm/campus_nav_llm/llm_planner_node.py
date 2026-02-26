"""LLM Planner Node — Phase 1 MVP.

Receives user natural language input, runs LLM Tool Call loop via
OpenRouter (OpenAI-compatible API), publishes tool commands to Task
Executor, receives results.

Phase 1 tools: navigate_to, get_robot_position, speak
"""
import json
import os
import threading
import logging

from openai import OpenAI

from campus_nav_llm.location_resolver import LocationResolver, load_semantic_map

logger = logging.getLogger(__name__)

MAX_TOOL_ITERATIONS = 10

# Phase 1: 3 tools (OpenAI function calling format)
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "navigate_to",
            "description": (
                "Navigate to a named location from the semantic map. "
                "The location must exist in the map. Use exact location "
                "names or known aliases."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "location_name": {
                        "type": "string",
                        "description": "Name or alias of the target location",
                    }
                },
                "required": ["location_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_robot_position",
            "description": "Get the robot's current (x, y, theta) position on the map.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "speak",
            "description": "Say something to the user (status update, greeting, confirmation, etc.)",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to say to the user"}
                },
                "required": ["text"],
            },
        },
    },
]


class LLMPlannerCore:
    """Core LLM planner logic — no ROS dependency.

    This class manages the LLM Tool Call loop via OpenRouter.
    Tool execution is delegated via the `tool_executor` callback.
    """

    def __init__(
        self,
        semantic_map: dict,
        model: str = "anthropic/claude-sonnet-4-6",
        api_key: str | None = None,
        base_url: str = "https://openrouter.ai/api/v1",
    ):
        self.client = OpenAI(
            api_key=api_key or os.environ.get("OPENROUTER_API_KEY", ""),
            base_url=base_url,
        )
        self.model = model
        self.resolver = LocationResolver(semantic_map)
        self.conversation_history: list[dict] = []
        self._lock = threading.Lock()

    def build_system_prompt(self) -> str:
        """Build the system prompt with available locations."""
        locations_text = self.resolver.get_all_locations_text()

        return (
            "You are a navigation assistant running on a TurtleBot 4 robot "
            "inside a classroom at Northeastern University Vancouver.\n\n"
            f"Available locations:\n{locations_text}\n\n"
            "Instructions:\n"
            "- Use tools to complete tasks step by step.\n"
            "- Always call navigate_to with the exact location name from the map.\n"
            "- If the user asks about your position, use get_robot_position.\n"
            "- After navigating, confirm arrival with a speak tool call.\n"
            "- Never guess coordinates; always use named locations.\n"
            "- If the user's request is ambiguous, use speak to ask for clarification.\n"
            "- For multi-step tasks (e.g. 'go to A then B'), execute them in order.\n"
        )

    def run_tool_loop(
        self,
        user_input: str,
        tool_executor: callable,
    ) -> str:
        """Run the LLM Tool Call loop for a user input.

        Args:
            user_input: Natural language command from user.
            tool_executor: Callback function(tool_name, tool_input) -> dict.
                           Called for each tool the LLM wants to invoke.

        Returns:
            Final text reply from the LLM.
        """
        with self._lock:
            self.conversation_history.append(
                {"role": "user", "content": user_input}
            )
            # Keep history bounded
            if len(self.conversation_history) > 20:
                self.conversation_history = self.conversation_history[-20:]
            messages = list(self.conversation_history)

        system_prompt = self.build_system_prompt()

        for iteration in range(MAX_TOOL_ITERATIONS):
            try:
                api_messages = [
                    {"role": "system", "content": system_prompt}
                ] + messages

                response = self.client.chat.completions.create(
                    model=self.model,
                    max_tokens=1024,
                    tools=TOOLS,
                    messages=api_messages,
                )
            except Exception as e:
                error_msg = f"LLM API error: {e}"
                logger.error(error_msg)
                return error_msg

            choice = response.choices[0]

            # stop: LLM is done, extract text reply
            if choice.finish_reason == "stop":
                reply = choice.message.content or "Done."
                with self._lock:
                    self.conversation_history.append(
                        {"role": "assistant", "content": reply}
                    )
                return reply

            # tool_calls: execute tool(s) and continue loop
            if choice.finish_reason == "tool_calls":
                # Append assistant message with tool calls
                messages.append(choice.message.model_dump())

                for tc in choice.message.tool_calls:
                    logger.info(
                        "[%d/%d] Tool: %s(%s)",
                        iteration + 1,
                        MAX_TOOL_ITERATIONS,
                        tc.function.name,
                        tc.function.arguments,
                    )

                    # Execute tool via callback
                    try:
                        tool_input = json.loads(tc.function.arguments)
                        result = tool_executor(tc.function.name, tool_input)
                    except Exception as e:
                        result = {"error": str(e)}

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(result),
                    })

        return "I could not complete the task within the step limit."


# ── ROS 2 Node wrapper (only imported when rclpy is available) ──

def create_ros_node():
    """Factory function to create the ROS 2 node.

    Import rclpy only when actually running on the robot.
    """
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String

    class LLMPlannerNode(Node):
        def __init__(self):
            super().__init__("llm_planner")

            # Parameters
            self.declare_parameter("semantic_map_path", "semantic_map.json")
            map_path = self.get_parameter("semantic_map_path").value
            semantic_map = load_semantic_map(map_path)

            self.planner = LLMPlannerCore(semantic_map)

            # Pub/Sub
            self.sub_input = self.create_subscription(
                String, "/user_input", self._on_user_input, 10
            )
            self.pub_tool_cmd = self.create_publisher(String, "/tool_cmd", 10)
            self.sub_tool_result = self.create_subscription(
                String, "/tool_result", self._on_tool_result, 10
            )
            self.pub_reply = self.create_publisher(String, "/robot_reply", 10)

            # Synchronization for tool results
            self._pending_result = None
            self._result_event = threading.Event()

            self.get_logger().info("LLM Planner ready (Phase 1: 3 tools)")

        def _on_user_input(self, msg):
            threading.Thread(
                target=self._handle_input,
                args=(msg.data,),
                daemon=True,
            ).start()

        def _on_tool_result(self, msg):
            self._pending_result = msg.data
            self._result_event.set()

        def _wait_for_tool_result(self, timeout=60.0):
            self._result_event.clear()
            self._pending_result = None
            answered = self._result_event.wait(timeout=timeout)
            if not answered:
                return json.dumps({"error": "Tool execution timed out"})
            return self._pending_result

        def _ros_tool_executor(self, tool_name, tool_input):
            """Execute tool by publishing to /tool_cmd and waiting for /tool_result."""
            cmd = json.dumps({
                "tool_name": tool_name,
                "tool_input": tool_input,
            })
            self.pub_tool_cmd.publish(String(data=cmd))
            result_str = self._wait_for_tool_result()
            return json.loads(result_str)

        def _handle_input(self, user_input):
            reply = self.planner.run_tool_loop(
                user_input, self._ros_tool_executor
            )
            self.pub_reply.publish(String(data=reply))

    return LLMPlannerNode


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
