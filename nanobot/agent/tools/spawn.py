"""Spawn tool and subagent management tools.

Provides 4 tools:
  - spawn              — 创建子 agent，支持自定义 model/tools/prompt/report_mode
  - subagent_message   — 向运行中的子 agent 发送纠正/补充消息
  - subagent_list      — 列出所有子 agent 及其状态
  - subagent_control   — 暂停/恢复/取消子 agent
"""

from __future__ import annotations

import json
from typing import Any, TYPE_CHECKING

from nanobot.agent.tools.base import Tool

if TYPE_CHECKING:
    from nanobot.agent.subagent import SubagentManager


# ═════════════════════════════════════════════════════════════════
# spawn — 创建子 agent
# ═════════════════════════════════════════════════════════════════

class SpawnTool(Tool):
    """Spawn a subagent with fully configurable parameters."""

    def __init__(self, manager: SubagentManager):
        self._manager = manager
        self._origin_channel = "cli"
        self._origin_chat_id = "direct"
        self._session_key = "cli:direct"

    def set_context(self, channel: str, chat_id: str) -> None:
        """Set the origin context for subagent announcements."""
        self._origin_channel = channel
        self._origin_chat_id = chat_id
        self._session_key = f"{channel}:{chat_id}"

    @property
    def name(self) -> str:
        return "spawn"

    @property
    def description(self) -> str:
        return (
            "Spawn a subagent to handle a task in the background. "
            "You can configure the subagent's model, tool set, system prompt, "
            "reporting behavior, and iteration limit. "
            "The subagent will execute the task autonomously and report back. "
            "Use subagent_message to send corrections mid-execution, "
            "subagent_control to pause/resume/cancel, "
            "and subagent_list to check status."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        # 动态构建模型描述：包含所有可用模型列表
        model_desc = (
            "LLM model for the subagent. "
            "If omitted, uses the default subagent model. "
            "The source will be automatically resolved based on the model name."
        )
        try:
            cfg = getattr(self._manager.provider, "cfg", None)
            if cfg and hasattr(cfg, "get_all_available_models"):
                all_models = cfg.get_all_available_models()
                if all_models:
                    models_str = ", ".join(
                        f"{m} (源{i}{', 多模态' if cfg.is_model_multimodal(m) else ''})"
                        for i, m in all_models
                    )
                    model_desc += f" Available models: [{models_str}]"
        except Exception:
            pass

        return {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "Detailed task description for the subagent",
                },
                "label": {
                    "type": "string",
                    "description": "Short display label (defaults to first 30 chars of task)",
                },
                "model": {
                    "type": "string",
                    "description": model_desc,
                },
                "tools": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Tool names to enable. If omitted, all available tools "
                        "are enabled (except spawn/management tools). "
                        "Example: [\"tavily_search\", \"read_file\"]"
                    ),
                },
                "system_prompt": {
                    "type": "string",
                    "description": (
                        "Custom system prompt for the subagent. "
                        "If omitted, a default focused prompt is generated"
                    ),
                },
                "report_mode": {
                    "type": "string",
                    "enum": [
                        "result_only",
                        "on_error",
                        "on_tool_call",
                        "every_step",
                    ],
                    "description": (
                        "Progress reporting: "
                        "'result_only' (default) — only final result; "
                        "'on_error' — errors + final; "
                        "'on_tool_call' — each tool call + final; "
                        "'every_step' — every LLM iteration + final"
                    ),
                },
                "max_iterations": {
                    "type": "integer",
                    "description": "Max agent iterations (default 15, max 50)",
                    "minimum": 1,
                    "maximum": 50,
                },
            },
            "required": ["task"],
        }

    async def execute(
        self,
        task: str,
        label: str | None = None,
        model: str | None = None,
        tools: list[str] | None = None,
        system_prompt: str | None = None,
        report_mode: str = "result_only",
        max_iterations: int = 15,
        **kwargs: Any,
    ) -> str:
        return await self._manager.spawn(
            task=task,
            label=label,
            model=model,
            tools=tools,
            system_prompt=system_prompt,
            report_mode=report_mode,
            max_iterations=max_iterations,
            origin_channel=self._origin_channel,
            origin_chat_id=self._origin_chat_id,
            session_key=self._session_key,
        )


# ═════════════════════════════════════════════════════════════════
# subagent_message — 向子 agent 发送消息/纠正
# ═════════════════════════════════════════════════════════════════

class SubagentMessageTool(Tool):
    """Send a message or correction to a running subagent."""

    def __init__(self, manager: SubagentManager):
        self._manager = manager

    @property
    def name(self) -> str:
        return "subagent_message"

    @property
    def description(self) -> str:
        return (
            "Send a message or correction to a running subagent. "
            "The message will be injected into the subagent's context "
            "as a supervisor correction before its next iteration. "
            "Use this to guide, correct, or provide additional information."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "string",
                    "description": "The subagent's task ID (from spawn result)",
                },
                "message": {
                    "type": "string",
                    "description": "The message or correction to send",
                },
            },
            "required": ["task_id", "message"],
        }

    async def execute(self, task_id: str, message: str, **kwargs: Any) -> str:
        return await self._manager.send_message(task_id, message)


# ═════════════════════════════════════════════════════════════════
# subagent_list — 列出子 agent 及状态
# ═════════════════════════════════════════════════════════════════

class SubagentListTool(Tool):
    """List all subagents and their current status."""

    def __init__(self, manager: SubagentManager):
        self._manager = manager

    @property
    def name(self) -> str:
        return "subagent_list"

    @property
    def description(self) -> str:
        return (
            "List all subagents (active and recent) with their current status, "
            "model, iteration progress, report mode, and recent tool usage."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def execute(self, **kwargs: Any) -> str:
        agents = self._manager.list_agents()
        if not agents:
            return "No subagents (active or recent)."
        return json.dumps(agents, ensure_ascii=False, indent=2)


# ═════════════════════════════════════════════════════════════════
# subagent_control — 暂停/恢复/取消子 agent
# ═════════════════════════════════════════════════════════════════

class SubagentControlTool(Tool):
    """Control a subagent: cancel, pause, or resume."""

    def __init__(self, manager: SubagentManager):
        self._manager = manager

    @property
    def name(self) -> str:
        return "subagent_control"

    @property
    def description(self) -> str:
        return (
            "Control a running subagent: "
            "'cancel' — stop immediately; "
            "'pause' — pause at next iteration boundary; "
            "'resume' — resume a paused subagent."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "string",
                    "description": "The subagent's task ID",
                },
                "action": {
                    "type": "string",
                    "enum": ["cancel", "pause", "resume"],
                    "description": "The action to perform",
                },
            },
            "required": ["task_id", "action"],
        }

    async def execute(self, task_id: str, action: str, **kwargs: Any) -> str:
        if action == "cancel":
            return await self._manager.cancel(task_id)
        elif action == "pause":
            return await self._manager.pause(task_id)
        elif action == "resume":
            return await self._manager.resume(task_id)
        return f"Unknown action: {action}"
