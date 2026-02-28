"""Subagent manager for background task execution.

升级协议:
  - 主 agent 可通过 spawn 工具自定义子 agent 的 model、tools、system_prompt、report_mode
  - 支持 4 种进度回传模式: result_only / on_error / on_tool_call / every_step
  - 主 agent 可通过 subagent_message 在子 agent 执行过程中发送纠正消息
  - 支持 pause / resume / cancel 操控子 agent
  - 子 agent 共享主 agent 的工具实例 (通过 parent_tools 筛选)
"""

from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.bus.events import InboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMProvider
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.event_log import EventLog, RealtimeEvent, RealtimeEventType


# ── 枚举 / 数据类 ────────────────────────────────────────────────

class ReportMode(str, Enum):
    """子 agent 向主 agent 回传进度的方式。"""
    RESULT_ONLY = "result_only"       # 仅最终结果
    ON_ERROR = "on_error"             # 出错时 + 最终结果
    ON_TOOL_CALL = "on_tool_call"     # 每次工具调用 + 最终结果
    EVERY_STEP = "every_step"         # 每轮 LLM 迭代 + 最终结果


class SubagentState(str, Enum):
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class SubagentInfo:
    """单个子 agent 的运行时状态。"""
    task_id: str
    task: str
    label: str
    model: str
    report_mode: ReportMode
    state: SubagentState = SubagentState.RUNNING
    iteration: int = 0
    max_iterations: int = 15
    tools_used: list[str] = field(default_factory=list)
    tool_names: list[str] = field(default_factory=list)
    origin: dict[str, str] = field(default_factory=dict)
    session_key: str | None = None
    persistent: bool = False  # True = LISTENER/PERSISTENT，不会因 LLM 回复而退出

    # 运行时消息列表 (与 _run_subagent 内的 messages 共享引用, 供 Dashboard 读取)
    messages: list[dict[str, Any]] = field(default_factory=list)

    # 主 agent -> 子 agent 的消息收件箱
    inbox: asyncio.Queue[str] = field(default_factory=lambda: asyncio.Queue())
    # 暂停控制 (set = 运行中, clear = 已暂停)
    pause_event: asyncio.Event = field(default_factory=lambda: asyncio.Event())

    def __post_init__(self) -> None:
        self.pause_event.set()  # 默认不暂停


# ── SubagentManager ──────────────────────────────────────────────

class SubagentManager:
    """管理后台子 agent 的创建、执行、通信和生命周期。"""

    # 子 agent 永远不能拥有的工具 (防止无限递归)
    _EXCLUDED_TOOLS = frozenset({
        "spawn", "subagent_message", "subagent_list", "subagent_control",
    })

    def __init__(
        self,
        provider: LLMProvider,
        workspace: Path,
        bus: MessageBus,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        parent_tools: ToolRegistry | None = None,
        event_log_resolver: Any | None = None,
    ):
        self.provider = provider
        self.workspace = workspace
        self.bus = bus
        self.default_model = model or provider.get_default_model()
        self.default_temperature = temperature
        self.default_max_tokens = max_tokens
        self.parent_tools = parent_tools  # 主 agent 的工具注册表 (共享实例)
        # event_log_resolver: 可调用对象，接受 session_key 返回 EventLog
        # 典型值: ExtendedSessionManager.get_event_log
        self._event_log_resolver = event_log_resolver

        self._running_tasks: dict[str, asyncio.Task[None]] = {}
        self._agents: dict[str, SubagentInfo] = {}
        self._session_tasks: dict[str, set[str]] = {}  # session_key -> {task_id, ...}

    def set_parent_tools(self, tools: ToolRegistry) -> None:
        """设置/更新主 agent 的工具注册表 (在工具注册完毕后调用)。"""
        self.parent_tools = tools

    def set_event_log_resolver(self, resolver: Any) -> None:
        """设置 EventLog 解析器。resolver(session_key) -> EventLog。"""
        self._event_log_resolver = resolver

    def _resolve_event_log(self, session_key: str | None) -> EventLog | None:
        """尝试解析父会话的 EventLog。"""
        if self._event_log_resolver and session_key:
            try:
                return self._event_log_resolver(session_key)
            except Exception:
                pass
        return None

    # ── spawn ─────────────────────────────────────────────────────

    async def spawn(
        self,
        task: str,
        label: str | None = None,
        model: str | None = None,
        tools: list[str] | None = None,
        system_prompt: str | None = None,
        report_mode: str = "result_only",
        max_iterations: int = 15,
        origin_channel: str = "cli",
        origin_chat_id: str = "direct",
        session_key: str | None = None,
        persistent: bool = False,
    ) -> str:
        """创建并启动一个子 agent。"""
        task_id = str(uuid.uuid4())[:8]
        display_label = label or task[:30] + ("..." if len(task) > 30 else "")
        origin = {"channel": origin_channel, "chat_id": origin_chat_id}

        # 解析 report_mode
        try:
            mode = ReportMode(report_mode)
        except ValueError:
            mode = ReportMode.RESULT_ONLY

        # 构建子 agent 的工具注册表
        tool_registry = self._build_tool_registry(tools)

        # 验证模型可用性，不可用则回退到默认模型
        resolved_model = model or self.default_model
        if model and hasattr(self.provider, 'cfg'):
            src_idx = self.provider.cfg.find_source_for_model(model)
            if src_idx is None:
                logger.warning(
                    "Subagent model '{}' not found in any enabled source, "
                    "falling back to default '{}'",
                    model, self.default_model,
                )
                resolved_model = self.default_model

        # 创建子 agent 状态
        info = SubagentInfo(
            task_id=task_id,
            task=task,
            label=display_label,
            model=resolved_model,
            report_mode=mode,
            max_iterations=max_iterations,
            tool_names=tool_registry.tool_names,
            origin=origin,
            session_key=session_key,
            persistent=persistent,
        )
        self._agents[task_id] = info

        # 启动后台任务
        bg_task = asyncio.create_task(
            self._run_subagent(info, tool_registry, system_prompt),
        )
        self._running_tasks[task_id] = bg_task
        if session_key:
            self._session_tasks.setdefault(session_key, set()).add(task_id)

        def _cleanup(_: asyncio.Task) -> None:
            # persistent agent 完成后不从跟踪表中移除（保留可查询状态）
            if not persistent:
                self._running_tasks.pop(task_id, None)
                if session_key and (ids := self._session_tasks.get(session_key)):
                    ids.discard(task_id)
                    if not ids:
                        del self._session_tasks[session_key]
            else:
                logger.info(
                    "Persistent subagent [{}] bg_task ended, keeping in registry",
                    task_id,
                )

        bg_task.add_done_callback(_cleanup)

        logger.info(
            "Spawned subagent [{}]: {} | model={} | tools={} | report={}",
            task_id, display_label, info.model,
            tool_registry.tool_names, mode.value,
        )
        return (
            f"Subagent [{display_label}] started (id: {task_id}).\n"
            f"Model: {info.model} | Tools: {', '.join(tool_registry.tool_names)} | "
            f"Report: {mode.value} | Max iterations: {max_iterations}"
        )

    # ── 消息通道 ──────────────────────────────────────────────────

    async def send_message(self, task_id: str, message: str) -> str:
        """向运行中的子 agent 发送纠正/补充消息。"""
        info = self._agents.get(task_id)
        if not info:
            return f"Error: subagent {task_id} not found"
        if info.state not in (SubagentState.RUNNING, SubagentState.PAUSED):
            return f"Error: subagent {task_id} is {info.state.value}"

        info.inbox.put_nowait(message)
        logger.info("Sent message to subagent [{}]: {}", task_id, message[:100])
        return f"Message delivered to subagent [{info.label}] (will be processed before next iteration)"

    # ── 暂停 / 恢复 / 取消 ────────────────────────────────────────

    async def pause(self, task_id: str) -> str:
        """暂停子 agent (在下一个迭代边界生效)。"""
        info = self._agents.get(task_id)
        if not info:
            return f"Error: subagent {task_id} not found"
        if info.state != SubagentState.RUNNING:
            return f"Error: subagent {task_id} is {info.state.value}, cannot pause"

        info.state = SubagentState.PAUSED
        info.pause_event.clear()
        logger.info("Paused subagent [{}] at iteration {}", task_id, info.iteration)
        return f"Subagent [{info.label}] paused at iteration {info.iteration}"

    async def resume(self, task_id: str) -> str:
        """恢复暂停的子 agent。"""
        info = self._agents.get(task_id)
        if not info:
            return f"Error: subagent {task_id} not found"
        if info.state != SubagentState.PAUSED:
            return f"Error: subagent {task_id} is {info.state.value}, cannot resume"

        info.state = SubagentState.RUNNING
        info.pause_event.set()
        logger.info("Resumed subagent [{}]", task_id)
        return f"Subagent [{info.label}] resumed"

    async def cancel(self, task_id: str) -> str:
        """取消子 agent。"""
        info = self._agents.get(task_id)
        if not info:
            return f"Error: subagent {task_id} not found"

        task = self._running_tasks.get(task_id)
        if task and not task.done():
            info.state = SubagentState.CANCELLED
            info.pause_event.set()  # 解除暂停阻塞
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
            logger.info("Cancelled subagent [{}]", task_id)
            return f"Subagent [{info.label}] cancelled"

        return f"Subagent [{info.label}] is already done ({info.state.value})"

    # ── 查询 ──────────────────────────────────────────────────────

    def update_agent(
        self,
        task_id: str,
        *,
        report_mode: str | None = None,
        model: str | None = None,
        max_iterations: int | None = None,
    ) -> str:
        """动态更新运行中子 agent 的参数。

        可更新字段: report_mode, model, max_iterations。
        下次 LLM 迭代时立即生效。
        """
        info = self._agents.get(task_id)
        if not info:
            return f"Error: subagent {task_id} not found"
        if info.state not in (SubagentState.RUNNING, SubagentState.PAUSED):
            return f"Error: subagent {task_id} is {info.state.value}, cannot update"

        changes: list[str] = []

        if report_mode is not None:
            try:
                new_mode = ReportMode(report_mode)
                old_mode = info.report_mode
                info.report_mode = new_mode
                changes.append(f"report_mode: {old_mode.value} → {new_mode.value}")
            except ValueError:
                return f"Error: invalid report_mode '{report_mode}'"

        if model is not None:
            old_model = info.model
            info.model = model
            changes.append(f"model: {old_model} → {model}")

        if max_iterations is not None:
            old_max = info.max_iterations
            info.max_iterations = max_iterations
            changes.append(f"max_iterations: {old_max} → {max_iterations}")

        if not changes:
            return f"No changes specified for subagent [{info.label}]"

        logger.info(
            "Updated subagent [{}]: {}", task_id, ", ".join(changes),
        )
        return (
            f"Subagent [{info.label}] updated: {'; '.join(changes)}"
        )

    def list_agents(self) -> list[dict[str, Any]]:
        """列出所有子 agent 及其状态。"""
        result = []
        for tid, info in self._agents.items():
            result.append({
                "task_id": tid,
                "label": info.label,
                "state": info.state.value,
                "model": info.model,
                "iteration": f"{info.iteration}/{info.max_iterations}",
                "report_mode": info.report_mode.value,
                "tools": info.tool_names,
                "recent_tools_used": info.tools_used[-5:],
                "has_pending_messages": not info.inbox.empty(),
            })
        return result

    def get_running_count(self) -> int:
        """返回当前活跃 (运行中+暂停) 的子 agent 数。"""
        return sum(
            1 for info in self._agents.values()
            if info.state in (SubagentState.RUNNING, SubagentState.PAUSED)
        )

    async def cancel_by_session(self, session_key: str) -> int:
        """取消指定 session 的所有子 agent。"""
        task_ids = list(self._session_tasks.get(session_key, []))
        count = 0
        for tid in task_ids:
            result = await self.cancel(tid)
            if "cancelled" in result.lower():
                count += 1
        return count

    # ── 核心执行循环 ──────────────────────────────────────────────

    async def _run_subagent(
        self,
        info: SubagentInfo,
        tools: ToolRegistry,
        custom_system_prompt: str | None,
    ) -> None:
        """子 agent 的主执行循环。"""
        task_id = info.task_id
        logger.info("Subagent [{}] starting: {}", task_id, info.label)

        try:
            # 构建初始消息
            system_prompt = custom_system_prompt or self._build_subagent_prompt(
                info.task, tools,
            )
            messages: list[dict[str, Any]] = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": info.task},
            ]
            # 将消息列表引用存到 info 上, 供外部 (Dashboard) 读取
            info.messages = messages

            final_result: str | None = None

            while info.iteration < info.max_iterations:
                # ── 暂停检查 ──
                await info.pause_event.wait()

                # ── 取消检查 ──
                if info.state == SubagentState.CANCELLED:
                    break

                # ── 处理收件箱 (主 agent 的纠正消息) ──
                corrections_injected = False
                while not info.inbox.empty():
                    try:
                        correction = info.inbox.get_nowait()
                        messages.append({
                            "role": "user",
                            "content": f"[Correction from supervisor]: {correction}",
                        })
                        corrections_injected = True
                        logger.info(
                            "Subagent [{}] received correction: {}",
                            task_id, correction[:100],
                        )
                    except asyncio.QueueEmpty:
                        break

                info.iteration += 1

                # ── LLM 调用 ──
                tool_defs = tools.get_definitions() if tools.tool_names else None
                response = await self.provider.chat(
                    messages=messages,
                    tools=tool_defs,
                    model=info.model,
                    temperature=self.default_temperature,
                    max_tokens=self.default_max_tokens,
                )

                if response.has_tool_calls:
                    # 构建 assistant 消息
                    tool_call_dicts = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(
                                    tc.arguments, ensure_ascii=False,
                                ),
                            },
                        }
                        for tc in response.tool_calls
                    ]
                    assistant_msg: dict[str, Any] = {
                        "role": "assistant",
                        "content": response.content or "",
                        "tool_calls": tool_call_dicts,
                    }
                    if response.reasoning_content:
                        assistant_msg["reasoning_content"] = response.reasoning_content
                    messages.append(assistant_msg)

                    # 执行工具并收集结果
                    tool_results: list[tuple[str, str]] = []
                    for tc in response.tool_calls:
                        args_str = json.dumps(tc.arguments, ensure_ascii=False)
                        logger.debug(
                            "Subagent [{}] executing: {}({})",
                            task_id, tc.name, args_str[:200],
                        )
                        result = await tools.execute(tc.name, tc.arguments)
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "name": tc.name,
                            "content": result,
                        })
                        info.tools_used.append(tc.name)
                        tool_results.append((tc.name, result))

                    # 进度回传: on_tool_call / every_step
                    if info.report_mode in (
                        ReportMode.ON_TOOL_CALL,
                        ReportMode.EVERY_STEP,
                    ):
                        parts = []
                        for name, res in tool_results:
                            preview = res[:200] + ("..." if len(res) > 200 else "")
                            parts.append(f"  {name}: {preview}")
                        progress_text = (
                            f"[Step {info.iteration}/{info.max_iterations}] "
                            f"Tool calls:\n" + "\n".join(parts)
                        )
                        await self._report_progress(info, progress_text)

                elif response.content:
                    # 将 assistant 回复追加到 messages（供 Dashboard 展示）
                    assistant_reply: dict[str, Any] = {
                        "role": "assistant",
                        "content": response.content,
                    }
                    if response.reasoning_content:
                        assistant_reply["reasoning_content"] = response.reasoning_content
                    messages.append(assistant_reply)

                    # 进度回传: every_step
                    if info.report_mode == ReportMode.EVERY_STEP:
                        preview = response.content[:300]
                        if len(response.content) > 300:
                            preview += "..."
                        await self._report_progress(
                            info,
                            f"[Step {info.iteration}/{info.max_iterations}] "
                            f"Thinking: {preview}",
                        )

                    if info.persistent:
                        # ── persistent agent: 不退出，将结果推送给父 agent 后等待新指令 ──
                        await self._deliver_persistent_result(
                            info, response.content,
                        )
                        logger.info(
                            "Persistent subagent [{}] entering idle wait",
                            task_id,
                        )
                        # 阻塞等待 inbox 消息（取消时会抛出 CancelledError）
                        try:
                            msg = await info.inbox.get()
                        except asyncio.CancelledError:
                            raise
                        messages.append({
                            "role": "user",
                            "content": f"[New instruction]: {msg}",
                        })
                        logger.info(
                            "Persistent subagent [{}] woke up: {}",
                            task_id, msg[:100],
                        )
                        # 重置迭代计数器，防止 idle 唤醒后被 max_iterations 限制
                        info.iteration = 0
                        continue
                    else:
                        # 普通 agent: 无工具调用 = 最终回答
                        final_result = response.content
                        break
                else:
                    if info.persistent:
                        # persistent agent 空响应也不退出
                        logger.warning(
                            "Persistent subagent [{}] got empty response, "
                            "waiting for inbox",
                            task_id,
                        )
                        try:
                            msg = await info.inbox.get()
                        except asyncio.CancelledError:
                            raise
                        messages.append({
                            "role": "user",
                            "content": f"[New instruction]: {msg}",
                        })
                        info.iteration = 0
                        continue
                    else:
                        # 空响应
                        final_result = (
                            "Task completed but no final response was generated."
                        )
                        break

            # ── 循环结束 ──
            if info.state == SubagentState.CANCELLED:
                logger.info("Subagent [{}] was cancelled", task_id)
                return

            if final_result is None:
                final_result = (
                    f"Reached max iterations ({info.max_iterations}). "
                    f"Last tools used: {info.tools_used[-3:]}"
                )

            info.state = SubagentState.COMPLETED
            logger.info("Subagent [{}] completed successfully", task_id)
            await self._announce_result(info, final_result, "ok")

        except asyncio.CancelledError:
            info.state = SubagentState.CANCELLED
            logger.info("Subagent [{}] cancelled", task_id)
        except Exception as e:
            info.state = SubagentState.FAILED
            error_msg = f"Error: {e}"
            logger.error("Subagent [{}] failed: {}", task_id, e)

            # on_error / on_tool_call / every_step 都回传错误
            if info.report_mode != ReportMode.RESULT_ONLY:
                await self._report_progress(
                    info, f"⚠️ Error encountered: {error_msg}",
                )

            await self._announce_result(info, error_msg, "error")

    # ── 持久 agent 结果推送 ───────────────────────────────────────

    async def _deliver_persistent_result(
        self,
        info: SubagentInfo,
        content: str,
    ) -> None:
        """持久 agent 产出 content-only 响应时，将结果推送给父 agent。

        与 _announce_result 类似，但不标记 agent 为 completed。
        同时写入 EventLog 并通过 bus 发送 wake 消息触发父 agent LLM 轮。
        """
        parent_session_key = (
            f"{info.origin['channel']}:{info.origin['chat_id']}"
            if info.origin else None
        )

        result_text = (
            f"[Persistent agent '{info.label}' produced a result]\n\n"
            f"{content}"
        )

        # 写入 EventLog (RESULT 类型)
        event_log = self._resolve_event_log(parent_session_key)
        if event_log is not None:
            event_log.append(RealtimeEvent.create(
                source_id=info.task_id,
                source_label=info.label,
                event_type=RealtimeEventType.RESULT,
                payload=result_text,
                task_id=info.task_id,
                iteration=info.iteration,
                max_iterations=info.max_iterations,
                status="delivering",
            ))

        # wake: 注入 InboundMessage 触发父 agent LLM 轮
        wake_content = (
            f"[Persistent agent '{info.label}' has produced a result]\n\n"
            f"Result:\n{content}\n\n"
            f"This agent is still running and waiting for new instructions. "
            f"Summarize the result naturally for the user."
        )
        msg = InboundMessage(
            channel="system",
            sender_id="subagent",
            chat_id=f"{info.origin['channel']}:{info.origin['chat_id']}",
            content=wake_content,
            metadata={
                "_subagent_result": True,
                "_persistent_delivery": True,
                "task_id": info.task_id,
            },
            session_key_override=parent_session_key,
        )
        await self.bus.publish_inbound(msg)
        logger.info(
            "Persistent subagent [{}] delivered result to parent (session: {})",
            info.task_id, parent_session_key,
        )

    # ── 进度回传 ──────────────────────────────────────────────────

    async def _report_progress(self, info: SubagentInfo, message: str) -> None:
        """向父 agent 发送进度通知。

        优先走 EventLog 快路径（O(1)，不触发 LLM）；
        若 EventLog 不可用则回退到 bus.publish_inbound()。
        """
        parent_session_key = (
            f"{info.origin['channel']}:{info.origin['chat_id']}"
            if info.origin else None
        )

        # 快路径: EventLog
        event_log = self._resolve_event_log(parent_session_key)
        if event_log is not None:
            event_log.append(RealtimeEvent.create(
                source_id=info.task_id,
                source_label=info.label,
                event_type=RealtimeEventType.PROGRESS,
                payload=message,
                task_id=info.task_id,
                iteration=info.iteration,
                max_iterations=info.max_iterations,
            ))
            logger.debug(
                "Subagent [{}] progress → event_log (session: {})",
                info.task_id, parent_session_key,
            )
            return

        # 回退: bus.publish_inbound()
        content = (
            f"[Subagent '{info.label}' progress \u2014 "
            f"iter {info.iteration}/{info.max_iterations}]\n\n"
            f"{message}\n\n"
            f"Use subagent_message(task_id=\"{info.task_id}\", message=\"...\") "
            f"to send corrections, or subagent_control to pause/cancel."
        )
        msg = InboundMessage(
            channel="system",
            sender_id="subagent_progress",
            chat_id=f"{info.origin['channel']}:{info.origin['chat_id']}",
            content=content,
            metadata={"_subagent_progress": True, "task_id": info.task_id},
            session_key_override=parent_session_key,
        )
        await self.bus.publish_inbound(msg)

    async def _announce_result(
        self,
        info: SubagentInfo,
        result: str,
        status: str,
    ) -> None:
        """子 agent 完成后向父 agent 发送最终结果。

        默认走 EventLog 快路径 + wake（触发父 agent LLM 轮）。
        若 EventLog 不可用则回退到 bus.publish_inbound()。
        """
        status_text = "completed successfully" if status == "ok" else "failed"
        parent_session_key = (
            f"{info.origin['channel']}:{info.origin['chat_id']}"
            if info.origin else None
        )

        announce_content = (
            f"[Subagent '{info.label}' {status_text}]\n\n"
            f"Task: {info.task}\n\n"
            f"Result:\n{result}"
        )

        # 快路径: EventLog (记录详细结果)
        event_log = self._resolve_event_log(parent_session_key)
        if event_log is not None:
            event_type = (
                RealtimeEventType.RESULT if status == "ok"
                else RealtimeEventType.ERROR
            )
            event_log.append(RealtimeEvent.create(
                source_id=info.task_id,
                source_label=info.label,
                event_type=event_type,
                payload=announce_content,
                task_id=info.task_id,
                status=status,
            ))
            logger.debug(
                "Subagent [{}] result → event_log (session: {})",
                info.task_id, parent_session_key,
            )

        # wake: 注入 InboundMessage 触发父 agent LLM 轮
        wake_content = (
            f"[Subagent '{info.label}' {status_text}]\n\n"
            f"Result:\n{result}\n\n"
            f"Summarize this naturally for the user. Keep it brief (1-2 sentences). "
            f"Do not mention technical details like 'subagent' or task IDs."
        )
        msg = InboundMessage(
            channel="system",
            sender_id="subagent",
            chat_id=f"{info.origin['channel']}:{info.origin['chat_id']}",
            content=wake_content,
            metadata={"_subagent_result": True, "task_id": info.task_id},
            session_key_override=parent_session_key,
        )
        await self.bus.publish_inbound(msg)
        logger.debug(
            "Subagent [{}] announced result to {} (session: {})",
            info.task_id, parent_session_key, parent_session_key,
        )

    # ── 工具注册表构建 ────────────────────────────────────────────

    def _build_tool_registry(
        self, requested_tools: list[str] | None,
    ) -> ToolRegistry:
        """为子 agent 构建筛选后的工具注册表。"""
        registry = ToolRegistry()

        if self.parent_tools is None:
            return registry

        if requested_tools is not None:
            # 仅包含明确请求的工具 (排除管理类工具)
            for name in requested_tools:
                if name in self._EXCLUDED_TOOLS:
                    logger.warning(
                        "Subagent cannot use excluded tool: {}", name,
                    )
                    continue
                tool = self.parent_tools.get(name)
                if tool:
                    registry.register(tool)
                else:
                    logger.warning("Subagent requested unknown tool: {}", name)
        else:
            # 包含所有父工具 (排除管理类)
            for name in self.parent_tools.tool_names:
                if name in self._EXCLUDED_TOOLS:
                    continue
                tool = self.parent_tools.get(name)
                if tool:
                    registry.register(tool)

        return registry

    # ── 默认系统提示 ──────────────────────────────────────────────

    def _build_subagent_prompt(self, task: str, tools: ToolRegistry) -> str:
        """为子 agent 生成默认系统提示。"""
        from datetime import datetime
        import time as _time

        now = datetime.now().strftime("%Y-%m-%d %H:%M (%A)")
        tz = _time.strftime("%Z") or "UTC"
        tool_list = ", ".join(tools.tool_names) if tools.tool_names else "none"

        return f"""# Subagent

## Current Time
{now} ({tz})

You are a subagent spawned by the main agent to complete a specific task.

## Available Tools
{tool_list}

## Rules
1. Stay focused — complete only the assigned task, nothing else
2. Your final response will be reported back to the main agent
3. If you receive a "[Correction from supervisor]" message, adjust your approach accordingly
4. Be concise but informative in your findings
5. If a tool fails, try alternative approaches before giving up

## Workspace
Your workspace is at: {self.workspace}

When you have completed the task, provide a clear summary of your findings or actions."""
