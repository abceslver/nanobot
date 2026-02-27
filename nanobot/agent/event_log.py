"""EventLog — per-session 环形事件缓冲区 (Layer 1 快路径)。

子 agent 的进度和结果通过 event_log 追加到父 agent 的会话中，
写入 O(1) 且不持有任何锁，不触发父 agent 的 LLM 轮。
父 agent 在下一轮的 context 构建时 drain 未读事件。

参见: agent_network_protocol_v1.md §2.4
"""

from __future__ import annotations

import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


# ── 事件类型 ──────────────────────────────────────────────────────

class RealtimeEventType(str, Enum):
    """快路径事件类型。"""
    PROGRESS = "progress"               # 子 agent 中间进度
    RESULT = "result"                   # 子 agent 最终结果
    ERROR = "error"                     # 子 agent 错误
    TOOL_CALL = "tool_call"             # 子 agent 工具调用记录
    PEER_MESSAGE = "peer_message"       # agent 间点对点
    DIRECT_USER_PUSH = "direct_push"    # 子 agent 直推用户 (仅审计记录)
    HEARTBEAT = "heartbeat"             # 存活探针


# ── 事件数据 ──────────────────────────────────────────────────────

@dataclass
class RealtimeEvent:
    """快路径事件 — 写入即可见，不触发 LLM。"""
    event_id: str
    timestamp: datetime
    source_id: str          # 发送方 agent_id (或 "parent")
    source_label: str       # 人类可读标签
    event_type: RealtimeEventType
    payload: str            # 事件内容
    metadata: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def create(
        source_id: str,
        source_label: str,
        event_type: RealtimeEventType,
        payload: str,
        **metadata: Any,
    ) -> RealtimeEvent:
        """工厂方法。"""
        return RealtimeEvent(
            event_id=uuid.uuid4().hex[:12],
            timestamp=datetime.now(),
            source_id=source_id,
            source_label=source_label,
            event_type=event_type,
            payload=payload,
            metadata=metadata,
        )


# ── EventLog 环形缓冲 ────────────────────────────────────────────

class EventLog:
    """
    Per-session 环形事件缓冲区。

    特性:
    - 写入 O(1)，不持有任何锁（不与 _processing_lock 或 per-session 锁竞争）
    - 读取 O(n)，在 context 构建时调用 drain_unread()
    - 环形缓冲自动淘汰最旧事件（maxlen）
    - 支持 Tier 0 结构化压缩（合并同源 + 丢弃心跳）
    """

    def __init__(self, max_events: int = 200):
        self._buffer: deque[RealtimeEvent] = deque(maxlen=max_events)
        self._unread_count: int = 0
        self._total_appended: int = 0  # 累计追加数（不受环形淘汰影响）

    # ── 写入 ──

    def append(self, event: RealtimeEvent) -> None:
        """O(1) 追加，不阻塞，不触发 LLM。"""
        self._buffer.append(event)
        self._unread_count = min(self._unread_count + 1, len(self._buffer))
        self._total_appended += 1

    # ── 读取 ──

    def drain_unread(self) -> list[RealtimeEvent]:
        """读取并标记所有未读事件（在 context 构建时调用）。"""
        if self._unread_count == 0:
            return []
        events = list(self._buffer)[-self._unread_count:]
        self._unread_count = 0
        return events

    def peek_recent(self, n: int = 10) -> list[RealtimeEvent]:
        """查看最近 n 条事件（不修改未读计数）。用于 /agent_info 查询。"""
        return list(self._buffer)[-n:]

    # ── 统计 ──

    @property
    def total_appended(self) -> int:
        return self._total_appended

    @property
    def unread_count(self) -> int:
        return self._unread_count

    @property
    def buffer_size(self) -> int:
        return len(self._buffer)

    def summary_needed(self, threshold: int = 50) -> bool:
        """事件日志是否需要触发总结（Tier 1 阈值）。"""
        return self._unread_count >= threshold

    # ── Tier 0 结构化压缩 ─────────────────────────────────────────

    @staticmethod
    def compress_for_context(
        events: list[RealtimeEvent],
        *,
        max_payload_len: int = 500,
        merge_same_source: bool = True,
        drop_heartbeat: bool = True,
    ) -> str:
        """
        将事件列表压缩为可注入 context 的文本。

        Tier 0 压缩规则（零 LLM 成本）:
        1. 丢弃 heartbeat 事件
        2. 合并连续同源事件为一条摘要
        3. 截断过长的 payload
        """
        if not events:
            return ""

        # 过滤
        if drop_heartbeat:
            events = [e for e in events if e.event_type != RealtimeEventType.HEARTBEAT]
        if not events:
            return ""

        # 合并连续同源
        if merge_same_source:
            events = EventLog._merge_consecutive(events)

        # 格式化
        lines: list[str] = []
        for e in events:
            payload = e.payload
            if len(payload) > max_payload_len:
                payload = payload[:max_payload_len] + "..."
            type_icon = _EVENT_ICONS.get(e.event_type, "•")
            lines.append(
                f"[{e.timestamp:%H:%M:%S}] {type_icon} {e.source_label}: {payload}"
            )

        return "\n".join(lines)

    @staticmethod
    def _merge_consecutive(events: list[RealtimeEvent]) -> list[RealtimeEvent]:
        """合并连续同源同类型事件。"""
        if len(events) <= 1:
            return events

        merged: list[RealtimeEvent] = []
        i = 0
        while i < len(events):
            current = events[i]
            # 看后续有多少连续同源同类型
            j = i + 1
            while (
                j < len(events)
                and events[j].source_id == current.source_id
                and events[j].event_type == current.event_type
            ):
                j += 1

            count = j - i
            if count == 1:
                merged.append(current)
            else:
                # 合并: 保留最后一条的 payload，标注数量
                last = events[j - 1]
                summary_event = RealtimeEvent(
                    event_id=last.event_id,
                    timestamp=last.timestamp,
                    source_id=last.source_id,
                    source_label=last.source_label,
                    event_type=last.event_type,
                    payload=f"({count}条) {last.payload}",
                    metadata={**last.metadata, "_merged_count": count},
                )
                merged.append(summary_event)
            i = j

        return merged


# ── 图标映射 ──────────────────────────────────────────────────────

_EVENT_ICONS: dict[RealtimeEventType, str] = {
    RealtimeEventType.PROGRESS: "⏳",
    RealtimeEventType.RESULT: "✅",
    RealtimeEventType.ERROR: "❌",
    RealtimeEventType.TOOL_CALL: "🔧",
    RealtimeEventType.PEER_MESSAGE: "💬",
    RealtimeEventType.DIRECT_USER_PUSH: "📤",
    RealtimeEventType.HEARTBEAT: "💓",
}
