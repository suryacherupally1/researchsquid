"""Session telemetry — SSE stream for real-time session events."""

import asyncio
import json
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, Optional


class SessionTelemetry:
    """
    In-memory pub/sub for session events.
    Publishes structured events that SSE endpoints consume.
    """

    def __init__(self):
        self._subscribers: Dict[str, list] = {}  # session_id -> [asyncio.Queue]

    async def subscribe(self, session_id: str) -> asyncio.Queue:
        """Subscribe to session events. Returns an asyncio.Queue."""
        if session_id not in self._subscribers:
            self._subscribers[session_id] = []
        queue: asyncio.Queue = asyncio.Queue()
        self._subscribers[session_id].append(queue)
        return queue

    async def unsubscribe(self, session_id: str, queue: asyncio.Queue):
        """Unsubscribe from session events."""
        if session_id in self._subscribers:
            self._subscribers[session_id] = [
                q for q in self._subscribers[session_id] if q is not queue
            ]

    async def publish(
        self,
        session_id: str,
        event_type: str,
        data: Dict[str, Any],
        agent_id: Optional[str] = None,
    ):
        """Publish an event to all subscribers of a session."""
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "session_id": session_id,
            "event_type": event_type,
            "agent_id": agent_id,
            "data": data,
        }
        if session_id in self._subscribers:
            dead = []
            for queue in self._subscribers[session_id]:
                try:
                    queue.put_nowait(event)
                except asyncio.QueueFull:
                    dead.append(queue)
            for q in dead:
                self._subscribers[session_id].remove(q)

    async def sse_stream(self, session_id: str) -> AsyncGenerator[str, None]:
        """Generate SSE-formatted events for a session."""
        queue = await self.subscribe(session_id)
        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=30.0)
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                except asyncio.TimeoutError:
                    # Send keepalive
                    yield f"data: {json.dumps({'event_type': 'keepalive'})}\n\n"
        finally:
            await self.unsubscribe(session_id, queue)


# Singleton
_telemetry: Optional[SessionTelemetry] = None


def get_telemetry() -> SessionTelemetry:
    global _telemetry
    if _telemetry is None:
        _telemetry = SessionTelemetry()
    return _telemetry
