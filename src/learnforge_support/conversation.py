"""Conversation management module for LearnForge Support Assistant.

Provides an in-memory bounded session history store. Tracks customer and assistant
turns up to a configured maximum per session ID without external databases.
"""

from collections import defaultdict


class ConversationStore:
    """In-memory bounded session history store."""

    def __init__(self, max_turns: int = 6) -> None:
        self.max_turns = max_turns
        self._sessions: dict[str, list[dict[str, str]]] = defaultdict(list)

    def add_turn(self, session_id: str, role: str, content: str) -> None:
        """Appends a conversation turn ('user' or 'assistant') and trims to max_turns."""
        history = self._sessions[session_id]
        history.append({"role": role, "content": content.strip()})
        if len(history) > self.max_turns:
            # Retain only the most recent max_turns
            self._sessions[session_id] = history[-self.max_turns :]

    def get_history(self, session_id: str) -> list[dict[str, str]]:
        """Returns a copy of the recent conversation history for the session."""
        return list(self._sessions.get(session_id, []))

    def clear(self, session_id: str) -> None:
        """Clears history for a specific session."""
        self._sessions.pop(session_id, None)

    def session_count(self) -> int:
        """Returns the number of active sessions in memory."""
        return len(self._sessions)
