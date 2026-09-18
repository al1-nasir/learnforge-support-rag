"""Unit tests for in-memory bounded conversation store.

Verifies turn appending, history trimming to max_turns, and multi-session isolation.
"""

from learnforge_support.conversation import ConversationStore


def test_conversation_store_turn_history():
    """Verify turns are stored sequentially in history."""
    store = ConversationStore(max_turns=4)
    store.add_turn("session-1", "user", "Hello")
    store.add_turn("session-1", "assistant", "Hi there!")

    history = store.get_history("session-1")
    assert len(history) == 2
    assert history[0] == {"role": "user", "content": "Hello"}
    assert history[1] == {"role": "assistant", "content": "Hi there!"}


def test_conversation_store_history_bounded_trimming():
    """Verify history retains only the most recent max_turns."""
    store = ConversationStore(max_turns=3)
    for i in range(5):
        store.add_turn("session-1", "user", f"Message {i}")

    history = store.get_history("session-1")
    assert len(history) == 3
    # Retained turns should be the last 3: Message 2, Message 3, Message 4
    assert history[0]["content"] == "Message 2"
    assert history[1]["content"] == "Message 3"
    assert history[2]["content"] == "Message 4"


def test_conversation_store_multi_session_isolation():
    """Verify different session IDs do not mix their histories."""
    store = ConversationStore(max_turns=4)
    store.add_turn("session-A", "user", "Message from A")
    store.add_turn("session-B", "user", "Message from B")

    hist_a = store.get_history("session-A")
    hist_b = store.get_history("session-B")

    assert len(hist_a) == 1 and hist_a[0]["content"] == "Message from A"
    assert len(hist_b) == 1 and hist_b[0]["content"] == "Message from B"

    store.clear("session-A")
    assert len(store.get_history("session-A")) == 0
    assert len(store.get_history("session-B")) == 1
