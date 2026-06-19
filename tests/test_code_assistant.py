"""Gate tests for the programming assistant (app/code_assistant.py).

All Ollama calls are mocked — no network, no model required.
"""

from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage


def _reset_graph():
    # The module caches a compiled graph; clear it so each test rebuilds with
    # its own mocked LLM.
    import app.code_assistant as ca

    ca._graph = None


def test_invoke_returns_text():
    _reset_graph()
    import app.code_assistant as ca

    fake_llm = MagicMock()
    fake_llm.invoke.return_value = AIMessage(content="```python\nprint('hi')\n```")
    with patch.object(ca, "_get_llm", return_value=fake_llm):
        out = ca.invoke_code_assistant("write hello world", thread_id="t1")
    assert "print('hi')" in out


def test_system_prompt_is_prepended():
    _reset_graph()
    import app.code_assistant as ca

    fake_llm = MagicMock()
    fake_llm.invoke.return_value = AIMessage(content="ok")
    with patch.object(ca, "_get_llm", return_value=fake_llm):
        ca.invoke_code_assistant("hello", thread_id="sysprompt")

    sent_messages = fake_llm.invoke.call_args.args[0]
    assert sent_messages[0].content == ca.SYSTEM_PROMPT
    assert sent_messages[0].type == "system"


def test_memory_persists_across_calls_same_thread():
    _reset_graph()
    import app.code_assistant as ca

    fake_llm = MagicMock()
    fake_llm.invoke.side_effect = [
        AIMessage(content="first answer"),
        AIMessage(content="second answer"),
    ]
    with patch.object(ca, "_get_llm", return_value=fake_llm):
        ca.invoke_code_assistant("first question", thread_id="mem")
        ca.invoke_code_assistant("second question", thread_id="mem")

    # On the second call the running history (system + Q1 + A1 + Q2) is sent.
    second_call_messages = fake_llm.invoke.call_args_list[1].args[0]
    contents = [m.content for m in second_call_messages]
    assert "first question" in contents
    assert "first answer" in contents
    assert "second question" in contents


def test_threads_are_isolated():
    _reset_graph()
    import app.code_assistant as ca

    fake_llm = MagicMock()
    fake_llm.invoke.return_value = AIMessage(content="answer")
    with patch.object(ca, "_get_llm", return_value=fake_llm):
        ca.invoke_code_assistant("thread A question", thread_id="A")
        ca.invoke_code_assistant("thread B question", thread_id="B")

    # Thread B must not see thread A's question.
    b_messages = [m.content for m in fake_llm.invoke.call_args_list[1].args[0]]
    assert "thread A question" not in b_messages
    assert "thread B question" in b_messages


def test_invoke_handles_llm_error_gracefully():
    _reset_graph()
    import app.code_assistant as ca

    fake_llm = MagicMock()
    fake_llm.invoke.side_effect = RuntimeError("ollama down")
    with patch.object(ca, "_get_llm", return_value=fake_llm):
        out = ca.invoke_code_assistant("hi", thread_id="err")
    assert "error" in out.lower()


def test_stream_yields_token_tuples():
    _reset_graph()
    import app.code_assistant as ca

    # Patch the compiled graph's .stream to emit message chunks.
    chunk1 = MagicMock(content="Hello")
    chunk2 = MagicMock(content=" world")
    fake_graph = MagicMock()
    fake_graph.stream.return_value = [(chunk1, {}), (chunk2, {})]
    with patch.object(ca, "get_graph", return_value=fake_graph):
        events = list(ca.stream_code_assistant("hi", thread_id="s"))

    assert ("token", "Hello") in events
    assert ("token", " world") in events


def test_stream_surfaces_error():
    _reset_graph()
    import app.code_assistant as ca

    fake_graph = MagicMock()
    fake_graph.stream.side_effect = RuntimeError("boom")
    with patch.object(ca, "get_graph", return_value=fake_graph):
        events = list(ca.stream_code_assistant("hi", thread_id="s2"))

    assert events and events[0][0] == "error"
