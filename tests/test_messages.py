import json

from ctx_squeeze.messages import Message, parse_messages, prune_messages


def make_message(role, content, calls=None, results=None, index=0):
    return Message(role=role, content=content, calls=calls, results=results, index=index)


# -- Message -------------------------------------------------------------


def test_message_to_dict_drops_calls_and_results():
    message = make_message("assistant", "hello", calls=["call1"])
    assert message.to_dict() == {"role": "assistant", "content": "hello"}


def test_message_equality_ignores_token_count():
    a = make_message("user", "hi there", index=2)
    b = make_message("user", "hi there", index=2)
    assert a == b
    assert a != make_message("user", "hi there", index=3)


# -- parse_messages: OpenAI shape -----------------------------------------


def test_parse_messages_plain_string_content():
    parsed = parse_messages([{"role": "user", "content": "hello"}])
    assert len(parsed) == 1
    assert parsed[0].role == "user"
    assert parsed[0].content == "hello"
    assert parsed[0].index == 0


def test_parse_messages_none_content_becomes_empty_string():
    parsed = parse_messages([{"role": "assistant", "content": None}])
    assert parsed[0].content == ""


def test_parse_messages_non_string_content_is_stringified():
    parsed = parse_messages([{"role": "assistant", "content": 42}])
    assert parsed[0].content == "42"


def test_parse_messages_tool_calls_appended_to_content():
    raw = {
        "role": "assistant",
        "content": "Let me check.",
        "tool_calls": [{"id": "call1", "function": {"name": "read_file", "arguments": '{"path": "a.txt"}'}}],
    }
    parsed = parse_messages([raw])[0]
    assert parsed.content == 'Let me check.\n[tool_call: read_file({"path": "a.txt"})]'
    assert parsed.calls == ["call1"]


def test_parse_messages_tool_calls_with_no_content_skip_leading_newline():
    raw = {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": "call1", "function": {"name": "search", "arguments": "{}"}}],
    }
    parsed = parse_messages([raw])[0]
    assert parsed.content == "[tool_call: search({})]"


def test_parse_messages_tool_role_records_result_id():
    raw = {"role": "tool", "tool_call_id": "call1", "content": "42"}
    parsed = parse_messages([raw])[0]
    assert parsed.results == ["call1"]
    assert parsed.content == "42"


def test_parse_messages_tool_role_without_call_id_records_no_result():
    raw = {"role": "tool", "content": "42"}
    parsed = parse_messages([raw])[0]
    assert parsed.results == []


# -- parse_messages: Anthropic content-block shape ------------------------


def test_parse_messages_anthropic_text_blocks_join_with_newline():
    raw = {"role": "assistant", "content": [{"type": "text", "text": "first"}, {"type": "text", "text": "second"}]}
    parsed = parse_messages([raw])[0]
    assert parsed.content == "first\nsecond"


def test_parse_messages_anthropic_tool_use_block_records_call_and_renders_input():
    raw = {
        "role": "assistant",
        "content": [{"type": "tool_use", "id": "call9", "name": "search", "input": {"q": "foo"}}],
    }
    parsed = parse_messages([raw])[0]
    assert parsed.calls == ["call9"]
    assert parsed.content == '[tool_call: search({"q": "foo"})]'


def test_parse_messages_anthropic_tool_result_block_records_result_id():
    raw = {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call9", "content": "result text"}]}
    parsed = parse_messages([raw])[0]
    assert parsed.results == ["call9"]
    assert parsed.content == "result text"


def test_parse_messages_anthropic_tool_result_with_nested_block_content():
    raw = {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "call9",
                "content": [{"type": "text", "text": "nested result"}],
            }
        ],
    }
    parsed = parse_messages([raw])[0]
    assert parsed.content == "nested result"


def test_parse_messages_anthropic_block_without_type_falls_back_to_text_key():
    raw = {"role": "assistant", "content": [{"text": "raw fallback"}]}
    parsed = parse_messages([raw])[0]
    assert parsed.content == "raw fallback"


def test_parse_messages_anthropic_non_dict_block_is_stringified():
    raw = {"role": "assistant", "content": ["plain string block"]}
    parsed = parse_messages([raw])[0]
    assert parsed.content == "plain string block"


# -- parse_messages: input handling ---------------------------------------


def test_parse_messages_accepts_json_string():
    history = [{"role": "user", "content": "hi"}]
    assert parse_messages(json.dumps(history)) == parse_messages(history)


def test_parse_messages_assigns_sequential_index():
    history = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}, {"role": "user", "content": "c"}]
    parsed = parse_messages(history)
    assert [message.index for message in parsed] == [0, 1, 2]


def test_parse_messages_rejects_non_list_top_level():
    try:
        parse_messages({"role": "user", "content": "hi"})
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def test_parse_messages_rejects_non_dict_entry():
    try:
        parse_messages(["not a dict"])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


# -- prune_messages: budget short-circuit ---------------------------------


def test_prune_messages_empty_list():
    result = prune_messages([], budget=100)
    assert result.messages == []
    assert result.original_count == 0
    assert result.final_count == 0


def test_prune_messages_under_budget_returns_all_messages_unchanged():
    messages = [make_message("system", "rules", index=0), make_message("user", "hi", index=1)]
    result = prune_messages(messages, budget=10_000)
    assert result.messages == messages
    assert result.original_tokens == result.final_tokens
    assert result.pinned_tool_results == set()


# -- prune_messages: structural rules --------------------------------------


def test_prune_messages_always_keeps_system_messages():
    messages = [
        make_message("system", " ".join(["system rule"] * 30), index=0),
        make_message("user", " ".join(["filler"] * 30), index=1),
        make_message("user", " ".join(["most recent question"] * 30), index=2),
    ]
    result = prune_messages(messages, budget=1, recent_turns=1)
    assert messages[0] in result.messages


def test_prune_messages_keeps_recent_user_turns_and_everything_after():
    messages = [
        make_message("system", "rules", index=0),
        make_message("user", " ".join(["turn one"] * 20), index=1),
        make_message("assistant", " ".join(["reply one"] * 20), index=2),
        make_message("user", " ".join(["turn two, the most recent question"] * 20), index=3),
        make_message("assistant", " ".join(["reply two"] * 20), index=4),
    ]
    result = prune_messages(messages, budget=1, recent_turns=1)
    kept = result.messages
    assert messages[3] in kept and messages[4] in kept
    assert messages[1] not in kept and messages[2] not in kept


def test_prune_messages_recent_turns_covering_all_user_turns_keeps_everything():
    messages = [
        make_message("user", " ".join(["turn one"] * 20), index=0),
        make_message("user", " ".join(["turn two"] * 20), index=1),
    ]
    result = prune_messages(messages, budget=1, recent_turns=5)
    assert result.messages == messages


def test_prune_messages_pins_tool_call_whose_result_is_kept():
    messages = [
        make_message("system", " ".join(["system rule word"] * 10), index=0),
        make_message("user", " ".join(["turn one filler"] * 10), index=1),
        make_message("assistant", " ".join(["issues call A filler"] * 10), calls=["callA"], index=2),
        make_message("assistant", " ".join(["other filler text"] * 10), index=3),
        make_message("user", " ".join(["turn two most recent"] * 10), index=4),
        make_message("tool", " ".join(["result for call A"] * 10), results=["callA"], index=5),
    ]
    result = prune_messages(messages, budget=1, recent_turns=1)
    assert "callA" in result.pinned_tool_results
    assert messages[2] in result.messages  # the call is pulled back in
    assert messages[1] not in result.messages
    assert messages[3] not in result.messages


def test_prune_messages_pins_tool_result_whose_call_is_kept():
    messages = [
        make_message("system", " ".join(["system rule word"] * 10), index=0),
        make_message("user", " ".join(["turn one filler"] * 10), index=1),
        make_message("tool", " ".join(["orphan result filler"] * 10), results=["callB"], index=2),
        make_message("assistant", " ".join(["other filler text"] * 10), index=3),
        make_message("user", " ".join(["turn two most recent"] * 10), index=4),
        make_message("assistant", " ".join(["issues call B filler"] * 10), calls=["callB"], index=5),
    ]
    result = prune_messages(messages, budget=1, recent_turns=1)
    assert "callB" in result.pinned_tool_results
    assert messages[2] in result.messages  # the earlier result is pulled back in
    assert messages[1] not in result.messages
    assert messages[3] not in result.messages


# -- prune_messages: elision markers ---------------------------------------


def test_prune_messages_elision_marker_reports_count_and_grammar():
    messages = [
        make_message("user", " ".join(["dropped"] * 10), index=0),
        make_message("user", " ".join(["dropped too"] * 10), index=1),
        make_message("user", " ".join(["kept, the most recent question"] * 10), index=2),
    ]
    result = prune_messages(messages, budget=1, recent_turns=1)
    markers = [message for message in result.messages if message.index == -1]
    assert len(markers) == 1
    assert markers[0].content == "[2 earlier messages elided]"
    assert markers[0].role == "system"


def test_prune_messages_singular_elision_marker():
    messages = [
        make_message("user", " ".join(["dropped"] * 10), index=0),
        make_message("user", " ".join(["kept, the most recent question"] * 10), index=1),
    ]
    result = prune_messages(messages, budget=1, recent_turns=1)
    markers = [message for message in result.messages if message.index == -1]
    assert markers[0].content == "[1 earlier message elided]"


def test_prune_messages_marker_false_omits_elision_markers():
    messages = [
        make_message("user", " ".join(["dropped"] * 10), index=0),
        make_message("user", " ".join(["kept, the most recent question"] * 10), index=1),
    ]
    result = prune_messages(messages, budget=1, recent_turns=1, marker=False)
    assert all(message.index != -1 for message in result.messages)
    assert result.messages == [messages[1]]


def test_prune_messages_final_count_excludes_markers():
    messages = [
        make_message("user", " ".join(["dropped"] * 10), index=0),
        make_message("user", " ".join(["dropped too"] * 10), index=1),
        make_message("user", " ".join(["kept, the most recent question"] * 10), index=2),
    ]
    result = prune_messages(messages, budget=1, recent_turns=1)
    assert result.final_count == 1
    assert len(result.messages) == 2  # marker + kept message


# -- PruneResult -------------------------------------------------------------


def test_prune_result_to_dicts_renders_plain_dicts():
    messages = [make_message("system", "rules", index=0), make_message("user", "hi", index=1)]
    result = prune_messages(messages, budget=10_000)
    assert result.to_dicts() == [{"role": "system", "content": "rules"}, {"role": "user", "content": "hi"}]
