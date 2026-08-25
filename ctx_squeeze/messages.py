"""Structural pruning for chat transcripts.

Unlike a document, a transcript has structure that carries meaning beyond its
text: a system prompt sets ground rules for every later turn, a tool call is
meaningless without its result (and vice versa), and the last few turns are
usually what the model needs to act on right now. Token-budget selection that
ignores this shape produces transcripts that read fine but no longer make
sense - a tool result with no call in sight, or a system prompt dropped in
favor of a mid-conversation aside.

``parse_messages`` accepts the OpenAI chat shape (``content`` is a string,
tool calls live in a ``tool_calls`` list, tool results are ``role: "tool"``
messages carrying a ``tool_call_id``) and the Anthropic content-block shape
(``content`` is a list of ``{"type": ...}`` blocks, including ``tool_use``
and ``tool_result``). Both normalize to the same flattened text per message,
so the rest of the pipeline never has to care which one it got.
"""

import json

from .tokens import estimate_tokens

__all__ = ["Message", "PruneResult", "parse_messages", "prune_messages"]


class Message(object):
    """One chat message, normalized to a flat ``role``/``content`` pair.

    ``calls`` and ``results`` hold the tool-call ids this message issues or
    resolves, so pruning can keep a call and its result together even when
    they land on opposite sides of the budget cut.
    """

    __slots__ = ("role", "content", "calls", "results", "index", "tokens")

    def __init__(self, role, content, calls=None, results=None, index=0):
        self.role = role
        self.content = content
        self.calls = list(calls) if calls else []
        self.results = list(results) if results else []
        self.index = index
        self.tokens = estimate_tokens(content)

    def to_dict(self):
        return {"role": self.role, "content": self.content}

    def __repr__(self):
        return "Message(role=%r, index=%d, tokens=%d)" % (self.role, self.index, self.tokens)

    def __eq__(self, other):
        if not isinstance(other, Message):
            return NotImplemented
        return (
            self.role == other.role
            and self.content == other.content
            and self.calls == other.calls
            and self.results == other.results
            and self.index == other.index
        )

    def __hash__(self):
        return hash((self.role, self.content, self.index))


class PruneResult(object):
    """The outcome of :func:`prune_messages`."""

    __slots__ = (
        "messages",
        "pinned_tool_results",
        "original_tokens",
        "final_tokens",
        "original_count",
        "final_count",
    )

    def __init__(self, messages, pinned_tool_results, original_tokens, final_tokens, original_count, final_count):
        self.messages = messages
        self.pinned_tool_results = pinned_tool_results
        self.original_tokens = original_tokens
        self.final_tokens = final_tokens
        self.original_count = original_count
        self.final_count = final_count

    def to_dicts(self):
        """Render :attr:`messages` (including elision markers) as plain dicts."""
        return [message.to_dict() for message in self.messages]

    def __repr__(self):
        return "PruneResult(kept=%d of %d, tokens=%d -> %d)" % (
            self.final_count,
            self.original_count,
            self.original_tokens,
            self.final_tokens,
        )


def _format_tool_call(tool_call):
    function = tool_call.get("function") or {}
    name = function.get("name", "")
    arguments = function.get("arguments", "")
    return "[tool_call: %s(%s)]" % (name, arguments)


def _append_text(content, extra):
    if not extra:
        return content
    return (content + "\n" + extra) if content else extra


def _flatten_blocks(blocks, calls, results):
    parts = []
    for block in blocks:
        if not isinstance(block, dict):
            parts.append(str(block))
            continue
        block_type = block.get("type")
        if block_type == "text":
            parts.append(block.get("text", ""))
        elif block_type == "tool_use":
            call_id = block.get("id")
            if call_id:
                calls.append(call_id)
            parts.append(
                "[tool_call: %s(%s)]"
                % (block.get("name", ""), json.dumps(block.get("input", {}), sort_keys=True))
            )
        elif block_type == "tool_result":
            tool_use_id = block.get("tool_use_id")
            if tool_use_id:
                results.append(tool_use_id)
            inner = block.get("content")
            if isinstance(inner, list):
                parts.append(_flatten_blocks(inner, [], []))
            elif inner is not None:
                parts.append(str(inner))
        elif "text" in block:
            parts.append(str(block["text"]))
    return "\n".join(part for part in parts if part)


def _parse_one(raw, index):
    if not isinstance(raw, dict):
        raise ValueError("each message must be a JSON object")

    role = raw.get("role", "")
    content_field = raw.get("content")
    calls = []
    results = []

    if isinstance(content_field, list):
        content = _flatten_blocks(content_field, calls, results)
    else:
        if content_field is None:
            content = ""
        elif isinstance(content_field, str):
            content = content_field
        else:
            content = str(content_field)
        for tool_call in raw.get("tool_calls") or []:
            call_id = tool_call.get("id")
            if call_id:
                calls.append(call_id)
            content = _append_text(content, _format_tool_call(tool_call))
        if role == "tool":
            tool_call_id = raw.get("tool_call_id")
            if tool_call_id:
                results.append(tool_call_id)

    return Message(role=role, content=content, calls=calls, results=results, index=index)


def parse_messages(history):
    """Parse a chat transcript into a list of :class:`Message`.

    ``history`` is a JSON array (either already parsed into a list, or a raw
    JSON string/bytes) of messages in the OpenAI or Anthropic shape.
    """
    if isinstance(history, (str, bytes)):
        history = json.loads(history)
    if not isinstance(history, list):
        raise ValueError("expected a JSON array of chat messages")
    return [_parse_one(raw, index) for index, raw in enumerate(history)]


def _recent_window_start(messages, recent_turns):
    """Return the index where the last ``recent_turns`` user turns begin."""
    if recent_turns <= 0:
        return len(messages)
    user_indices = [i for i, message in enumerate(messages) if message.role == "user"]
    if not user_indices or len(user_indices) <= recent_turns:
        return 0
    return user_indices[-recent_turns]


def _elision_marker(count):
    content = "[%d earlier message%s elided]" % (count, "" if count == 1 else "s")
    return Message(role="system", content=content, index=-1)


def prune_messages(messages, budget, recent_turns=2, marker=True):
    """Prune ``messages`` to fit ``budget`` estimated tokens where possible.

    Three rules take priority over the raw token count: every system message
    is kept, the last ``recent_turns`` user turns (and everything from the
    first of them onward) are kept whole, and a tool call is never separated
    from its result - keeping one pulls the other back in even if it falls
    outside the recent window. Those rules can keep the result over budget;
    there is no fallback that would violate them just to hit a number.

    Messages dropped from a contiguous run are replaced by a single system
    message noting how many were elided, unless ``marker`` is False.
    """
    original_tokens = sum(message.tokens for message in messages)
    original_count = len(messages)

    if not messages or original_tokens <= budget:
        return PruneResult(
            messages=list(messages),
            pinned_tool_results=set(),
            original_tokens=original_tokens,
            final_tokens=original_tokens,
            original_count=original_count,
            final_count=original_count,
        )

    recent_start = _recent_window_start(messages, recent_turns)
    keep = [message.role == "system" or i >= recent_start for i, message in enumerate(messages)]

    call_owner = {}
    result_owner = {}
    for i, message in enumerate(messages):
        for call_id in message.calls:
            call_owner[call_id] = i
        for result_id in message.results:
            result_owner[result_id] = i

    pinned = set()
    changed = True
    while changed:
        changed = False
        for i, message in enumerate(messages):
            if not keep[i]:
                continue
            for call_id in message.calls:
                j = result_owner.get(call_id)
                if j is not None and not keep[j]:
                    keep[j] = True
                    pinned.add(call_id)
                    changed = True
            for result_id in message.results:
                j = call_owner.get(result_id)
                if j is not None and not keep[j]:
                    keep[j] = True
                    pinned.add(result_id)
                    changed = True

    result_messages = []
    elided_run = 0
    for i, message in enumerate(messages):
        if keep[i]:
            if elided_run and marker:
                result_messages.append(_elision_marker(elided_run))
            elided_run = 0
            result_messages.append(message)
        else:
            elided_run += 1
    if elided_run and marker:
        result_messages.append(_elision_marker(elided_run))

    return PruneResult(
        messages=result_messages,
        pinned_tool_results=pinned,
        original_tokens=original_tokens,
        final_tokens=sum(message.tokens for message in result_messages),
        original_count=original_count,
        final_count=sum(1 for kept in keep if kept),
    )
