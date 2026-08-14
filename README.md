# ctx-squeeze

Fit long documents and chat transcripts into an LLM context budget, without an
API call and without a tokenizer dependency.

Context windows are large but not free: every token you carry forward is paid
for on every turn, and the middle of a long prompt is where models pay the least
attention. `ctx-squeeze` decides *what to drop* using cheap, inspectable
heuristics, and guarantees the result fits the budget you asked for.

Four things it does:

- **Estimates tokens** with a character-class heuristic, so nothing has to be
  downloaded and no vocabulary file has to match your provider.
- **Keeps code blocks intact.** A half-truncated fenced block is worse than no
  block at all, so fences are treated as atoms.
- **Removes near-duplicates.** Agent transcripts repeat: the same file read three
  times, the same traceback after each retry. Shingling plus Jaccard similarity
  catches those even when a timestamp differs.
- **Prunes chat transcripts structurally.** System messages survive, recent turns
  survive whole, and a tool result is never separated from the call that issued
  it in either direction.

Pure Python, standard library only, Python 3.9+.

## Install

```bash
pip install .
```

Or run it straight out of a checkout:

```bash
python -m ctx_squeeze.cli --budget 4000 --strategy score notes.md
```

## Usage

Given a `sample.md` with 12 paragraphs (229 estimated tokens):

```console
$ ctx-squeeze --budget 90 --strategy score --stats sample.md
kept 3 of 12 segments | 229 -> 69 tokens (budget 90)
# Nightly build postmortem

The nightly job started failing on Tuesday after the runner image was bumped.
Every run now spends eleven minutes reinstalling dependencies from scratch.

[9 segments elided]

Add an alert that fires when the nightly job runs longer than eight minutes.
```

Head-and-tail instead of keyword scoring:

```console
$ ctx-squeeze --budget 90 --strategy head-tail --stats sample.md
kept 5 of 12 segments | 229 -> 78 tokens (budget 90)
# Nightly build postmortem

[7 segments elided]

## Fix

Restore the cache step and pin the runner image to the previous minor version.
Re-run the workflow twice to confirm the cache is populated and then read.

## Follow up

Add an alert that fires when the nightly job runs longer than eight minutes.
```

Stages compose left to right, so you can strip repeats before scoring:

```bash
ctx-squeeze --budget 4000 --strategy dedupe,score --jaccard 0.7 transcript.txt
```

### Chat transcripts

`--messages` switches to message-level pruning. The input is a JSON array in the
OpenAI shape (Anthropic-style `content` block lists are accepted too):

```console
$ ctx-squeeze --messages --budget 120 --recent-turns 1 --stats chat.json
kept 7 of 11 messages | 172 -> 106 tokens (budget 120)
[
  {
    "role": "system",
    "content": "You are a careful build engineer."
  },
  {
    "role": "system",
    "content": "[4 earlier messages elided]"
  },
  {
    "role": "user",
    "content": "Can you patch the workflow file?"
  },
  ...
]
```

### Options

| Flag | Default | Meaning |
| --- | --- | --- |
| `--budget N` | required | Target size in estimated tokens |
| `--strategy S` | `score` | Comma-separated pipeline: `head-tail`, `score`, `dedupe` |
| `--head-ratio F` | `0.5` | Share of the budget spent on the head in `head-tail` |
| `--jaccard F` | `0.8` | Similarity at which two segments count as duplicates |
| `--shingle-size N` | `5` | Words per shingle in the `dedupe` stage |
| `--messages` | off | Treat the input as a JSON chat transcript |
| `--recent-turns N` | `2` | User turns kept whole in `--messages` mode |
| `--no-marker` | off | Omit the `[N segments elided]` markers |
| `--stats` | off | Print a token summary to stderr |
| `--json` | off | Emit a JSON report instead of plain text |
| `-o PATH` | stdout | Write the result to a file |

Use `-` as the input path to read standard input.

## Library API

```python
from ctx_squeeze import squeeze, prune_messages, parse_messages, estimate_tokens

result = squeeze(open("notes.md").read(), budget=4000, strategy="dedupe,score")
result.text            # the compacted document
result.original_tokens # 12480
result.final_tokens    # 3971  (never above the budget)
result.segments_out    # 41 of result.segments_in
result.notes           # ['dedupe dropped 6 near-duplicate segment(s)']

pruned = prune_messages(parse_messages(history), budget=8000, recent_turns=3)
pruned.messages              # list of Message, ready for to_dicts()
pruned.pinned_tool_results   # tool_call_ids kept from outside the recent window
```

Lower-level pieces are exported too and are useful on their own:

| Function | Purpose |
| --- | --- |
| `estimate_tokens(text)` | Heuristic token count |
| `truncate_to_tokens(text, n)` | Binary-search hard truncation |
| `split_segments(text)` | Paragraph/code-block segmentation with line numbers |
| `shingles(text, size)` / `jaccard(a, b)` | Near-duplicate primitives |
| `score_segments(segments)` | TF-IDF keyword density per segment |
| `select_by_score(segments, budget)` | Extractive selection under a budget |

## About the token estimator

`estimate_tokens` is an **estimate**, not a tokenizer. It scores runs of
characters: roughly four characters per token for words, three per token for
digit runs, one token per CJK character, half a token per newline, and 0.6 per
symbol. On English prose it lands within about 10% of a BPE tokenizer, which is
accurate enough for budgeting and fast enough to run on every paragraph.

If you need exact counts for billing, count with your provider's own endpoint
and treat the numbers here as a planning figure.

## Test

```bash
python -m pytest tests -q
```

## License

MIT, see [LICENSE](LICENSE). Copyright (c) 2026 amandah47.
