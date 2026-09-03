"""Command-line front end for ctx-squeeze.

Wires the library's two entry points - :func:`squeeze` for free text and
:func:`prune_messages` for chat transcripts - to argv, and handles the parts
that only make sense at the command line: reading a file or stdin, printing a
one-line stats summary to stderr so it never pollutes piped stdout, and an
optional ``--json`` report shape for callers that want the numbers as well as
the text.
"""

import argparse
import json
import sys

from .compactor import STRATEGIES, squeeze
from .messages import parse_messages, prune_messages

__all__ = ["main"]


def _build_parser():
    parser = argparse.ArgumentParser(
        prog="ctx-squeeze",
        description="Fit long documents and chat transcripts into an LLM context budget.",
    )
    parser.add_argument("path", help='input file to compact, or "-" to read stdin')
    parser.add_argument("--budget", type=int, required=True, help="target size in estimated tokens")
    parser.add_argument(
        "--strategy",
        default="score",
        help="comma-separated pipeline: %s (default: score)" % ", ".join(sorted(STRATEGIES)),
    )
    parser.add_argument("--head-ratio", type=float, default=0.5, help="share of the budget spent on the head in head-tail")
    parser.add_argument("--jaccard", type=float, default=0.8, help="similarity at which two segments count as duplicates")
    parser.add_argument("--shingle-size", type=int, default=5, help="words per shingle in the dedupe stage")
    parser.add_argument("--messages", action="store_true", help="treat the input as a JSON chat transcript")
    parser.add_argument("--recent-turns", type=int, default=2, help="user turns kept whole in --messages mode")
    parser.add_argument("--no-marker", action="store_true", help="omit the [N ... elided] markers")
    parser.add_argument("--stats", action="store_true", help="print a token summary to stderr")
    parser.add_argument("--json", dest="as_json", action="store_true", help="emit a JSON report instead of plain text")
    parser.add_argument("-o", "--output", default=None, help="write the result to a file instead of stdout")
    return parser


def _read_input(path):
    if path == "-":
        return sys.stdin.read()
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def _write_output(text, path):
    if not text.endswith("\n"):
        text += "\n"
    if path is None:
        sys.stdout.write(text)
    else:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text)


def _run_document(raw, args):
    result = squeeze(
        raw,
        budget=args.budget,
        strategy=args.strategy,
        head_ratio=args.head_ratio,
        jaccard_threshold=args.jaccard,
        shingle_size=args.shingle_size,
        marker=not args.no_marker,
    )

    if args.stats:
        print(
            "kept %d of %d segments | %d -> %d tokens (budget %d)"
            % (result.segments_out, result.segments_in, result.original_tokens, result.final_tokens, args.budget),
            file=sys.stderr,
        )

    if args.as_json:
        return json.dumps(
            {
                "text": result.text,
                "original_tokens": result.original_tokens,
                "final_tokens": result.final_tokens,
                "segments_in": result.segments_in,
                "segments_out": result.segments_out,
                "notes": result.notes,
            },
            indent=2,
        )
    return result.text


def _run_messages(raw, args):
    messages = parse_messages(raw)
    result = prune_messages(messages, budget=args.budget, recent_turns=args.recent_turns, marker=not args.no_marker)

    if args.stats:
        print(
            "kept %d of %d messages | %d -> %d tokens (budget %d)"
            % (result.final_count, result.original_count, result.original_tokens, result.final_tokens, args.budget),
            file=sys.stderr,
        )

    if args.as_json:
        return json.dumps(
            {
                "messages": result.to_dicts(),
                "original_tokens": result.original_tokens,
                "final_tokens": result.final_tokens,
                "original_count": result.original_count,
                "final_count": result.final_count,
                "pinned_tool_results": sorted(result.pinned_tool_results),
            },
            indent=2,
        )
    return json.dumps(result.to_dicts(), indent=2)


def main(argv=None):
    args = _build_parser().parse_args(argv)

    try:
        raw = _read_input(args.path)
    except OSError as exc:
        print("ctx-squeeze: %s" % exc, file=sys.stderr)
        return 1

    try:
        if args.messages:
            output_text = _run_messages(raw, args)
        else:
            output_text = _run_document(raw, args)
    except ValueError as exc:
        print("ctx-squeeze: %s" % exc, file=sys.stderr)
        return 1

    _write_output(output_text, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
