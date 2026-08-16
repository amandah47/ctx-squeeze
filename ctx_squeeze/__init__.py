"""Context compaction for LLM prompts, using only the standard library."""

from .compactor import STRATEGIES, SqueezeResult, squeeze
from .dedupe import dedupe_segments, find_near_duplicates, jaccard, shingles
from .messages import Message, PruneResult, parse_messages, prune_messages
from .scoring import score_segments, select_by_score
from .segments import Segment, join_segments, split_segments
from .tokens import estimate_tokens, fits_budget, truncate_to_tokens

__version__ = "0.1.0"

__all__ = [
    "STRATEGIES",
    "Message",
    "PruneResult",
    "Segment",
    "SqueezeResult",
    "__version__",
    "dedupe_segments",
    "estimate_tokens",
    "find_near_duplicates",
    "fits_budget",
    "jaccard",
    "join_segments",
    "parse_messages",
    "prune_messages",
    "score_segments",
    "select_by_score",
    "shingles",
    "split_segments",
    "squeeze",
    "truncate_to_tokens",
]
