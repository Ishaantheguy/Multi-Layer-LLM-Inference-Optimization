import re
from dataclasses import dataclass


@dataclass
class CacheDecision:
    request_type: str
    answer_type: str
    store_exact: bool
    store_semantic: bool
    semantic_lookup: bool
    reason: str


# Requests whose answers may become stale quickly.
TIME_SENSITIVE_PATTERNS = [
    r"\bcurrent\b",
    r"\bright now\b",
    r"\blatest\b",
    r"\btoday\b",
    r"\bthis week\b",
    r"\bcurrent price\b",
    r"\bstock price\b",
    r"\bweather\b",
    r"\bnews\b",
    r"\bexchange rate\b",
]

# Requests that may contain personalized or sensitive information.
PERSONAL_PATTERNS = [
    r"\bmy account\b",
    r"\bmy balance\b",
    r"\bmy transaction\b",
    r"\bmy medical\b",
    r"\bmy password\b",
    r"\bmy address\b",
    r"\bmy personal\b",
    r"\bmy history\b",
    r"\bmy portfolio\b",
]

# Topics where semantically similar prompts can have materially
# different correct answers.
HIGH_RISK_PATTERNS = [
    r"\bdiagnos",
    r"\bprescription\b",
    r"\bdosage\b",
    r"\blegal advice\b",
    r"\btax filing\b",
    r"\bexecute this code\b",
    r"\bdelete .*files?\b",
    r"\btransfer .*money\b",
]


def matches_any(text, patterns):
    return any(
        re.search(pattern, text, re.IGNORECASE)
        for pattern in patterns
    )


def classify_request(payload):
    messages = payload.get("messages", [])

    if not isinstance(messages, list) or not messages:
        return "invalid"

    # Conservative policy: only classify a single plain-text
    # user message for semantic reuse.
    if len(messages) != 1:
        return "conversation"

    message = messages[0]

    if not isinstance(message, dict):
        return "invalid"

    if message.get("role") != "user":
        return "conversation"

    query = message.get("content")

    if not isinstance(query, str) or not query.strip():
        return "non_text"

    query = query.strip()

    if matches_any(query, PERSONAL_PATTERNS):
        return "personal"

    if matches_any(query, TIME_SENSITIVE_PATTERNS):
        return "time_sensitive"

    if matches_any(query, HIGH_RISK_PATTERNS):
        return "high_risk"

    if re.search(
        r"\b(write|debug|review|explain|implement|refactor)\b"
        r".*\b(code|function|program|script|algorithm)\b",
        query,
        re.IGNORECASE,
    ):
        return "coding"

    if re.search(
        r"\b(calculate|compute|solve|evaluate)\b",
        query,
        re.IGNORECASE,
    ):
        return "calculation"

    return "general"


def classify_answer(answer):
    if not isinstance(answer, str) or not answer.strip():
        return "empty"

    answer = answer.strip()

    if re.search(
        r"\bI don't know\b|\bI cannot determine\b"
        r"|\binsufficient information\b",
        answer,
        re.IGNORECASE,
    ):
        return "uncertain"

    if re.search(
        r"\b(as of|as at|current price|latest price)\b",
        answer,
        re.IGNORECASE,
    ):
        return "potentially_stale"

    if len(answer) > 12000:
        return "long"

    if "```" in answer:
        return "code"

    return "general"


def choose_cache_policy(payload, answer):
    request_type = classify_request(payload)
    answer_type = classify_answer(answer)

    # This policy applies to ordinary, non-streaming chat
    # completions. Tool calls, streaming, and other response
    # modes need separate handling.
    if payload.get("stream", False):
        return CacheDecision(
            request_type=request_type,
            answer_type=answer_type,
            store_exact=False,
            store_semantic=False,
            semantic_lookup=False,
            reason="Streaming requests are not cacheable here.",
        )

    if payload.get("tools") or payload.get("tool_choice"):
        return CacheDecision(
            request_type=request_type,
            answer_type=answer_type,
            store_exact=False,
            store_semantic=False,
            semantic_lookup=False,
            reason="Tool-enabled requests need a separate cache policy.",
        )

    # Do not persist personalized or sensitive requests by default.
    if request_type == "personal":
        return CacheDecision(
            request_type=request_type,
            answer_type=answer_type,
            store_exact=False,
            store_semantic=False,
            semantic_lookup=False,
            reason="Personalized request: avoid persistent response caching.",
        )

    # Avoid caching requests that can become stale quickly.
    if request_type == "time_sensitive":
        return CacheDecision(
            request_type=request_type,
            answer_type=answer_type,
            store_exact=False,
            store_semantic=False,
            semantic_lookup=False,
            reason="Time-sensitive request: require freshness controls.",
        )

    # High-risk queries must not reuse responses solely because
    # embeddings are similar.
    if request_type == "high_risk":
        return CacheDecision(
            request_type=request_type,
            answer_type=answer_type,
            store_exact=False,
            store_semantic=False,
            semantic_lookup=False,
            reason="High-risk request: bypass response caches.",
        )

    # Only cache ordinary, stable responses exactly when the
    # request's full generation configuration is part of its key.
    store_exact = request_type not in {"invalid", "non_text"}

    # Semantic admission is deliberately narrower than exact
    # admission.
    store_semantic = (
        request_type == "general"
        and answer_type == "general"
        and len(answer.strip()) <= 4000
    )

    semantic_lookup = request_type == "general"

    if answer_type in {"uncertain", "potentially_stale", "empty"}:
        store_semantic = False

    return CacheDecision(
        request_type=request_type,
        answer_type=answer_type,
        store_exact=store_exact,
        store_semantic=store_semantic,
        semantic_lookup=semantic_lookup,
        reason="Applied request-type and answer-type admission rules.",
    )
