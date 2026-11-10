import re
from dataclasses import dataclass


@dataclass
class ReferenceResult:
    needs_memory: bool
    reason: str


class ReferenceSignalDetector:

    # --------------------------------------------------
    # Pronouns
    # --------------------------------------------------

    PRONOUNS = {
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        "they",
        "them",
        "their",
        "he",
        "his",
        "him",
        "she",
        "her",
        "hers",
    }

    # --------------------------------------------------
    # Follow-up phrases
    # --------------------------------------------------

    FOLLOW_UP = [
        "tell me more",
        "explain more",
        "go on",
        "continue",
        "continue please",
        "continue further",
        "elaborate",
        "expand",
        "more details",
        "in detail",
        "can you expand",
    ]

    # --------------------------------------------------
    # Previous answer references
    # --------------------------------------------------

    PREVIOUS = [
        "previous answer",
        "last answer",
        "earlier answer",
        "your answer",
        "above answer",
        "previous response",
        "last response",
        "that answer",
        "that explanation",
        "our conversation",
    ]

    # --------------------------------------------------
    # Ordinal references
    # --------------------------------------------------

    ORDINAL = [
        "first one",
        "second one",
        "third one",
        "fourth one",
        "last one",
        "next one",
        "former",
        "latter",
    ]

    # --------------------------------------------------
    # Comparison references
    #
    # NOTE:
    # Generic words such as "difference", "better",
    # "worse", and "versus" are intentionally NOT treated
    # as memory signals by themselves.
    #
    # A standalone query such as:
    # "What is the difference between inheritance and
    # composition?"
    #
    # is a normal knowledge query.
    # --------------------------------------------------

    COMPARISON = [
        "compare it",
        "compare them",
        "which one",
    ]

    # --------------------------------------------------
    # Continuation questions
    #
    # These are only considered conversational signals
    # when they form a very short follow-up query.
    #
    # Example:
    # "Why?"
    # "How?"
    # "Examples?"
    #
    # But:
    # "Why is inheritance useful?"
    #
    # is a normal standalone knowledge query.
    # --------------------------------------------------

    CONTINUATION = [
        "why",
        "how",
        "example",
        "examples",
        "advantages",
        "disadvantages",
        "applications",
        "use cases",
        "benefits",
        "limitations",
    ]

    # --------------------------------------------------
    # Initialization
    # --------------------------------------------------

    def __init__(self):
        pass

    # --------------------------------------------------
    # Detect reference signals
    # --------------------------------------------------

    def detect(self, query: str) -> ReferenceResult:

        if not query:
            return ReferenceResult(
                needs_memory=False,
                reason="empty_query"
            )

        text = query.lower().strip()

        # Normalize whitespace while preserving the
        # original query for the caller.
        text = re.sub(r"\s+", " ", text)

        # Tokenize the query.
        tokens = re.findall(r"\b[\w'-]+\b", text)

        # --------------------------------------------------
        # Previous answer / conversation references
        #
        # These are strong signals because the user is
        # explicitly referring to previous conversation.
        # --------------------------------------------------

        for phrase in self.PREVIOUS:

            if phrase in text:

                return ReferenceResult(
                    needs_memory=True,
                    reason="previous_answer"
                )

        # --------------------------------------------------
        # Follow-up phrases
        #
        # Explicit continuation phrases are strong signals.
        # --------------------------------------------------

        for phrase in self.FOLLOW_UP:

            if phrase in text:

                return ReferenceResult(
                    needs_memory=True,
                    reason="follow_up"
                )

        # --------------------------------------------------
        # Ordinal references
        #
        # Only conversational forms are included.
        #
        # We intentionally do NOT match generic words such
        # as "second", because:
        #
        # "second parameter"
        # "second law"
        # "second phase"
        #
        # are not necessarily conversational references.
        # --------------------------------------------------

        for phrase in self.ORDINAL:

            if phrase in text:

                return ReferenceResult(
                    needs_memory=True,
                    reason="ordinal_reference"
                )

        # --------------------------------------------------
        # Comparison references
        #
        # Only explicitly referential comparisons trigger.
        #
        # Standalone:
        # "What is the difference between OOP and procedural
        # programming?"
        #
        # must remain standalone.
        # --------------------------------------------------

        for phrase in self.COMPARISON:

            if phrase in text:

                return ReferenceResult(
                    needs_memory=True,
                    reason="comparison"
                )

        # --------------------------------------------------
        # Pronouns
        #
        # Pronouns are useful signals, but we don't blindly
        # classify every occurrence as conversational memory.
        #
        # Strong contextual forms are checked first.
        # --------------------------------------------------

        for token in tokens:

            if token in self.PRONOUNS:

                return ReferenceResult(
                    needs_memory=True,
                    reason="pronoun"
                )

        # --------------------------------------------------
        # Very short continuation questions
        #
        # Only short queries are treated as conversational
        # continuation signals.
        #
        # Examples:
        #
        # "Why?"
        # "How?"
        # "Examples?"
        # "Advantages?"
        #
        # Longer standalone questions are not classified
        # here.
        # --------------------------------------------------

        if len(tokens) <= 3:

            if any(
                word in tokens
                for word in self.CONTINUATION
            ):

                return ReferenceResult(
                    needs_memory=True,
                    reason="continuation"
                )

        # --------------------------------------------------
        # Standalone query
        # --------------------------------------------------

        return ReferenceResult(
            needs_memory=False,
            reason="standalone"
        )

"""
Detects whether a user's query depends on previous conversation.
It decides whether the rewrite stage should receive
conversation history or treat the query as standalone.
"""