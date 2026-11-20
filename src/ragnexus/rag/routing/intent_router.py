from __future__ import annotations

import re
from ragnexus.llm.providers.base import ChatModel


class IntentRouter:
    """Low-cost three-way conversational intent router.

    The router distinguishes requests to recall this conversation from requests
    about a topic. Explicit general and memory signals are handled locally so
    routine turns do not incur an additional LLM round trip. Knowledge is the
    safe default; knowledge follow-ups still receive conversation history in
    the normal RAG pipeline.
    """

    MEMORY = "memory"
    KNOWLEDGE = "knowledge"
    GENERAL = "general"

    _GENERAL_EXACT = {
        "hi", "hello", "hey", "good morning", "good afternoon", "good evening",
        "good night", "bye", "goodbye", "thanks", "thank you", "thank you so much",
        "nice to meet you", "how are you", "how are you?", "who are you",
        "who are you?", "tell me a joke", "good to see you",
    }
    _MEMORY_PATTERNS = (
        re.compile(r"\bwhat did (?:i|we|you) (?:ask|say|tell|discuss|decide|agree|mention)\b"),
        re.compile(r"\bwhat have i (?:said|told|shared|mentioned)\b"),
        re.compile(r"\bwhat was (?:my|our|your) (?:first|previous|last)\b"),
        re.compile(r"\bwhat is my (?:name|goal|goals|preference|preferred|project|role)\b"),
        re.compile(r"\bremind me (?:what|about)\b"),
        re.compile(r"\bsummar(?:ize|ise) (?:our|this) conversation\b"),
        re.compile(r"\bfrom (?:our|this) conversation\b"),
        re.compile(r"\bwhat did we conclude\b"),
        re.compile(r"\bwhat did you recommend earlier\b"),
        re.compile(r"\bwhat have we discussed\b"),
        re.compile(r"\b(?:my|our|the) (?:previous|earlier|last) (?:question|message|answer|conversation)\b"),
    )

    def __init__(self, model: ChatModel | None = None):
        # Keep the constructor compatible with the existing bootstrap factory.
        self.model = model

    def classify(self, query: str, conversation_history=None) -> str:
        value = " ".join((query or "").casefold().split()).strip(" .!?\t\n")
        if not value:
            return self.KNOWLEDGE
        if value in self._GENERAL_EXACT:
            return self.GENERAL
        if any(pattern.search(value) for pattern in self._MEMORY_PATTERNS):
            return self.MEMORY
        # A factual follow-up such as "what about that?" remains knowledge;
        # FollowUpResolver receives the same history to resolve its reference.
        return self.KNOWLEDGE

    def explain(self, query: str, conversation_history=None) -> dict[str, str]:
        intent = self.classify(query, conversation_history)
        reasons = {
            self.MEMORY: "explicit_conversation_recall",
            self.GENERAL: "casual_or_general_interaction",
            self.KNOWLEDGE: "factual_or_technical_knowledge_request",
        }
        return {"intent": intent, "reason": reasons[intent]}
