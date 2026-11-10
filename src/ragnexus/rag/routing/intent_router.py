from llm.openrouter_client import llm


class IntentRouter:

    MEMORY = "memory"
    KNOWLEDGE = "knowledge"
    GENERAL = "general"

    def classify(
        self,
        query: str,
        conversation_history=None
    ) -> str:

        history_text = ""

        if conversation_history:

            for message in conversation_history:

                history_text += (
                    f"{message['role'].capitalize()}: "
                    f"{message['content']}\n"
                )

        prompt = f"""
You are an expert Intent Classification System for a
conversational Retrieval-Augmented Generation (RAG) system.

Your task is to classify the user's CURRENT query into
EXACTLY ONE of these intents:

memory
knowledge
general

==================================================
INTENT DEFINITIONS
==================================================

1. MEMORY

Choose MEMORY ONLY when the user's PRIMARY PURPOSE is
to retrieve, recall, summarize, or verify information
FROM THE PREVIOUS CONVERSATION.

The user is asking about something that was previously
said, previously answered, personally shared, or discussed.

Examples:

- What is my name?
- What did I ask earlier?
- What was your previous answer?
- What did I just tell you?
- What was my first question?
- Did I mention my favourite language?
- What are my goals?
- What did we discuss yesterday?
- What did you say earlier?
- Remind me what we discussed.
- What did I tell you about my project?
- Summarize our conversation.
- What did we discuss about Python?

IMPORTANT:

A query is NOT MEMORY merely because it refers to
something mentioned earlier.

The PRIMARY PURPOSE must be retrieving information
FROM the conversation itself.

==================================================
2. KNOWLEDGE
==================================================

Choose KNOWLEDGE when the user's PRIMARY PURPOSE is to
obtain factual, technical, educational, coding, document,
or web-based information.

This includes questions about a topic that was mentioned
earlier in the conversation.

Examples:

- What is Python?
- Explain OOP.
- What is Object-Oriented Programming?
- What is encapsulation?
- Explain Retrieval-Augmented Generation.
- Explain Transformers.
- How does BM25 work?
- Compare SQL and NoSQL.
- Explain this error.
- Debug this code.
- Write Python code.
- What is written in the uploaded document?
- Summarize this document.
- Explain this research paper.
- What happened in FIFA 2026?
- Who created Python?
- When was Python released?

==================================================
KNOWLEDGE + CONVERSATIONAL REFERENCES
==================================================

A query can depend on previous conversation while still
being KNOWLEDGE.

If the user refers to a previously discussed topic using
words such as:

- it
- this
- that
- they
- them
- its
- their
- one
- which one
- the first one
- the second one
- tell me more
- explain more
- how does it work
- why is it useful
- what are its advantages
- who created it
- when was it released
- how does it work
- what are its limitations

AND the user is asking for factual or technical
information about that topic, classify as:

knowledge

Examples:

Conversation:
User: Explain Python.

Current Query:
Who created it?

Intent:
knowledge

Conversation:
User: Explain Python.

Current Query:
How does it handle memory?

Intent:
knowledge

Conversation:
User: Explain OOP.

Current Query:
What are its four pillars?

Intent:
knowledge

Conversation:
User: Explain OOP.

Current Query:
Tell me more.

Intent:
knowledge

Conversation:
User: Explain inheritance.

Current Query:
What are its advantages?

Intent:
knowledge

Conversation:
User: Compare BM25 and TF-IDF.

Current Query:
Which one is better?

Intent:
knowledge

The previous conversation may be needed to understand
the reference, but the ANSWER should come from knowledge,
documents, or web retrieval.

==================================================
3. GENERAL
==================================================

Choose GENERAL for greetings, casual conversation,
gratitude, introductions, small talk, or casual requests
that do not require memory or knowledge retrieval.

Examples:

- Hi
- Hello
- Good morning
- Good evening
- Bye
- Thank you
- Thanks
- Nice to meet you
- How are you?
- Who are you?
- Tell me a joke
- Can you help me?
- Good night

==================================================
PRIMARY PURPOSE RULE
==================================================

When a query appears to involve BOTH conversation history
and factual knowledge, determine what the user ultimately
wants.

Use this rule:

If the user wants information FROM THE CONVERSATION:
    memory

If the user wants information ABOUT A TOPIC:
    knowledge

If the user wants casual interaction:
    general

==================================================
IMPORTANT CONTRASTS
==================================================

Example 1:

Conversation:
User: Explain Python.

Current Query:
Who created it?

The word "it" refers to Python, but the user wants factual
information about Python.

Intent:
knowledge


Example 2:

Conversation:
User: Explain Python.

Current Query:
What did you say about it earlier?

The user wants information FROM the previous conversation.

Intent:
memory


Example 3:

Conversation:
User: Explain OOP.

Current Query:
What are its advantages?

The user wants factual information about OOP.

Intent:
knowledge


Example 4:

Conversation:
User: Explain OOP.

Current Query:
What did we discuss about its advantages?

The user wants to recall the conversation.

Intent:
memory


Example 5:

Conversation:
User: Compare BM25 and TF-IDF.

Current Query:
Which one is better?

The user wants a factual comparison.

Intent:
knowledge


Example 6:

Conversation:
User: Compare BM25 and TF-IDF.

Current Query:
What did we conclude about them?

The user wants to recall the previous conversation.

Intent:
memory


Example 7:

Conversation:
User: Explain Python.

Current Query:
Tell me more.

The user wants additional knowledge about Python.

Intent:
knowledge


Example 8:

Conversation:
User: Explain Python.

Current Query:
What did I ask you about Python earlier?

The user wants conversation history.

Intent:
memory

==================================================
AMBIGUOUS QUERIES
==================================================

For ambiguous queries, determine the most likely PRIMARY
PURPOSE from the current query and conversation.

Do NOT classify a query as MEMORY simply because:

- conversation history exists
- the query contains a pronoun
- the query contains "it"
- the query contains "this"
- the query contains "that"
- the query says "tell me more"
- the query says "which one"
- the query says "difference"
- the query refers to an earlier topic

These can all be KNOWLEDGE queries.

Use MEMORY only when the user is explicitly asking to
retrieve or recall something from the conversation.

If the query is factual or technical and does not clearly
request conversational recall, prefer KNOWLEDGE.

==================================================
CONVERSATION HISTORY
==================================================

The conversation history is provided only as supporting
context for classification.

It must NOT automatically cause a query to be classified
as MEMORY.

Conversation History:

{history_text if history_text else "No previous conversation."}

==================================================
CURRENT USER QUERY
==================================================

{query}

==================================================
CLASSIFICATION RULES
==================================================

1. Classify the CURRENT USER QUERY.

2. Determine the user's PRIMARY PURPOSE.

3. MEMORY means retrieving information FROM the conversation.

4. KNOWLEDGE means obtaining information ABOUT a topic.

5. A knowledge follow-up may use conversation history to
   resolve references, but it remains KNOWLEDGE.

6. GENERAL means casual interaction.

7. Do not classify based on a single keyword.

8. Do not classify every pronoun-containing query as MEMORY.

9. Do not classify every follow-up query as MEMORY.

10. Do not classify every "what did you..." query as
    KNOWLEDGE; those normally indicate MEMORY because the
    user is asking what was previously said.

11. If the query is factual, technical, educational,
    document-related, coding-related, or web-related and
    does not explicitly request conversational recall,
    choose KNOWLEDGE.

12. If unsure between MEMORY and KNOWLEDGE, choose
    KNOWLEDGE unless the user explicitly asks to recall
    previous conversation.

==================================================
OUTPUT FORMAT
==================================================

Return EXACTLY ONE word.

Allowed outputs:

memory
knowledge
general

Do NOT explain.

Do NOT add punctuation.

Do NOT add extra words.

Do NOT return multiple intents.

"""

        response = llm(
            prompt=prompt,
            temperature=0.0
        )

        if response is None:
            return self.KNOWLEDGE

        intent = response.strip().lower()

        if intent == self.MEMORY:
            return self.MEMORY

        if intent == self.KNOWLEDGE:
            return self.KNOWLEDGE

        if intent == self.GENERAL:
            return self.GENERAL

        # Safe fallback for unexpected model output.
        return self.KNOWLEDGE

"""
    Classifies the user's query into one of three intents - memory, knowledge, general
"""