import json
from typing import Dict, List

import redis


class ShortTermMemory:
    """
    Redis-backed short-term conversation memory.

    Stores only the active session's recent conversation in Redis.
    PostgreSQL remains responsible for long-term/durable conversation
    persistence.

    Redis key format:
        rag:short_memory:{session_id}

    Stored value:
        Redis List containing JSON-encoded messages.
    """

    KEY_PREFIX = "rag:short_memory"

    def __init__(
        self,
        session_id: str,
        max_messages: int = 20,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
        redis_password: str | None = None,
        ttl_seconds: int | None = None,
    ):
        if not session_id or not session_id.strip():
            raise ValueError("session_id must be a non-empty string.")

        if max_messages <= 0:
            raise ValueError("max_messages must be greater than 0.")

        if ttl_seconds is not None and ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be greater than 0.")

        self.session_id = session_id
        self.max_messages = max_messages
        self.ttl_seconds = ttl_seconds

        self._redis = redis.Redis(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            password=redis_password,
            decode_responses=True,
        )

        self._key = f"{self.KEY_PREFIX}:{self.session_id}"

        # Fail fast if Redis is unavailable.
        self._redis.ping()

    # --------------------------------------------------
    # Internal helpers
    # --------------------------------------------------

    @staticmethod
    def _validate_message(role: str, content: str) -> None:
        if not role or not role.strip():
            raise ValueError("Message role must be a non-empty string.")

        if content is None:
            raise ValueError("Message content cannot be None.")

    @staticmethod
    def _serialize_message(role: str, content: str) -> str:
        return json.dumps(
            {
                "role": role,
                "content": content,
            },
            ensure_ascii=False,
        )

    @staticmethod
    def _deserialize_message(value: str) -> Dict[str, str]:
        try:
            message = json.loads(value)
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError(
                "Invalid message stored in Redis."
            ) from exc

        if not isinstance(message, dict):
            raise ValueError("Redis message must deserialize to a dictionary.")

        return {
            "role": str(message.get("role", "")),
            "content": str(message.get("content", "")),
        }

    def _refresh_ttl(self) -> None:
        """
        Refresh session TTL when configured.

        This keeps active sessions alive while they are being used.
        """
        if self.ttl_seconds is not None:
            self._redis.expire(
                self._key,
                self.ttl_seconds,
            )

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------

    def add_message(
        self,
        role: str,
        content: str,
    ) -> None:
        """
        Add a message to the current session.

        Only the latest `max_messages` messages are retained.
        """

        self._validate_message(role, content)

        message = self._serialize_message(
            role=role,
            content=content,
        )

        pipeline = self._redis.pipeline()

        pipeline.rpush(
            self._key,
            message,
        )

        pipeline.ltrim(
            self._key,
            -self.max_messages,
            -1,
        )

        if self.ttl_seconds is not None:
            pipeline.expire(
                self._key,
                self.ttl_seconds,
            )

        pipeline.execute()

    def get_history(self) -> List[Dict[str, str]]:
        """
        Return the complete short-term history currently stored
        for this session.
        """

        messages = self._redis.lrange(
            self._key,
            0,
            -1,
        )

        return [
            self._deserialize_message(message)
            for message in messages
        ]

    def get_recent_messages(
        self,
        n: int = 5,
    ) -> List[Dict[str, str]]:
        """
        Return the latest `n` messages from the current session.
        """

        if n <= 0:
            return []

        messages = self._redis.lrange(
            self._key,
            -n,
            -1,
        )

        return [
            self._deserialize_message(message)
            for message in messages
        ]

    def clear(self) -> None:
        """
        Delete the current session's short-term memory.
        """

        self._redis.delete(self._key)

    def size(self) -> int:
        """
        Return the number of messages currently stored
        for this session.
        """

        return self._redis.llen(self._key)

    def is_empty(self) -> bool:
        """
        Return True if the current session has no messages.
        """

        return self.size() == 0

    def exists(self) -> bool:
        """
        Return True if Redis currently contains memory
        for this session.
        """

        return bool(self._redis.exists(self._key))