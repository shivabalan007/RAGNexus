from memory.short_term import ShortTermMemory
from memory.long_term import LongTermMemory


class MemoryManager:
    """
    Coordinates short-term Redis memory and long-term PostgreSQL memory.

    Architecture:

        Redis
          ↓
    Active session history

        PostgreSQL
          ↓
    Durable conversation history

    Redis is treated as the fast session cache.
    PostgreSQL remains the long-term source of truth.
    """

    def __init__(
        self,
        short_memory: ShortTermMemory,
        long_memory: LongTermMemory,
        session_id: str,
        user_id: str = "default_user",
    ):
        if not session_id or not session_id.strip():
            raise ValueError("session_id must be a non-empty string.")

        if not user_id or not user_id.strip():
            raise ValueError("user_id must be a non-empty string.")

        self.short_memory = short_memory
        self.long_memory = long_memory
        self.session_id = session_id
        self.user_id = user_id

        self._loaded = False

    # --------------------------------------------------
    # Session Loading
    # --------------------------------------------------

    def load_session(self) -> None:
        """
        Load the current session into Redis-backed short-term memory.

        Loading strategy:

        1. If this MemoryManager already loaded the session,
           do nothing.

        2. If Redis already contains the session,
           use Redis directly.

        3. If Redis does not contain the session,
           load the latest messages from PostgreSQL and
           populate Redis.

        This avoids querying PostgreSQL every time a session
        is reused.
        """

        if self._loaded:
            return

        print("\n========== MEMORY LOAD ==========")
        print("Session ID :", self.session_id)

        # --------------------------------------------------
        # 1. Check Redis first
        # --------------------------------------------------

        if self.short_memory.exists():

            print("Redis Session : FOUND")
            print("Source        : Redis")

            self._loaded = True

            print("History Size  :", self.short_memory.size())
            print("=" * 50)

            return

        # --------------------------------------------------
        # 2. Redis does not contain session
        #    Hydrate from PostgreSQL
        # --------------------------------------------------

        print("Redis Session : NOT FOUND")
        print("Source        : PostgreSQL")

        history = self.long_memory.get_recent_messages(
            session_id=self.session_id,
            limit=self.short_memory.max_messages,
        )

        # Redis should be empty at this point, but clear it
        # defensively before hydration.
        self.short_memory.clear()

        # --------------------------------------------------
        # 3. Populate Redis
        # --------------------------------------------------

        for message in history:
            self.short_memory.add_message(
                role=message["role"],
                content=message["content"],
            )

        self._loaded = True

        print("Loaded From PG :", len(history))
        print("Redis Size     :", self.short_memory.size())
        print("=" * 50)

    # --------------------------------------------------
    # History
    # --------------------------------------------------

    def get_history(self):
        """
        Return the active session history from Redis.
        """

        if not self._loaded:
            self.load_session()

        return self.short_memory.get_history()

    # --------------------------------------------------
    # Add Message
    # --------------------------------------------------

    def add_message(
        self,
        role: str,
        content: str,
    ) -> None:
        """
        Persist a message to PostgreSQL and then update Redis.

        PostgreSQL is written first because it is the durable
        source of truth.

        Redis is then updated as the active-session cache.
        """

        if not role or not role.strip():
            raise ValueError("Message role must be a non-empty string.")

        if content is None:
            raise ValueError("Message content cannot be None.")

        # --------------------------------------------------
        # 1. Persist to PostgreSQL
        # --------------------------------------------------

        self.long_memory.save_message(
            session_id=self.session_id,
            user_id=self.user_id,
            role=role,
            message=content,
        )

        # --------------------------------------------------
        # 2. Update Redis
        # --------------------------------------------------

        self.short_memory.add_message(
            role=role,
            content=content,
        )

    # --------------------------------------------------
    # Clear Session
    # --------------------------------------------------

    def clear(self) -> None:
        """
        Clear both Redis short-term memory and
        PostgreSQL long-term memory.
        """

        print("\n========== MEMORY CLEAR ==========")
        print("Session ID :", self.session_id)

        # PostgreSQL is the source of truth.
        self.long_memory.clear_session(
            self.session_id
        )

        # Remove active Redis session.
        self.short_memory.clear()

        self._loaded = False

        print("PostgreSQL : CLEARED")
        print("Redis      : CLEARED")
        print("=" * 50)

    # --------------------------------------------------
    # Recent Messages
    # --------------------------------------------------

    def get_recent_messages(
        self,
        n: int = 5,
    ):
        """
        Return the latest n messages from Redis.
        """

        if not self._loaded:
            self.load_session()

        return self.short_memory.get_recent_messages(n)
"""
Combines Short-Term and Long-Term memory.
"""