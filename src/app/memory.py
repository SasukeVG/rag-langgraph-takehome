from typing import List

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain.memory import ConversationBufferMemory
from loguru import logger


class SessionMemory:
    def __init__(self):
        self.memory = ConversationBufferMemory(
            return_messages=True,
            memory_key="messages",
        )
        logger.debug("Session memory initialized")

    def add_user_message(self, content: str) -> None:
        self.memory.chat_memory.add_user_message(content)
        logger.debug(f"Added user message to memory: {content[:50]}...")

    def add_ai_message(self, content: str) -> None:
        self.memory.chat_memory.add_ai_message(content)
        logger.debug(f"Added AI message to memory: {content[:50]}...")

    def get_messages(self) -> List[BaseMessage]:
        return self.memory.chat_memory.messages

    def get_conversation_history(self) -> List[BaseMessage]:
        return self.get_messages()

    def clear(self) -> None:
        self.memory.clear()
        logger.info("Memory cleared")

    def to_langchain_messages(self) -> List[BaseMessage]:
        messages = []
        for msg in self.memory.chat_memory.messages:
            if isinstance(msg, HumanMessage):
                messages.append(HumanMessage(content=msg.content))
            elif isinstance(msg, AIMessage):
                messages.append(AIMessage(content=msg.content))
            else:
                messages.append(msg)
        return messages
