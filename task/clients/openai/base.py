from abc import ABC

from task.clients.base import AIClient
from task.models.message import Message
from task.models.role import Role


class BaseOpenAIClient(AIClient, ABC):
    def __init__(
        self, endpoint: str, model_name: str, system_prompt: str, api_key: str
    ):
        if not api_key or api_key.strip() == "":
            raise ValueError("API key cannot be null or empty")

        super().__init__(
            endpoint=endpoint,
            model_name=model_name,
            system_prompt=system_prompt,
            api_key=api_key,
        )

    def populate_system_message(self, messages: list[Message]) -> None:
        if messages[0].role != Role.SYSTEM:
            system_msg = Message(role=Role.SYSTEM, content=self._system_prompt)
            messages.insert(0, system_msg)
