from anthropic import Anthropic, AsyncAnthropic

from task.clients.base import AIClient
from task.models.message import Message
from task.models.role import Role


class AnthropicAIClient(AIClient):
    def __init__(
        self, endpoint: str, model_name: str, api_key: str, system_prompt: str
    ):
        super().__init__(
            endpoint=endpoint,
            model_name=model_name,
            api_key=api_key,
            system_prompt=system_prompt,
        )
        self._client = Anthropic()
        self._asyncClient = AsyncAnthropic()

    def get_completion(self, messages: list[Message], **kwargs) -> Message:
        formatted_messages = [message.to_dict() for message in messages]

        response = self._client.messages.create(
            model=self._model_name,
            messages=formatted_messages,
            max_tokens=1024,
            system=self._system_prompt,
        )

        bot_msg = response.content[0].text
        print(f"🤖: {bot_msg}")

        return Message(role=Role.AI, content=bot_msg)

    async def stream_completion(self, messages: list[Message], **kwargs) -> Message:
        formatted_messages = [message.to_dict() for message in messages]

        stream = await self._asyncClient.messages.create(
            model=self._model_name,
            messages=formatted_messages,
            max_tokens=1024,
            system=self._system_prompt,
            stream=True,
        )

        bot_msg = ""
        print("🤖:", end=" ")

        async for chunk in stream:
            if chunk.type != "content_block_delta":
                continue

            delta = chunk.delta.text

            bot_msg += delta
            print(delta, end="", flush=True)

        print()

        return Message(role=Role.AI, content=bot_msg)

    async def close_recources(self):
        pass
