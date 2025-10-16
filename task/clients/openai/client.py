from openai import AsyncOpenAI, OpenAI

from task.clients.openai.base import BaseOpenAIClient
from task.models.message import Message
from task.models.role import Role


class OpenAIClient(BaseOpenAIClient):
    def __init__(
        self, endpoint: str, model_name: str, system_prompt: str, api_key: str
    ):
        super().__init__(
            endpoint=endpoint,
            model_name=model_name,
            system_prompt=system_prompt,
            api_key=api_key,
        )

        self.client = OpenAI(api_key=self._api_key)
        self.asyncClient = AsyncOpenAI(api_key=self._api_key)

    def get_completion(self, messages: list[Message], **kwargs) -> Message:
        self.populate_system_message(messages)
        formatted_messages = [message.to_dict() for message in messages]
        completion = self.client.chat.completions.create(
            model=self._model_name, messages=formatted_messages
        )

        bot_msg = completion.choices[0].message.content
        print(f"🤖: {bot_msg}")

        return Message(role=Role.AI, content=bot_msg)

    async def stream_completion(self, messages: list[Message], **kwargs) -> Message:
        self.populate_system_message(messages)
        formatted_messages = [message.to_dict() for message in messages]
        completion = await self.asyncClient.chat.completions.create(
            model=self._model_name, messages=formatted_messages, stream=True
        )

        bot_msg = ""

        print("🤖:", end=" ")

        async for chunk in completion:
            delta = chunk.choices[0].delta.content

            if delta:
                bot_msg += delta
                print(delta, end="", flush=True)

        print()

        return Message(role=Role.AI, content=bot_msg)

    async def close_recources(self):
        pass
