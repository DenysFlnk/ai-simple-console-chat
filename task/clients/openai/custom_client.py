import json

import aiohttp
import requests

from task.clients.openai.base import BaseOpenAIClient
from task.models.message import Message
from task.models.role import Role


class CustomOpenAIClient(BaseOpenAIClient):
    def __init__(
        self, endpoint: str, model_name: str, system_prompt: str, api_key: str
    ):
        super().__init__(
            endpoint=endpoint,
            model_name=model_name,
            system_prompt=system_prompt,
            api_key=api_key,
        )
        self._session = None

    def _prepare_headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }

    def get_completion(self, messages: list[Message], **kwargs) -> Message:
        self.populate_system_message(messages)

        headers = self._prepare_headers()
        body = {
            "model": self._model_name,
            "messages": [message.to_dict() for message in messages],
        }

        response = requests.post(url=self._endpoint, headers=headers, json=body).json()
        bot_msg = response["choices"][0]["message"]["content"]
        print(f"🤖: {bot_msg}")

        return Message(role=Role.AI, content=bot_msg)

    async def stream_completion(self, messages: list[Message], **kwargs) -> Message:
        if self._session is None:
            self._session = aiohttp.ClientSession()

        self.populate_system_message(messages)

        headers = self._prepare_headers()
        body = {
            "model": self._model_name,
            "messages": [message.to_dict() for message in messages],
            "stream": True,
        }

        print("🤖:", end=" ")

        bot_msg = ""
        async with self._session.post(
            self._endpoint, headers=headers, json=body
        ) as response:
            async for chunk in response.content:
                if not chunk.strip() or not chunk.startswith(b"data: "):
                    continue

                data = chunk[len(b"data: ") :].decode()

                if "[DONE]" in data:
                    break

                content = json.loads(data)["choices"][0]

                if "delta" in content and "content" in content["delta"]:
                    delta = content["delta"]["content"]
                    bot_msg += delta
                    print(delta, end="", flush=True)

        print()

        return Message(role=Role.AI, content=bot_msg)

    async def close_recources(self):
        await self._session.close()
