import json

import aiohttp
import requests

from task.clients.base import AIClient
from task.models.message import Message
from task.models.role import Role


class CustomAnthropicAIClient(AIClient):
    def __init__(
        self,
        endpoint: str,
        model_name: str,
        api_key: str,
        system_prompt: str,
        anthropic_version: str,
    ):
        super().__init__(
            endpoint=endpoint,
            model_name=model_name,
            api_key=api_key,
            system_prompt=system_prompt,
        )

        self._anthropic_version = anthropic_version
        self._session = None

    def _prepare_headers(self) -> dict[str, str]:
        return {
            "x-api-key": self._api_key,
            "anthropic-version": self._anthropic_version,
            "content-type": "application/json",
        }

    def get_completion(self, messages: list[Message], **kwargs) -> Message:
        headers = self._prepare_headers()
        body = {
            "model": self._model_name,
            "system": self._system_prompt,
            "max_tokens": 1024,
            "messages": [message.to_dict() for message in messages],
        }

        response = requests.post(url=self._endpoint, headers=headers, json=body).json()
        bot_msg = response["content"][0]["text"]
        print(f"🤖: {bot_msg}")

        return Message(role=Role.AI, content=bot_msg)

    async def stream_completion(self, messages: list[Message], **kwargs) -> Message:
        if self._session is None:
            self._session = aiohttp.ClientSession()

        headers = self._prepare_headers()
        body = {
            "model": self._model_name,
            "system": self._system_prompt,
            "max_tokens": 1024,
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

                content = json.loads(data)

                if content["type"] == "content_block_delta":
                    delta = content["delta"]["text"]
                    bot_msg += delta
                    print(delta, end="", flush=True)

        print()

        return Message(role=Role.AI, content=bot_msg)

    async def close_recources(self):
        await self._session.close()
