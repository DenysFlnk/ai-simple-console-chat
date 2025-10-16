import asyncio

from task.clients.anthropic.client import AnthropicAIClient
from task.clients.anthropic.custom_client import CustomAnthropicAIClient
from task.clients.base import AIClient
from task.clients.openai.client import OpenAIClient
from task.clients.openai.custom_client import CustomOpenAIClient
from task.constants import (
    ANTHROPIC_API_KEY,
    ANTHROPIC_ENDPOINT,
    ANTHROPIC_MODEL,
    ANTHROPIC_VERSION,
    DEFAULT_SYSTEM_PROMPT,
    OPENAI_API_KEY,
    OPENAI_ENDPOINT,
    OPENAI_MODEL,
)
from task.models.conversation import Conversation
from task.models.message import Message
from task.models.role import Role


async def start(stream: bool, client: AIClient) -> None:
    conversation = Conversation()

    while True:
        user_input = input("🧑: ")

        if user_input == "Exit":
            await client.close_recources()
            exit(0)

        user_msg = Message(Role.USER, user_input)
        conversation.add_message(user_msg)

        bot_msg = None
        if stream:
            bot_msg = await client.stream_completion(conversation.get_messages())
        else:
            bot_msg = client.get_completion(conversation.get_messages())

        conversation.add_message(bot_msg)


openAiClient = OpenAIClient(
    endpoint=OPENAI_ENDPOINT,
    model_name=OPENAI_MODEL,
    system_prompt=DEFAULT_SYSTEM_PROMPT,
    api_key=OPENAI_API_KEY,
)

customOpenAIClient = CustomOpenAIClient(
    endpoint=OPENAI_ENDPOINT,
    model_name=OPENAI_MODEL,
    system_prompt=DEFAULT_SYSTEM_PROMPT,
    api_key=OPENAI_API_KEY,
)

anthropicAiClient = AnthropicAIClient(
    endpoint=ANTHROPIC_ENDPOINT,
    model_name=ANTHROPIC_MODEL,
    system_prompt=DEFAULT_SYSTEM_PROMPT,
    api_key=ANTHROPIC_API_KEY,
)
customAnthropicAIClient = CustomAnthropicAIClient(
    endpoint=ANTHROPIC_ENDPOINT,
    model_name=ANTHROPIC_MODEL,
    system_prompt=DEFAULT_SYSTEM_PROMPT,
    api_key=ANTHROPIC_API_KEY,
    anthropic_version=ANTHROPIC_VERSION,
)
asyncio.run(start(stream=True, client=customAnthropicAIClient))
