import asyncio

from task.clients.anthropic.client import AnthropicAIClient
from task.clients.base import AIClient
from task.clients.openai.client import OpenAIClient
from task.clients.openai.custom_client import CustomOpenAIClient
from task.constants import (
    ANTHROPIC_API_KEY,
    ANTHROPIC_ENDPOINT,
    ANTHROPIC_MODEL,
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
    OPENAI_ENDPOINT, OPENAI_MODEL, DEFAULT_SYSTEM_PROMPT, OPENAI_API_KEY
)

customOpenAIClient = CustomOpenAIClient(
    OPENAI_ENDPOINT, OPENAI_MODEL, DEFAULT_SYSTEM_PROMPT, OPENAI_API_KEY
)

anthropicAiClient = AnthropicAIClient(
    ANTHROPIC_ENDPOINT, ANTHROPIC_MODEL, DEFAULT_SYSTEM_PROMPT, ANTHROPIC_API_KEY
)
# customAnthropicAIClient = CustomAnthropicAIClient(
#    ANTHROPIC_ENDPOINT, "", DEFAULT_SYSTEM_PROMPT, ANTHROPIC_API_KEY
# )
asyncio.run(start(stream=True, client=anthropicAiClient))
