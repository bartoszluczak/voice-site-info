import asyncio

import openai
from fastapi import Response
import vocode
from vocode.streaming.models.agent import RESTfulUserImplementedAgentConfig, ChatGPTAgentConfig, AgentConfig
from vocode.streaming.models.message import BaseMessage
from vocode.streaming.telephony.hosted.inbound_call_server import InboundCallServer
from vocode.streaming.models.telephony import TwilioConfig
from vocode.streaming.telephony.hosted.inbound_call_user_agent_server import InboundCallUserAgentServer

vocode.api_key = "c6641933c930ca4a767582a2dd3edbc6"

webpage_address = 'https://stockbuddyapp.com/'

async def getPageName(page_url):
    openai.api_type = "open_ai"
    openai.api_base = "https://api.openai.com/v1"
    openai.api_version = None
    openai.api_key = "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"

    messages = [
        {"role": "system", "content": "Based on information on site " + page_url + " write short professional telephone greeting with some fictional assistant. Give her some friendly name."},
    ]

    chat_parameters = {
        "messages": messages,
        "max_tokens": 100,
        "temperature": 0.8,
        "model": "gpt-3.5-turbo"
    }
    return openai.ChatCompletion.create(**chat_parameters)

if __name__ == '__main__':
    intro_text = asyncio.run(getPageName(webpage_address))

    server = InboundCallServer(
        agent_config=RESTfulUserImplementedAgentConfig(
            initial_message=BaseMessage(text=intro_text.choices[0].message.content),
            prompt_preamble="You are a helpful AI assistant. Answer questions in 50 words or less.",
            respond=RESTfulUserImplementedAgentConfig.EndpointConfig(
                url="https://2db2-109-173-156-91.ngrok.io/respond",
            ),
        ),
        twilio_config=TwilioConfig(
            account_sid='AC968d1b274d38b28de856214f38e2ba36',
            auth_token='34516b27dab5978600e8bf98f74f4083',
        ),
    )

    server.app.get("/")(lambda: Response(
        content=
        f"<div>Paste the following URL into your Twilio config: /vocode",
        media_type="text/html"))
    server.run(host="0.0.0.0", port=3000)
