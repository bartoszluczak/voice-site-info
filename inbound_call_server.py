import asyncio
import os

import openai
import vocode
from fastapi import Response
from vocode.streaming.models.agent import RESTfulUserImplementedAgentConfig
from vocode.streaming.models.message import BaseMessage
from vocode.streaming.models.synthesizer import ElevenLabsSynthesizerConfig
from vocode.streaming.models.transcriber import DeepgramTranscriberConfig, PunctuationEndpointingConfig, \
    GoogleTranscriberConfig, TimeEndpointingConfig
from vocode.streaming.telephony.hosted.inbound_call_server import InboundCallServer
from vocode.streaming.models.telephony import TwilioConfig
from dotenv import load_dotenv

load_dotenv()

vocode.api_key = os.getenv("VOCODE_API_KEY")
webpage_address = os.getenv("SOURCE_PAGE_URL")


async def get_page_name(page_url):
    openai.api_type = "open_ai"
    openai.api_base = "https://api.openai.com/v1"
    openai.api_version = None
    openai.api_key = os.getenv('OPENAI_API_KEY')

    messages = [
        {"role": "system",
         "content": "Write short professional telephone greeting for fictional company with frictional assistant. Make up a companny name and assistant name"},
    ]

    chat_parameters = {
        "messages": messages,
        "max_tokens": 100,
        "temperature": 0.5,
        "model": os.getenv("OPENAI_MODEL")
    }
    return openai.ChatCompletion.create(**chat_parameters)


if __name__ == '__main__':
    intro_text = asyncio.run(get_page_name(webpage_address))

    server = InboundCallServer(
        agent_config=RESTfulUserImplementedAgentConfig(
            initial_message=BaseMessage(text=intro_text.choices[0].message.content),
            prompt_preamble="You are a helpful assistant. Answer questions in 50 words or less.",
            respond=RESTfulUserImplementedAgentConfig.EndpointConfig(
                # url=os.getenv("AGENT_URL") + ":" + os.getenv("AGENT_PORT") + "/respond",
                url=os.getenv("AGENT_URL") + "/respond",
            ),
        ),
        twilio_config=TwilioConfig(
            account_sid=os.getenv("TWILIO_ACCOUNT_SID"),
            auth_token=os.getenv("TWILIO_AUTH_TOKEN"),
        ),
        transcriber_config=GoogleTranscriberConfig.from_telephone_input_device(
            # endpointing_config=PunctuationEndpointingConfig()
        )
        # transcriber_config=DeepgramTranscriberConfig.from_telephone_input_device(
        #     endpointing_config=PunctuationEndpointingConfig()
        # ),
    )

    server.app.get("/")(lambda: Response(
        content=
        f"<div>Vocode Twilio endpoint",
        media_type="text/html"))
    server.run(host="localhost", port=int(os.getenv("INBOUND_CALL_SERVER_PORT")))
