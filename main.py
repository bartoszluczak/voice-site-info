import logging
import os
import asyncio
from typing import Optional

import openai
from fastapi import FastAPI
from vocode.streaming.models.model import BaseModel
from vocode.streaming.models.synthesizer import SynthesizerConfig, ElevenLabsSynthesizerConfig

from vocode.streaming.models.synthesizer import ElevenLabsSynthesizerConfig

from vocode.streaming.models.telephony import TwilioConfig
from pyngrok import ngrok
from vocode.streaming.models.transcriber import TranscriberConfig
from vocode.streaming.telephony.config_manager.redis_config_manager import (
    RedisConfigManager,
)
from vocode.streaming.models.agent import ChatGPTAgentConfig, AgentConfig
from vocode.streaming.models.message import BaseMessage
from vocode.streaming.telephony.server.base import (
    TwilioInboundCallConfig,
    TelephonyServer,
)
from vocode.streaming.models.agent import RESTfulUserImplementedAgentConfig
from vocode.streaming.models.message import BaseMessage
from speller_agent import SpellerAgentFactory
import sys

# if running from python, this will load the local .env
# docker-compose will load the .env file by itself
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(docs_url=None)

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

config_manager = RedisConfigManager()

BASE_URL = os.getenv("BASE_URL")


async def get_page_name(page_url):
    openai.api_type = "open_ai"
    openai.api_base = "https://api.openai.com/v1"
    openai.api_version = None
    openai.api_key = "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"

    messages = [
        {"role": "system",
         "content": "Based on information on site " + page_url + " write short professional telephone greeting with some fictional assistant. Give her some friendly name."},
    ]

    chat_parameters = {
        "messages": messages,
        "max_tokens": 100,
        "temperature": 0.8,
        "model": "gpt-3.5-turbo-16k"
    }

    resp = await openai.ChatCompletion.create(**chat_parameters)
    return resp


if not BASE_URL:
    ngrok_auth = os.environ.get("NGROK_AUTH_TOKEN")
    if ngrok_auth is not None:
        ngrok.set_auth_token(ngrok_auth)
    port = sys.argv[sys.argv.index("--port") + 1] if "--port" in sys.argv else 3001

    # Open a ngrok tunnel to the dev server
    BASE_URL = ngrok.connect(port).public_url.replace("http://", "")
    logger.info('ngrok tunnel "{}" -> "http://0.0.0.0:{}"'.format(BASE_URL, port))

if not BASE_URL:
    raise ValueError("BASE_URL must be set in environment if not using pyngrok")

webpage_address = "https://stockbuddyapp.com/"


# intro_text = get_page_name(webpage_address)
# print(intro_text)


telephony_server = TelephonyServer(
    base_url=BASE_URL,
    config_manager=config_manager,
    inbound_call_configs=[
        TwilioInboundCallConfig(
            url="/inbound_call",
            agent_config=RESTfulUserImplementedAgentConfig(
                initial_message=BaseMessage(text="Hello I am your assistant"),
                prompt_preamble="You are a helpful assistant. Answer questions in 50 words or less.",
                respond=RESTfulUserImplementedAgentConfig.EndpointConfig(
                    url="http://35.208.224.244:4001/respond",
                ),
            ),
            twilio_config=TwilioConfig(
                account_sid=os.environ["TWILIO_ACCOUNT_SID"],
                auth_token=os.environ["TWILIO_AUTH_TOKEN"],
            ),
            synthesizer_config=ElevenLabsSynthesizerConfig.from_telephone_output_device(
                api_key=os.getenv("ELEVENLABS_API_KEY"),
                voice_id=os.getenv("VOICE_ID")
    )
        )
    ],
    # agent_factory=SpellerAgentFactory(),
    logger=logger,
)

app.include_router(telephony_server.get_router())
