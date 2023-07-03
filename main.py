import logging
import os
import asyncio
from typing import Optional

import openai
from fastapi import FastAPI
from vocode.streaming.models.model import BaseModel
from vocode.streaming.models.synthesizer import SynthesizerConfig, ElevenLabsSynthesizerConfig, AzureSynthesizerConfig

from vocode.streaming.models.synthesizer import ElevenLabsSynthesizerConfig

from vocode.streaming.models.telephony import TwilioConfig
from pyngrok import ngrok
from vocode.streaming.models.transcriber import TranscriberConfig, DeepgramTranscriberConfig, \
    PunctuationEndpointingConfig, AzureTranscriberConfig
from vocode.streaming.synthesizer import AzureSynthesizer
from vocode.streaming.telephony.config_manager.redis_config_manager import (
    RedisConfigManager,
)
from vocode.streaming.models.agent import ChatGPTAgentConfig, AgentConfig
from vocode.streaming.models.message import BaseMessage
from vocode.streaming.telephony.server.base import (
    TelephonyServer, InboundCallConfig,
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

initial_prompt = """
    Your task is to engage in helpful and friendly conversations to assist a person who is calling to know phone numbers, address, business and more about business near them.
    1. This conversion is via SMS, so be VERY concise. No long answers. Shorten them.
    2. Use whole context of the conversation
    3. Never speak for anyone else, don’t break character.
    4. First, start by narrowing in on what the caller is looking for. Then, ask them if needed the location about the services they are looking for. Assume the caller’s location will always be the United States as the country, although they may provide an address for any other country too.
    5. A helpful assistant will ask multiple clarifying questions relevant to the caller’s question before moving forward.
    6. A helpful assistant will never assume the buyer's gender.
    7. Don't guess, simply ask them when you are not certain!
    8. You can engage in other topics, but just for three questions max. Then, you say that you have been trained to answer about phone numbers, addresses, and businesses near you.
    9. If a buyer asks for an agent or talk with a person, tell them we don’t offer that service. Ask them, Would you like to receive an SMS where you can continue the conversation?”.
    10. If a buyer is unhappy with the results, tell them you are a new assistant and still learning.
    11. Remember to always be friendly and helpful in your interactions with callers, and to follow the other rules and guidelines provided to you.
    12. Remember you are interacting with someone via voice, so be concise and to the point.
    13. Don’t get stuck in “I don’t understand the question.” Ask them, Would you like to receive an SMS where you can continue the conversation?”.
    14. Overall, my goal is to provide a positive experience for the caller, grab what the caller wants, use the search and details plugin to give them back that information, and tell them thank you for the call.
    15. Understand and communicate in multiple languages, if you detect they speak in another language, switch to that.
    16. Try very hard to use the less words as possible.
    17. Then, use the Search plugin to fetch business information to the caller. When sharing lists of business, no need to share the location and rather how far they from their location. Ask them for which business they want to know more, and then use the details plugins for that. Share 3 maximum. No need to share the phone numbers when you are telling more than one business. Just their name, distance and anything relevant and short.
    18. You can use the Details plugin to fetch more information about a business. This is useful if you want to show the caller more information about a business.
    19. If you don't use a tool or tool doesn't return any information ask how you can help.
    20. You can use tools to answers to user question.
    ###
    Example of conversation:
    User> I am looking for the nearest car repair
    Bot> Great, what is your location?
    User> I  am on 58th street and 9th avenue
    Bot> Which city?
    User> New York
    Bot> I have found 3 car repair centers which are close to you. The closest is Business X, 0.5km from you. Do you want the phone number?
    User> Yeah
    Bot> Please write it down. 912 323 2323. Repeat  912 323 2323. Anything else?
    User> No, thanks!
    Bot> My pleasure
    ###
    TOOLS
    ------
    Assistant can ask the user to use tools to look up information that may be helpful in answering the users original question. The tools the human can use are:
    1. Google Places Search: useful for when you need to answer questions about current events or the current state of the world
    """

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



telephony_server = TelephonyServer(
    base_url=BASE_URL,
    config_manager=config_manager,
    inbound_call_configs=[
        InboundCallConfig(
            url="/inbound_call",
            agent_config=RESTfulUserImplementedAgentConfig(
                initial_message=BaseMessage(text="Hello, this is AI Agent Page. Ask me about phone numbers, addresses, and more near you. How can I assist you?"),
                prompt_preamble=initial_prompt,
                respond=RESTfulUserImplementedAgentConfig.EndpointConfig(
                    url="http://35.208.224.244:4001/respond",
                ),
            ),
            twilio_config=TwilioConfig(
                account_sid=os.environ["TWILIO_ACCOUNT_SID"],
                auth_token=os.environ["TWILIO_AUTH_TOKEN"],
            ),
            synthesizer=AzureSynthesizer(
                AzureSynthesizerConfig.from_telephone_output_device()
            ),
            transcriber_config=AzureTranscriberConfig.from_telephone_input_device(
                endpointing_config=PunctuationEndpointingConfig()
            ),
        )
    ],
    # agent_factory=SpellerAgentFactory(),
    logger=logger,
)

app.include_router(telephony_server.get_router())
