import os
import openai
from vocode.streaming.models.agent import RESTfulAgentOutput, RESTfulAgentText
from vocode.streaming.user_implemented_agent.restful_agent import RESTfulAgent
from dotenv import load_dotenv

load_dotenv()

webpage_address = os.getenv("SOURCE_PAGE_URL")


class YourAgent(RESTfulAgent):
    def __init__(self, ):
        super().__init__()

    openai.api_type = "open_ai"
    openai.api_base = "https://api.openai.com/v1"
    openai.api_version = None
    openai.api_key = os.getenv('OPENAI_API_KEY')

    async def respond(self, input: str, conversation_id: str) -> RESTfulAgentOutput:
        messages = [
            {"role": "system", "content": "Hello how can I help you"},
            {"role": "user",
             "content": "Answer to user question " + input + "only from information from " + webpage_address}
        ]

        chat_parameters = {
            "messages": messages,
            "max_tokens": 100,
            "temperature": 0.8,
            "model": "gpt-3.5-turbo"
        }

        chat_completion = await openai.ChatCompletion.acreate(**chat_parameters)
        text = chat_completion.choices[0].message.content

        return RESTfulAgentText(response=text)


if __name__ == '__main__':
    agent = YourAgent()
    agent.run(host="0.0.0.0", port=int(os.getenv("AGENT_PORT")))
