import os
import openai
import asyncio
import requests
from vocode.streaming.models.agent import RESTfulAgentOutput, RESTfulAgentText
from vocode.streaming.user_implemented_agent.restful_agent import RESTfulAgent
from dotenv import load_dotenv
from bs4 import BeautifulSoup
load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')
webpage_address = os.getenv("SOURCE_PAGE_URL")

async def get_embeddings(input_text):
    return openai.Embedding.create(
        model="text-embedding-ada-002",
        input=input_text
    )

response = requests.get(webpage_address)
soup = BeautifulSoup(response.text, 'html.parser')
all_texts = soup.get_text()
embeddings = asyncio.run(get_embeddings(all_texts))
message_context =  embeddings.data[0].embedding

class YourAgent(RESTfulAgent):
    def __init__(self, ):
        super().__init__()

    openai.api_type = "open_ai"
    openai.api_base = "https://api.openai.com/v1"
    openai.api_version = None
    openai.api_key = os.getenv('OPENAI_API_KEY')

    async def respond(self, input: str, conversation_id: str) -> RESTfulAgentOutput:
        prompt = """
                Context: %s
                ###
                Question: %s
                Completion:
        """ %(message_context, input)

        print(prompt)

        messages = [
            {"role": "system", "content": "Hello how can I help you"},
            {"role": "user",
             "content": prompt}
        ]

        chat_parameters = {
            "messages": messages,
            "max_tokens": 100,
            "temperature": 0.8,
            "model": os.getenv("OPENAI_MODEL")
        }

        chat_completion = await openai.ChatCompletion.acreate(**chat_parameters)
        text = chat_completion.choices[0].message.content

        return RESTfulAgentText(response=text)


if __name__ == '__main__':
    agent = YourAgent()
    agent.run(host="0.0.0.0", port=int(os.getenv("AGENT_PORT")))
