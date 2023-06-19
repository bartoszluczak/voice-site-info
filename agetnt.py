import os

import numpy as np
import openai
import asyncio

import pandas as pd
import requests
from openai.embeddings_utils import get_embedding
from vocode.streaming.models.agent import RESTfulAgentOutput, RESTfulAgentText
from vocode.streaming.user_implemented_agent.restful_agent import RESTfulAgent
from openai.embeddings_utils import get_embedding, cosine_similarity
from dotenv import load_dotenv
from bs4 import BeautifulSoup
load_dotenv()





def search_reviews(df, search_input, n=3, pprint=True):
    search_embedding = get_embedding(
        search_input,
        engine="text-embedding-ada-002"
    )

    df["similarity"] = df.embedding.apply(lambda x: cosine_similarity(x, search_embedding))

    results = (
        df.sort_values("similarity", ascending=False)
        .head(n)
        .Combined.str.replace("Title: ", "")
        .str.replace("; Content:", ": ")
    )

    return results

class YourAgent(RESTfulAgent):
    def __init__(self, ):
        super().__init__()

    openai.api_type = "open_ai"
    openai.api_base = "https://api.openai.com/v1"
    openai.api_version = None
    openai.api_key = os.getenv('OPENAI_API_KEY')

    async def respond(self, input: str, conversation_id: str) -> RESTfulAgentOutput:
        print('AGENT')
        messages = [{"role": "system", "content": "You are a assistant answer the question by using the provided context. If your are unable to answer the question using the provided context, say ‘I don’t know’"}]
        results = search_reviews(df, input, n=3)

        prompt = f"""
                Context: {results}
                ###
                Question: {input}
        """

        messages.append({"role": "user", "content": prompt})
        print("PROMPT: ", prompt)

        chat_parameters = {
            "messages": messages,
            "max_tokens": 100,
            "temperature": 0.8,
            "model": os.getenv("OPENAI_MODEL")
        }

        chat_completion = await openai.ChatCompletion.acreate(**chat_parameters)
        text = chat_completion.choices[0].message.content

        messages.append({"role": "system", "content": text})
        print("COMPLETION: ", text)
        return RESTfulAgentText(response=text)


if __name__ == '__main__':
    datafile_path = "./website-summary-with-embeddings.csv"
    df = pd.read_csv(datafile_path)
    df["embedding"] = df.embedding.apply(eval).apply(np.array)

    print(df)
    agent = YourAgent()
    agent.run(host="localhost", port=int(os.getenv("AGENT_PORT")))
