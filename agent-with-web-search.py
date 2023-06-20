import asyncio
import logging
import os
import openai
from vocode.streaming.models.agent import RESTfulAgentOutput, RESTfulAgentText
from vocode.streaming.user_implemented_agent.restful_agent import RESTfulAgent
from dotenv import load_dotenv

load_dotenv()

from langchain.tools import Tool, GooglePlacesTool

messages = []
search_results = ''

def check_if_string_contains_details(string):
    contains_details = False
    terms = ["details", "send", "sms"];
    words = string.lower().split()
    for term in terms:
        if term in words:  # see if one of the words in the sentence is the word we want
            contains_details = True

    return contains_details


class YourAgent(RESTfulAgent):
    def __init__(self):
        super().__init__()

    openai.api_type = "open_ai"
    openai.api_base = "https://api.openai.com/v1"
    openai.api_version = None
    openai.api_key = os.getenv('OPENAI_API_KEY')

    async def respond(self, input: str, conversation_id: str) -> RESTfulAgentOutput:
        global messages
        global search_results

        print('AGENT')
        print("USER INPUT: ", input)
        print("MESSAGES: ", messages)

        messages.append({"role": "system",
                         "content": "You are a assistant answer the question by using the provided context."})

        messages.append(
            {"role": 'user', "content": f"Check if user '{input} want to get details and return true or false"})

        is_input_contains_details = check_if_string_contains_details(input)

        if is_input_contains_details:
            print("DETAILS: ", is_input_contains_details)
            prompt = f"""
             Context: Here are list of ordered places given by Google Places {search_results}
             ###
             Question: Will send you details vis SMS after this call
            """
        elif 'bye' in input:
            prompt = f"""
            Context: Here are list of ordered places given by Google Places {search_results}
            ####
            Question: Write some response about thanking for a call and that you are glad that you helped
            """
        else:
            search = GooglePlacesTool()
            search_results = search.run(input)

            prompt = f"""
                Context: Here are list of ordered places given by Google Places {search_results}
                ###
                Question: Here is the user input {input} take first of the result in context and write short description about it with name and adress. Propose also that you can send details via SMS.
                 """

        messages.append({"role": "user", "content": prompt})

        chat_parameters = {
            "messages": messages,
            "max_tokens": 100,
            "temperature": 0.6,
            "model": os.getenv("OPENAI_MODEL")
        }
        #
        chat_completion = await openai.ChatCompletion.acreate(**chat_parameters)
        text = chat_completion.choices[0].message.content
        #
        messages.append({"role": "system", "content": text})
        print("COMPLETION: ", text)

        return RESTfulAgentText(response=text)


if __name__ == '__main__':
    agent = YourAgent()
    agent.run(host="localhost", port=int(os.getenv("AGENT_PORT")))
