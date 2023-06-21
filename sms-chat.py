import asyncio

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from flask import Flask, request
from langchain.tools import GooglePlacesTool
from twilio.twiml.messaging_response import MessagingResponse
import os
import openai

load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')
print(openai.api_key)

app = Flask(__name__)

initial_prompt = """
    Your task is to engage in helpful and friendly conversations to assist a person who is calling to know phone numbers, address, business and more about business near them.
    - This conversion is via SMS, so be VERY concise. No long answers. Shorten them.
    - Use whole context of the conversation
    - Never speak for anyone else, don’t break character.
    - First, start by narrowing in on what the caller is looking for. Then, ask them if needed the location about the services they are looking for. Assume the caller’s location will always be the United States as the country, although they may provide an address for any other country too.
    - A helpful assistant will ask multiple clarifying questions relevant to the caller’s question before moving forward.
    - A helpful assistant will never assume the buyer's gender.
    - Don't guess, simply ask them when you are not certain!
    - You can engage in other topics, but just for three questions max. Then, you say that you have been trained to answer about phone numbers, addresses, and businesses near you.
    - If a buyer asks for an agent or talk with a person, tell them we don’t offer that service. Ask them, Would you like to receive an SMS where you can continue the conversation?”.
    - If a buyer is unhappy with the results, tell them you are a new assistant and still learning.
    - Remember to always be friendly and helpful in your interactions with callers, and to follow the other rules and guidelines provided to you.
    - Remember you are interacting with someone via voice, so be concise and to the point.
    - Don’t get stuck in “I don’t understand the question.” Ask them, Would you like to receive an SMS where you can continue the conversation?”.
    - Overall, my goal is to provide a positive experience for the caller, grab what the caller wants, use the search and details plugin to give them back that information, and tell them thank you for the call.
    - Understand and communicate in multiple languages, if you detect they speak in another language, switch to that.
    - Try very hard to use the less words as possible.
    - Then, use the Search plugin to fetch business information to the caller. When sharing lists of business, no need to share the location and rather how far they from their location. Ask them for which business they want to know more, and then use the details plugins for that. Share 3 maximum. No need to share the phone numbers when you are telling more than one business. Just their name, distance and anything relevant and short.
    - You can use the Details plugin to fetch more information about a business. This is useful if you want to show the caller more information about a business.
    ####
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
    """

messages = []


async def get_embeddings(input_text):
    return openai.Embedding.create(
        model="text-embedding-ada-002",
        input=input_text
    )


async def summarize(messages):
    new_prompt = [{"role": "system", "content": f"Summarize the whole conversation {messages}"}]

    summary = openai.ChatCompletion.create(
        model="gpt-4",
        messages=new_prompt,
        max_tokens=500,
        temperature=0.7
    )

    messages = [{"role": "system", "content": summary["choices"][0]["message"]}]


@app.route("/sms", methods=['POST'])
def chatgpt():
    inb_msg = request.form['Body'].lower()

    search = GooglePlacesTool()
    search_results = search.run(inb_msg)
    first_result = search_results.split("2.", 1)[0]

    first_result = first_result.replace("1. ", "Name: ")
    messages.insert(0, {"role": "system", "content": initial_prompt})

    prompt = f"""
            context: {first_result}
            ###
            Question: {inb_msg}
            Completion:
    """

    messages.append({"role": "user", "content": prompt})

    print(messages)

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        max_tokens=500,
        temperature=0.7
    )

    chatgpt_response = response["choices"][0]["message"]

    messages.append(chatgpt_response)

    resp = MessagingResponse()
    resp.message(chatgpt_response["content"])

    asyncio.run(summarize(summarize))
    return str(resp)


if __name__ == "__main__":
    # datafile_path = "./prompt-embeddings.csv"
    # df = pd.read_csv(datafile_path)
    # dfList = df.to_numpy().flatten().tolist()
    # print(dfList)
    app.run(host="localhost", port=6000, debug=False)
