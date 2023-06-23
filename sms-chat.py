from dotenv import load_dotenv
from flask import Flask, request
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.tools import GooglePlacesTool
from langchain.utilities import GooglePlacesAPIWrapper
from twilio.twiml.messaging_response import MessagingResponse
import os
import openai
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory

load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.langchain.plus"
os.environ["LANGCHAIN_API_KEY"] = os.getenv('LANGCHAIN_API_KEY')
os.environ["LANGCHAIN_SESSION"] = "agent-prod"

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


llm = ChatOpenAI(temperature=0.6, model_name="gpt-3.5-turbo-0613")
# memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")
memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=100, memory_key="chat_history", return_messages=True)
memory.chat_memory.add_ai_message(initial_prompt)

gplaceapi = GooglePlacesAPIWrapper(top_k_results=1)
search = GooglePlacesTool(api_wrapper=gplaceapi)

tools = [
    Tool(
        name="Google Places Search",
        func=search.run,
        description="useful for when you need to answer questions about current events or the current state of the world"
    ),
]


@app.route("/sms", methods=['POST'])
def chatgpt():
    user_number = request.form["From"]
    user_msg = request.form['Body'].lower()

    inb_msg = f"Message from number {user_number}, message content {user_msg}"
    agent_chain = initialize_agent(tools, llm, agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, verbose=True,
                                   memory=memory)

    chatgpt_response = agent_chain.run(input=inb_msg)

    resp = MessagingResponse()
    resp.message(chatgpt_response)

    return str(resp)


if __name__ == "__main__":
    app.run(host="localhost", port=6000, debug=False)
