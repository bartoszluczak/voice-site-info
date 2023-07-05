import json
import os
import datetime
import time

import asyncio as asyncio
import openai
    import requests
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain.agents import ConversationalChatAgent, AgentExecutor
from langchain.callbacks.manager import trace_as_chain_group
from langchain.chat_models import ChatOpenAI
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain.schema import messages_from_dict, SystemMessage
from langchain.tools import GooglePlacesTool, Tool
from langchain.utilities import GooglePlacesAPIWrapper
from supabase import create_client
from twilio.twiml.messaging_response import MessagingResponse

load_dotenv()

url = os.getenv("SUPABASE_URL_CHAT")
key = os.getenv("SUPABASE_KEY_CHAT")
supabase = create_client(url, key)
openai.api_key = os.getenv('OPENAI_API_KEY')
openai.api_base = "https://oai.hconeai.com/v1"

app = Flask(__name__)
cors = CORS(app)
# cors = CORS(app, origins=["*", "http://localhost:6001", "https://88db-35-208-224-244.ngrok-free.app"], allow_headers=["Access-Control-Allow-Origin", "Content-Type"])

respTimes = list()
initial_prompt_l = """
Your task is to act as a 411-type assistant called “Bart” to assist callers who are dialling into the service to inquire about phone numbers, addresses, and businesses near them. I will be chatting with you here as a caller in a role play.
The people calling in are an older demographic greater than 60 years of age. You are interacting with them via voice, so use clear speech and be concise while still offering multiple business options to their questions when possible.
If you do not know the location of the user, always ask them for their location in order to provide relevant recommendations.
You are to introduce yourself as Paige. Never speak for anyone else, and don’t break character.
After introducing yourself, inquire how you may help the caller. Always end the introduction asking where they are located so you can use that information to provide businesses closest to them.
Always provide answers of relevant businesses that are closest to the caller. If the caller says at any point those businesses are too far, ask the caller for a more precise location of where they reside in order for you to provide better business recommendations.
Whenever needed, ask multiple clarifying questions relevant to the caller’s question before moving forward.
Never assume the caller’s gender. You may ask for their name if you’d like to address them.
If you don’t understand the caller’s request, never guess or make up responses. Simply ask them to clarify their request when you are not certain.
If the caller’s questions are not relevant to finding information about businesses through this 411-type service, you may engage in their non-relevant questions for one question maximum. After that, you should say that you have been trained to answer about phone numbers, addresses, and businesses near you.
If the caller asks for an agent or to talk with a person, inform them we don’t offer that service, and ask them, “Would you like to receive an SMS instead where you can continue the conversation?”
If a caller is unhappy with the results, tell them you are a new assistant and still learning. You’d be glad to take their feedback on how to improve the service.
Remember to always be friendly and helpful in your interactions with callers, and to follow all these rules and guidelines provided to you.
Don’t get stuck in “I don’t understand the question.” After not understanding a caller’s questions three times, ask them, “Would you like to receive an SMS where you can continue the conversation?”.
Your ultimate goal is to provide a positive experience for the caller, understand the caller’s request, and process their request by using tools to provide them with relevant information to their questions.
When sharing recommendations of businesses that are retrieved, only share a list of 3 results maximum and sort the recommendations by proximity to the caller’s location. Do not share the location or phone numbers of the businesses at this stage when offering multiple recommendations unless the caller requests it. At this stage share only the name of the business, the distance from caller’s location, and the number of total reviews as well as rating the business has.
After sharing the list of 3 business results at maximum, follow this by asking the caller which of the businesses they want to know more about. You can then retreive more information about a business. This is useful if you want to provide the caller more information about a specific business.
If you can not retrieve any information relevant to the caller’s request, let the user know you found no matches.
#### Example of conversation: User> I am looking for the nearest car repair Bot> Great, what is your location? User> I am on 58th street and 9th avenue Bot> Which city? User> New York Bot> I have found 3 car repair centers which are close to you. The closest is Business X, 0.5km from you with a 4.6 rating out of 5 and 320 reviews. Do you want the phone number? User> Yeah Bot> Please write it down. 912 323 2323. Repeat 912 323 2323. Anything else? User> No, thanks! Bot> My pleasure!
------TOOLS
You can use tools to look up information that may be helpful in answering the caller’s questions about businesses. The tools the human can use are:
1. Google Places Search
2. Yelp Plugin
3. Retrieval Plugin
"""

initial_prompt_s = "Your task is to engage in helpful and friendly conversations to assist a person who is calling to know phone numbers, address, business and more about business near them"

history = ChatMessageHistory()

chatgpt_functions = [
    {
        "name": "search_google_places",
        "description": "Get places and businesses near the given location return only one result",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The street, city and state, e.g. San Francisco, CA",
                },
                "place": {
                    "type": 'string',
                    "description": "The type or name of the place or business"
                }
            },
            "required": ["place", "location"],
        },
    }
]

llm = ChatOpenAI(temperature=0.6, model_name="gpt-3.5-turbo-0613", request_timeout=120, headers={
        "Helicone-Auth": "Bearer sk-guw2jga-rznuisi-qylitfa-i3sux6a",
        "Helicone-Property-App": "chat",
    })

gplaceapi = GooglePlacesAPIWrapper(top_k_results=1)
search = GooglePlacesTool(api_wrapper=gplaceapi)

tools = [
    Tool(
        name="Google Places Search",
        func=search.run,
        description="useful for when you need to answer questions about current events or the current state of the world"
    ),
]


def openai_chat_agent(messages):
    start = time.time()
    global respTimes
    chatgpt_response = openai.ChatCompletion.create(model='gpt-3.5-turbo-0613', messages=messages,
                                                    functions=chatgpt_functions,
                                                    function_call="auto",
                                                    stream=False, max_tokens=50,
                                                    headers={
                                                        "Helicone-Auth": "Bearer sk-guw2jga-rznuisi-qylitfa-i3sux6a",
                                                        "Helicone-Property-App": "chat",
                                                    }
                                                    )
    response_message = chatgpt_response["choices"][0]["message"]


    if response_message.get("function_call"):
        startFnCall = time.time()
        available_functions = {
            "search_google_places": search_google_places,
        }
        function_name = response_message["function_call"]["name"]
        fuction_to_call = available_functions[function_name]
        function_args = json.loads(response_message["function_call"]["arguments"])
        function_response = fuction_to_call(
            place=function_args.get("place"),
            location=function_args.get("location"),
        )
        partTime = time.time()

        messages.append(response_message)
        respTimes.append(partTime - startFnCall)

        messages.append(
            {
                "role": "function",
                "name": function_name,
                "content": function_response,
            }
        )
        second_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            messages=messages,
            headers={
                "Helicone-Auth": "Bearer sk-guw2jga-rznuisi-qylitfa-i3sux6a",
                "Helicone-Property-App": "chat",
            }
        )
        endFnCall = time.time()
        respTimes.append(endFnCall - startFnCall)
        response_message = second_response["choices"][0]["message"]
    # respTimes.append(time.time() - start)
    return response_message


def langchain_chat_agent(messages, input_msg):

    for msg in messages:
        if msg['role'] == 'user':
            history.add_user_message(msg['content'])
        elif msg['role'] == 'system':
            continue
        elif msg['role'] == 'assistant':
            history.add_ai_message(msg['content'])

    memory = ConversationBufferMemory(chat_memory=history,
                                      return_messages=True, memory_key="chat_history")

    custom_agent = ConversationalChatAgent.from_llm_and_tools(llm=llm, tools=tools, system_message=initial_prompt_l,
                                                              handle_parsing_errors=True)
    agent_executor = AgentExecutor.from_agent_and_tools(agent=custom_agent, tools=tools, memory=memory,
                                                        handle_parsing_errors=True )
    agent_executor.verbose = False


    chatgpt_response = agent_executor.run(input=input_msg)

    return chatgpt_response


def search_google_places(place, location):
    url = f"https://maps.googleapis.com/maps/api/place/textsearch/json?query={place + location}&key={os.getenv('GPLACES_API_KEY')}&radius=50"

    payload = {}
    headers = {}

    response = requests.request("GET", url, headers=headers, data=payload)
    results = []
    dicts = json.loads(response.text)
    print(dicts)
    for result in dicts['results']:
        results.append({"name": result['name'], "address": result['formatted_address'], "place_id": result["place_id"],
                        "rating": result["rating"]})
    res = json.dumps(results[0:3])
    print(res)
    return res

def update_db(user_msg_history, uuid, messages, conversation_name, conversation_model):
    if len(user_msg_history.data) > 0 and user_msg_history.data[0]['id']:
        print("Data exist")

        data = supabase.table("chat_conversations").update(
            {"id": uuid, "last_update": str(datetime.datetime.now()), "messages": messages, "resp_time": respTimes}).eq("id", uuid).execute()
    else:
        print("Data not exist")
        data, count = supabase.table('chat_conversations').insert(
            {"id": uuid, "created_at": str(datetime.datetime.now()), "messages": messages,
             "chat_name": conversation_name, "agent_model": conversation_model, "resp_time": respTimes}).execute()

@app.route("/sms", methods=['POST'])
def sms_chatgpt():
    messages = []
    user_number = request.form["From"]
    user_msg = request.form['Body'].lower()

    user_msg_history = supabase.table('conversations').select('id', 'conversations').eq('phone_number',
                                                                                        user_number).execute()

    if len(user_msg_history.data) > 0:
        msg_history = user_msg_history.data[0]['conversations']
        if len(msg_history) > 0:
            dicts = json.loads(msg_history)

            for msg in dicts:
                messages.append(msg)

    else:
        messages.append({"role": "system", "content": initial_prompt_l})

    inb_msg = f"Message from number {user_number}, message content {user_msg}"
    messages.append({"role": "user", "content": inb_msg})

    functions = [
        {
            "name": "search_google_places",
            "description": "Get places and businesses near the given location return only one result",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The street, city and state, e.g. San Francisco, CA",
                    },
                    "place": {
                        "type": 'string',
                        "description": "The type or name of the place or business"
                    }
                },
                "required": ["place", "location"],
            },
        }
    ]
    chatgpt_response = openai.ChatCompletion.create(model='gpt-3.5-turbo-0613', messages=messages,
                                                    functions=functions,
                                                    function_call="auto",
                                                    stream=False, max_tokens=50,
                                                    headers={
                                                        "Helicone-Auth": "Bearer sk-guw2jga-rznuisi-qylitfa-i3sux6a",
                                                        "Helicone-Property-App": "chat",
                                                    }
                                                    )
    response_message = chatgpt_response["choices"][0]["message"]

    if response_message.get("function_call"):
        available_functions = {
            "search_google_places": search_google_places,
        }
        function_name = response_message["function_call"]["name"]
        fuction_to_call = available_functions[function_name]
        function_args = json.loads(response_message["function_call"]["arguments"])
        function_response = fuction_to_call(
            place=function_args.get("place"),
            location=function_args.get("location"),
        )
        messages.append(response_message)
        messages.append(
            {
                "role": "function",
                "name": function_name,
                "content": function_response,
            }
        )
        second_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            messages=messages,
            headers={
                "Helicone-Auth": "Bearer sk-guw2jga-rznuisi-qylitfa-i3sux6a",
                "Helicone-Property-App": "chat",
            }
        )

        response_message = second_response["choices"][0]["message"]
    messages.append(response_message)

    if len(user_msg_history.data) > 0 and user_msg_history.data[0]['id']:
        user_id = user_msg_history.data[0]['id']
        data = supabase.table("conversations").update(
            {"id": user_id, "last_update": str(datetime.datetime.now()), "phone_number": user_number,
             "conversations": messages}).eq("phone_number",
                                            user_number).execute()
    else:
        data, count = supabase.table('conversations').insert(
            {"created_at": str(datetime.datetime.now()), "phone_number": user_number,
             "conversations": messages}).execute()

    resp = MessagingResponse()
    resp.message(response_message.content)

    return resp


@app.route("/get_messages", methods=["GET"])
def get_messages():
    uuid = request.args.get('conv_uuid')
    msg_history = supabase.table('chat_conversations').select('id', 'messages', 'agent_model', 'resp_time').eq('id', uuid).execute()

    if len(msg_history.data) > 0:
        return json.dumps(msg_history.data)
    else:
        return "No data found for given conversation UUID"


@app.route("/chat", methods=['POST'])
def chat_chatgpt():
    print("CHAT")
    start = time.time()
    messages = []
    global respTimes
    uuid = request.form["conv_uuid"]
    conversation_model = request.form['conv_agent']
    conversation_name = request.form['conv_name']
    user_msg = request.form['body'].lower()

    user_msg_history = supabase.table('chat_conversations').select('id', 'messages', 'agent_model', 'resp_time').eq('id',
                                                                                                       uuid).execute()

    if len(user_msg_history.data) > 0:
        print("IF")
        print(messages)
        conversation_model = user_msg_history.data[0]['agent_model']
        msg_history = user_msg_history.data[0]['messages']
        respTimes = user_msg_history.data[0]['resp_time']
        if len(msg_history) > 0:
            for msg in msg_history:
                messages.append(msg)

    elif 'initial_input' in user_msg:
        print("Edit initial input")
        new_initial_prompt = user_msg[user_msg.index("=")+1:]
        print(new_initial_prompt)
        new_msg = {"role": "system", "content": new_initial_prompt}
        messages.append(new_msg)
        respTimes.append(0)
        update_db(user_msg_history, uuid, messages, conversation_name, conversation_model)
        return json.dumps({"resp_time": time.time() - start, "data": new_msg})

    else:
        print("ELSE")
        print(messages)
        messages.append({"role": "system", "content": initial_prompt_l})
        respTimes.append(0)

    inp_msg = f"{user_msg}"
    messages.append({"role": "user", "content": inp_msg})
    respTimes.append(0)

    chatgpt_response = ''

    if conversation_model == 'openai':
        print("RUN OPENAI")
        chatgpt_response = openai_chat_agent(messages)

    elif conversation_model == 'langchain':
        print("RUN LANGCHAIN")
        langchain_response = langchain_chat_agent(messages, inp_msg)
        chatgpt_response = {"role": 'assistant', 'content': langchain_response}

    end = time.time()
    messages.append(chatgpt_response)
    respTimes.append(end - start)

    update_db(user_msg_history, uuid, messages, conversation_name, conversation_model)
    # if len(user_msg_history.data) > 0 and user_msg_history.data[0]['id']:
    #     print("Data exist")
    #
    #     data = supabase.table("chat_conversations").update(
    #         {"id": uuid, "last_update": str(datetime.datetime.now()), "messages": messages, "resp_time": respTimes}).eq("id", uuid).execute()
    # else:
    #     print("Data not exist")
    #     data, count = supabase.table('chat_conversations').insert(
    #         {"id": uuid, "created_at": str(datetime.datetime.now()), "messages": messages,
    #          "chat_name": conversation_name, "agent_model": conversation_model, "resp_time": respTimes}).execute()


    return json.dumps({"resp_time": end - start, "data": chatgpt_response})
    # return jsonify({"resp_time": end - start, "data": chatgpt_response})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=6001, debug=False)
