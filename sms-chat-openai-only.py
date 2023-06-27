import json
import os
import datetime
import openai
import requests
from dotenv import load_dotenv
from flask import Flask, request
from langchain.memory import ChatMessageHistory
from supabase import create_client
from twilio.twiml.messaging_response import MessagingResponse

load_dotenv()

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")
supabase = create_client(url, key)
openai.api_key = os.getenv('OPENAI_API_KEY')

app = Flask(__name__)

initial_prompt_l = """
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
    21. Always return exactly one place/business.
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
    TOOLS
    ------
    Assistant can ask the user to use tools to look up information that may be helpful in answering the users original question. The tools the human can use are:
    1. Google Places Search: useful for when you need to answer questions about current events or the current state of the world
    """

initial_prompt_s = "Your task is to engage in helpful and friendly conversations to assist a person who is calling to know phone numbers, address, business and more about business near them"

history = ChatMessageHistory()

def search_google_places(place, location):
    url = f"https://maps.googleapis.com/maps/api/place/textsearch/json?query={place+location}&key={os.getenv('GPLACES_API_KEY')}&radius=50"

    payload = {}
    headers = {}

    response = requests.request("GET", url, headers=headers, data=payload)
    results = []
    dicts = json.loads(response.text)
    for result in dicts['results']:
        results.append({"name": result['name'], "address": result['formatted_address'], "place_id": result["place_id"], "rating": result["rating"]})
    res = json.dumps(results[0:3])
    return res


@app.route("/sms", methods=['POST'])
def sms_chatgpt():
    messages = []
    user_number = request.form["From"]
    user_msg = request.form['Body'].lower()

    user_msg_history = supabase.table('conversations').select('id', 'conversations').eq('phone_number',
                                                                                        user_number).execute()

    if len(user_msg_history.data) > 0:
        msg_history = user_msg_history.data[0]['conversations']
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
                                                    stream=False, max_tokens=50)
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
    return resp.message(response_message.content)

@app.route("/chat", methods=['POST'])
def chat_chatgpt():
    messages = []
    user_number = request.form["From"]
    user_msg = request.form['Body'].lower()

    user_msg_history = supabase.table('conversations').select('id', 'conversations').eq('phone_number',
                                                                                        user_number).execute()

    if len(user_msg_history.data) > 0:
        msg_history = user_msg_history.data[0]['conversations']
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
                                                    stream=False, max_tokens=50)
    response_message = chatgpt_response["choices"][0]["message"]
    # messages.append({"role": "assistant", "content": response_message})

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

    return response_message.content

if __name__ == "__main__":
    app.run(host="localhost", port=6000, debug=False)
