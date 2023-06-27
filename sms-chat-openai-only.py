import json
import os
import datetime
import openai
from dotenv import load_dotenv
from flask import Flask, request
from langchain import OpenAI
from langchain.agents import Tool, ConversationalChatAgent, AgentExecutor
from langchain.callbacks.manager import trace_as_chain_group
from langchain.schema import messages_from_dict, messages_to_dict

from langchain.tools import GooglePlacesTool
from langchain.utilities import GooglePlacesAPIWrapper
from langchain.chat_models import ChatOpenAI
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from supabase import create_client

load_dotenv()

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")
supabase = create_client(url, key)
openai.api_key = os.getenv('OPENAI_API_KEY')
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.langchain.plus"
os.environ["LANGCHAIN_API_KEY"] = os.getenv('LANGCHAIN_API_KEY')
os.environ["LANGCHAIN_SESSION"] = os.getenv('LANGCHAIN_SESSION_SMS')

app = Flask(__name__)

initial_prompt2 = """
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

initial_prompt = "Your task is to engage in helpful and friendly conversations to assist a person who is calling to know phone numbers, address, business and more about business near them"

history = ChatMessageHistory()
llm = ChatOpenAI(temperature=0.6, model_name="gpt-3.5-turbo-0613", request_timeout=120, max_tokens=20)

gplaceapi = GooglePlacesAPIWrapper(top_k_results=1)
search = GooglePlacesTool(api_wrapper=gplaceapi)

tools = [
    Tool(
        name="Google Places Search",
        func=search.run,
        description="useful for when you need to answer questions about current events or the current state of the world"
    ),
]


def search_google_places(place, location):
    print(place, location)
    res = json.dumps(search.run(place + location))
    print(res)
    return res


@app.route("/sms", methods=['POST'])
def chatgpt():
    user_number = request.form["From"]
    user_msg = request.form['Body'].lower()

    user_msg_history = supabase.table('conversations').select('id', 'conversations').eq('phone_number',
                                                                                        user_number).execute()

    if len(user_msg_history.data) > 0:
        msg = user_msg_history.data[0]
        dicts = json.loads(msg['conversations'])
        new_messages = messages_from_dict(dicts)
        history.messages = new_messages

    history.add_user_message(user_msg)
    # memory = ConversationBufferMemory(chat_memory=history,
    #                                   return_messages=True, memory_key="chat_history")

    # custom_agent = ConversationalChatAgent.from_llm_and_tools(llm=llm, tools=tools, system_message=initial_prompt, handle_parsing_errors=True,)
    # agent_executor = AgentExecutor.from_agent_and_tools(agent=custom_agent, tools=tools, memory=memory, handle_parsing_errors=True,)
    # agent_executor.verbose = False

    inb_msg = f"Message from number {user_number}, message content {user_msg}"
    chatgpt_response = ''
    # with trace_as_chain_group("conversation_with_bot") as group_manager:
    #     chatgpt_response = agent_executor.run(input=inb_msg, tags=['user_bot_conversation', str(user_number)],
    #                                           callbacks=group_manager)

    dicts = messages_to_dict(history.messages)
    messages = []
    messages.append({"role": "system", 'content': initial_prompt2})
    for single_msg in dicts:
        role = 'system' if single_msg['type'] == 'ai' else 'user' if single_msg['type'] == 'human' else 'assistant'
        messages.append({'role': role, 'content': single_msg['data']['content']})

    functions = [
                                                        {
                                                            "name": "search_google_places",
                                                            "description": "Get places and businesses near the given location",
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
        print("IN IF")
        available_functions = {
            "search_google_places": search_google_places,
        }  # only one function in this example, but you can have multiple
        function_name = response_message["function_call"]["name"]
        fuction_to_call = available_functions[function_name]
        function_args = json.loads(response_message["function_call"]["arguments"])
        function_response = fuction_to_call(
            place=function_args.get("place"),
            location=function_args.get("location"),
        )

        messages.append(response_message)  # extend conversation with assistant's reply
        messages.append(
            {
                "role": "function",
                "name": function_name,
                "content": function_response,
            }
        )  # extend conversation with function response
        second_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            messages=messages,
        )  # get a new response from GPT where it can see the function response
        print(second_response)
        return second_response

    # history.add_ai_message(chatgpt_response)
    # dicts = messages_to_dict(history.messages)

    res = ""

    # for event in chatgpt_response:
    # #     # STREAM THE ANSWER
    #     print(event, end='', flush=True)  # Print the response
    # #     if event.choices[0].finish_reason != 'stop':
    #     text = event.choices[0].delta.content
    #     # res = res + text
    #     if len(text) > 0:
    #         return text
    #
    #
    # print("RES: " + res)

    if len(user_msg_history.data) > 0 and user_msg_history.data[0]['id']:
        user_id = user_msg_history.data[0]['id']
        data = supabase.table("conversations").update(
            {"id": user_id, "last_update": str(datetime.datetime.now()), "phone_number": user_number,
             "conversations": dicts}).eq("phone_number",
                                         user_number).execute()
    else:
        data, count = supabase.table('conversations').insert(
            {"created_at": str(datetime.datetime.now()), "phone_number": user_number,
             "conversations": dicts}).execute()
    # memory.clear()
    history.clear()
    # resp = MessagingResponse()
    # resp.message(chatgpt_response)

    return chatgpt_response
    # return chatgpt_response


if __name__ == "__main__":
    app.run(host="localhost", port=6000, debug=False)
