import json
import os
import openai
from langchain.agents import initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryBufferMemory
from langchain.tools import GooglePlacesTool, Tool
from langchain.utilities import GooglePlacesAPIWrapper
from vocode.streaming.models.agent import RESTfulAgentOutput, RESTfulAgentText
from vocode.streaming.user_implemented_agent.restful_agent import RESTfulAgent
from dotenv import load_dotenv
import requests

load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.langchain.plus"
os.environ["LANGCHAIN_API_KEY"] = os.getenv('LANGCHAIN_API_KEY')
os.environ["LANGCHAIN_SESSION"] = os.getenv('LANGCHAIN_SESSION_VOICE')

initial_prompt = """
Your task is to act as a 411-type assistant called “Paige” to assist callers who are dialling into the service to inquire about phone numbers, addresses, and businesses near them. I will be chatting with you here as a caller in a role play.
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

llm = ChatOpenAI(temperature=0.6, model_name="gpt-3.5-turbo-0613")
memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=100, memory_key="chat_history", return_messages=True)
memory.chat_memory.add_ai_message(initial_prompt)

gplaceapi = GooglePlacesAPIWrapper(top_k_results=1)
search = GooglePlacesTool(api_wrapper=gplaceapi)


def check_if_string_contains_details(string):
    contains_details = False
    terms = ["details", "send", "sms"]
    words = string.lower().split()
    for term in terms:
        if term in words:  # see if one of the words in the sentence is the word we want
            contains_details = True

    return contains_details


tools = [
    Tool(
        name="Google Places Search",
        func=search.run,
        description="useful for when you need to answer questions about current events or the current state of the world"
    ),
    Tool(
        name="Check if user wants details",
        func=check_if_string_contains_details,
        description="useful for when you need to answer questions if user wants details about business"
    )
]

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


def search_google_places(place, location):
    url = f"https://maps.googleapis.com/maps/api/place/textsearch/json?query={place + location}&key={os.getenv('GPLACES_API_KEY')}&radius=50"

    payload = {}
    headers = {}

    response = requests.request("GET", url, headers=headers, data=payload)
    results = []
    dicts = json.loads(response.text)
    for result in dicts['results']:
        results.append({"name": result['name'], "address": result['formatted_address'], "place_id": result["place_id"]})
    res = json.dumps(results[0:3])
    return res


def openai_chat_agent(messages):
    chatgpt_response = openai.ChatCompletion.create(model='gpt-3.5-turbo-16k', messages=messages,
                                                    functions=chatgpt_functions,
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
            model="gpt-3.5-turbo-16k",
            messages=messages,
        )

        response_message = second_response["choices"][0]["message"]

    return response_message


messages = []


class YourAgent(RESTfulAgent):

    def __init__(self):
        super().__init__()

    openai.api_type = "open_ai"
    openai.api_base = "https://api.openai.com/v1"
    openai.api_version = None
    openai.api_key = os.getenv('OPENAI_API_KEY')

    async def respond(self, input: str, conversation_id: str) -> RESTfulAgentOutput:
        global messages
        print('AGENT')
        print("USER INPUT: ", input)

        user_input = f"""
        Follow the instructions in the System role always.
        Keep those instructions in context all the time.
        ###
        User: {input}
        Completion: 
        """

        messages.append({"role": "system", "content": initial_prompt})
        messages.append({"role": "user", "content": user_input})

        chatgpt_response = openai_chat_agent(messages)
        messages.append(chatgpt_response)
        print("COMPLETION: ", chatgpt_response.content)

        if len(chatgpt_response.content) > 0:
            return RESTfulAgentText(response=chatgpt_response.content)
        else:
            return RESTfulAgentText(response="I'm sorry but I don't have an answer to your question")


if __name__ == '__main__':
    agent = YourAgent()
    agent.run(host="0.0.0.0", port=int(os.getenv("AGENT_PORT")))
