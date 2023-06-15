import openai
from vocode.streaming.models.agent import RESTfulAgentOutput, RESTfulAgentEnd, RESTfulAgentText, ChatGPTAgentConfig
from vocode.streaming.user_implemented_agent.restful_agent import RESTfulAgent

web_page = "https://stockbuddyapp.com/"

class YourAgent(RESTfulAgent):
    def __init__(self,):
        super().__init__()

    openai.api_type = "open_ai"
    openai.api_base = "https://api.openai.com/v1"
    openai.api_version = None
    openai.api_key = "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"


    async def respond(self, input: str, conversation_id: str) -> RESTfulAgentOutput:
        messages = [
                {"role": "system", "content": "Hello how can I help you"},
                {"role": "user", "content": "Answer to user question " + input + "only from information from " + web_page}
                ]

        chat_parameters = {
                "messages": messages,
                "max_tokens": 100,
                "temperature": 0.8,
                "model": "gpt-3.5-turbo"
                }

        chat_completion = await openai.ChatCompletion.acreate(**chat_parameters)
        text = chat_completion.choices[0].message.content
        print(chat_completion)
        print(text)


        return RESTfulAgentText(response=text)


if __name__ == '__main__':
    agent = YourAgent()
    agent.run(port=4000)
