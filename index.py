from flask import Flask
app = Flask(__name__)

@app.route("/api/python")
def hello_world():
    import os
    from vocode.streaming.telephony.hosted.outbound_call import OutboundCall
    from vocode.streaming.models.agent import ChatGPTAgentConfig
    from vocode.streaming.models.telephony import CallEntity, TwilioConfig
    from vocode.streaming.models.message import BaseMessage

    import vocode
    vocode.api_key = 'ef791bec2a57cf9cf5df29ae7b6b61b4'

    if __name__ == '__main__':
        call = OutboundCall(
            recipient=CallEntity(
                phone_number="+16203123467",
            ),
            caller=CallEntity(
                phone_number="+48726188555",
            ),
            agent_config=ChatGPTAgentConfig(
                initial_message=BaseMessage(text="Hello!"),
                prompt_preamble="Have a pleasant conversation about life",
            ),
            twilio_config=TwilioConfig(
                account_sid="AC968d1b274d38b28de856214f38e2ba36",
                auth_token="34516b27dab5978600e8bf98f74f4083",
            )
        )
        call.start()
        input("Press enter to end the call...")
        call.end()