from dotenv import load_dotenv
from langchain.tools import Tool, GooglePlacesTool

load_dotenv()

search = GooglePlacesTool()

print(search.run("restauracja, poznan grunwaldzka 80?"))
