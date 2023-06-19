import os

import requests
from bs4 import BeautifulSoup

webpage_address = "https://stockbuddyapp.com/"

response = requests.get(webpage_address)
soup = BeautifulSoup(response.text, 'html.parser')

wrapper = soup.find("div", class_="elementor")

sections = wrapper.find_all("div", class_="elementor-container")
for section in sections:
    print(section.get_text())
