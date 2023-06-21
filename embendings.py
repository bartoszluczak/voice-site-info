import os

import openai
import json
import asyncio
import pandas as pd
import numpy as np
import csv

from openai.embeddings_utils import get_embedding

prompt = """
Your task is to engage in helpful and friendly conversations to assist a person who is calling to know phone numbers, address, business and more about business near them.
- This conversion is via voice, so be VERY concise. No long answers. Shorten them.
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
###
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

async def get_embeddings(input_text):
    return openai.Embedding.create(
        model="text-embedding-ada-002",
        input=input_text
    )


# def get_embedding(text, model="text-embedding-ada-002"):
#    text = text.replace("\n", " ")
#    return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']


if __name__ == '__main__':
    openai.api_key = "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"

    output = asyncio.run(get_embeddings(prompt))
    print(output.data[0].embedding)
    embeddings = pd.DataFrame(output.data[0].embedding)

    # df['embeddings'] = df.combined.apply(lambda x: get_embedding(x, model='text-embedding-ada-002'))

    embeddings.to_csv("prompt-embeddings.csv", index=False)

    # with open('./website.json', 'r') as f:
    #     data = json.load(f)
    # with open('website-summary.csv', 'w', newline='') as file:
    #     writer = csv.writer(file)
    #     field = ["ID", "Title", "Content"]
    #     writer.writerow(field)
    #     i=1
    #     for key, value in data.items():
    #         print(value['content'])
    #         title = value['title'].replace(",", "").replace("\n", "")
    #         newValue = value['content'].replace(",", "").replace("\n", "")
    #         # newStr = "Title: {title} \n ### \n Content: {content}".format(title=value['title'], content=value['content'])
    #
    #         writer.writerow([i, title, newValue])
    #         i += 1

        # df['ada_embedding'] = df.combined.apply(lambda x: get_embedding(x, model='text-embedding-ada-002'))
        # df.to_csv('output/embedded_1k_reviews.csv', index=False)
#         emb = asyncio.run(get_embeddings(value['content']))
#         newDict[key] = emb
#
# print(newDict)

####################################
###### PREPARE EMBEDDINGS
# datafile_path = "data/fine_food_reviews_with_embeddings_1k.csv"
#
# input_datapath = "./website-summary.csv"  # to save space, we provide a pre-filtered dataset
# df = pd.read_csv(input_datapath, index_col=0)
# df = df[["Title", "Content"]]
# df = df.dropna()
# # print(df)
# df["Combined"] = (
#     "Title: " + df.Title.str.strip() + "; Content: " + df.Content.str.strip()
# )
#
# embedding_model = "text-embedding-ada-002"
#
# df["embedding"] = df.Combined.apply(lambda x: get_embedding(x, engine=embedding_model))
# df.to_csv("./website-summary-with-embeddings.csv")
# ##############################################


#####################################################
### SEARCH ALGORITHM
datafile_path = "./website-summary-with-embeddings.csv"

df = pd.read_csv(datafile_path)
OPENAI_API_KEY

from openai.embeddings_utils import get_embedding, cosine_similarity

# # search through the reviews for a specific product
def search_reviews(df, product_description, n=3, pprint=True):
    product_embedding = get_embedding(
        product_description,
        engine="text-embedding-ada-002"
    )

    print(product_embedding)
    # df["similarity"] = df.embedding.apply(lambda x: cosine_similarity(x, product_embedding))
    #
    # results = (
    #     df.sort_values("similarity", ascending=False)
    #     .head(n)
    #     .Combined.str.replace("Title: ", "")
    #     .str.replace("; Content:", ": ")
    # )
    # if pprint:
    #     for r in results:
    #         # print(r[:200])
    #         print(r)
    # return results


results = search_reviews(df, "pricing", n=5)