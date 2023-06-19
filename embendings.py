import os

import openai
import json
import asyncio
import pandas as pd
import numpy as np
import csv

from openai.embeddings_utils import get_embedding


# async def get_embeddings(input_text):
#     return openai.Embedding.create(
#         model="text-embedding-ada-002",
#         input=input_text
#     )


# def get_embedding(text, model="text-embedding-ada-002"):
#    text = text.replace("\n", " ")
#    return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']


if __name__ == '__main__':
    openai.api_key = "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"

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
df["embedding"] = df.embedding.apply(eval).apply(np.array)

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