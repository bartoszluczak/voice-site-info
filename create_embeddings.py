
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

import os
from supabase import create_client, Client

url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(url, key)

# DB_CONNECTION = "postgresql://postgres:TFemCVdnF5SS9Q7bCZ6YUj5tf7@db.dqgbusdsghtmwovrkpna.supabase.co:5432/postgres"

print(supabase)

# vx = vecs.create_client(DB_CONNECTION)
# # db = vx.create_collection(name="documents", dimension=4)
# # db = vx.get_collection(name="documents")
# db2 = vx.get_collection(name="docs")
#
# datafile_path = "./website-summary-with-embeddings.csv"
#
# df = pd.read_csv(datafile_path)
# df["embedding"] = df.embedding.apply(eval).apply(np.array)
# # print(df)
#
# vectorsArray=[]

# for index, row in df.iterrows():
#    #     # ("vec1", [0.1, 0.2, 0.3], {"year": 1990})
#    #     vectorsArray.append((str(index), row["embedding"], {"content": row["Title"]}))
#    data = {"id": str(index), "title": row["Title"], "content": row["Content"], "combined": row["Combined"], "embedding": row["embedding"].tolist()}
#    print(type(data))
#    supabase.table('pages').insert(data).execute()
# #
#
# print(vectorsArray[0])
# #
# # ## add records to the collection
# db2.upsert(
#         vectors=vectorsArray[0]
# )

# response = supabase.table('pages').select("""""").execute()
# print(response)

