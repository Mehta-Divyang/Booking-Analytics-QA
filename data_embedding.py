import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np

df = pd.read_csv("cleaned_bookings.csv")

df["arrival_date"] = pd.to_datetime(df["arrival_date"])

df["year_month"] = df["arrival_date"].dt.to_period("M")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

df["text_data"] = df.apply(lambda row: 
    f"Booking in {row['hotel']} on {row['arrival_date']} by guest from {row['country']}. "
    f"Price: {row['adr']}€, Canceled: {bool(row['is_canceled'])}", axis=1)

embeddings = embedding_model.encode(df["text_data"].tolist(), convert_to_numpy=True)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

faiss.write_index(index, "faiss_index.bin")
np.save("embeddings.npy", embeddings)
df.to_csv("cleaned_bookings.csv", index=False)

print("Data embeddings generated and stored!")