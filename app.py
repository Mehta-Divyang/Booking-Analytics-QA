from flask import Flask, request, jsonify
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import io
import base64
import time
import json

faiss_index = None

def load_faiss_index():
    global faiss_index
    if faiss_index is None:
        faiss_index = faiss.read_index("faiss_index.bin")
        print("FAISS index loaded into memory...")

load_faiss_index()  

df = pd.read_csv("cleaned_bookings.csv")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def answer_query(query):
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    _, idx = faiss_index.search(query_embedding, k=1)
    result = df.iloc[idx[0][0]]  
    return {
        "hotel": result["hotel"],
        "arrival_date": str(result["arrival_date"]),
        "country": result["country"],
        "price": result["adr"],
        "canceled": bool(result["is_canceled"])
    }

query_history_file = "query_history.json"

def save_query_history(query, response):
    try:
        with open(query_history_file, "r") as f:
            history = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        history = []

    history.append({"question": query, "response": response})

    with open(query_history_file, "w") as f:
        json.dump(history, f, indent=4)

def precompute_analytics():
    analytics_data = {}

    df["year_month"] = pd.to_datetime(df["arrival_date"]).dt.to_period("M")
    revenue_trends = df.groupby("year_month")["adr"].sum().to_dict()
    analytics_data["revenue_trends"] = revenue_trends

    analytics_data["cancellation_rate"] = round(df["is_canceled"].mean() * 100, 2)

    analytics_data["top_countries"] = df["country"].value_counts().head(5).to_dict()

    analytics_data["lead_time_avg"] = round(df["lead_time"].mean(), 2)

    with open("analytics_cache.json", "w") as f:
        json.dump(analytics_data, f, indent=4)

    print("Precomputed analytics stored successfully!")

precompute_analytics()

app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to the LLM-Powered Booking Analytics System!"

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    query = data.get("question", "")

    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    _, idx = faiss_index.search(query_embedding, k=1)
    result = df.iloc[idx[0][0]]

    response = {
        "hotel": result["hotel"],
        "arrival_date": str(result["arrival_date"]),
        "country": result["country"],
        "price": result["adr"],
        "canceled": bool(result["is_canceled"])
    }

    save_query_history(query, response)
    return jsonify(response)

@app.route("/analytics", methods=["POST"])
def analytics():
    with open("analytics_cache.json", "r") as f:
        analytics_data = json.load(f)

    data = request.json
    analysis_type = data.get("type", "")

    return jsonify({analysis_type: analytics_data.get(analysis_type, "Invalid type")})

if __name__ == "__main__":
    app.run(debug=True)
