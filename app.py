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
    start_time = time.time()  

    try:
        data = request.json
        analysis_type = data.get("type", "")

        if analysis_type == "revenue_trends":
            df["year_month"] = pd.to_datetime(df["arrival_date"]).dt.to_period("M")
            monthly_revenue = df.groupby("year_month")["adr"].sum()

            plt.figure(figsize=(8, 4))
            monthly_revenue.plot(kind="line", marker="o", color="b")
            plt.title("Revenue Trends Over Time")
            plt.xlabel("Month")
            plt.ylabel("Total Revenue")
            plt.grid(True)

            img = io.BytesIO()
            plt.savefig(img, format="png")
            img.seek(0)
            plot_url = base64.b64encode(img.getvalue()).decode()
            plt.close()

            response = {"analysis": "revenue_trends", "plot": plot_url}

        elif analysis_type == "cancellation_rate":
            cancellation_rate = df["is_canceled"].mean() * 100
            response = {"analysis": "cancellation_rate", "rate": f"{cancellation_rate:.2f}%"}

        elif analysis_type == "top_countries":
            country_counts = df["country"].value_counts().head(5).to_dict()
            response = {"analysis": "top_countries", "countries": country_counts}

        elif analysis_type == "lead_time_distribution":
            lead_time_avg = df["lead_time"].mean()
            response = {"analysis": "lead_time_distribution", "average_lead_time": round(lead_time_avg, 2)}

        else:
            response = {"error": "Invalid analysis type"}

    except Exception as e:
        response = {"error": str(e)}

    end_time = time.time()  
    response["response_time"] = f"{(end_time - start_time):.3f} seconds"  
    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)
