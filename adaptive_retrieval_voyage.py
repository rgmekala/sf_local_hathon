#!/usr/bin/env python3
import os
import numpy as np
from datetime import datetime, UTC
from pymongo import MongoClient
import voyageai

# --------------------------------------------------
# Config
# --------------------------------------------------

MONGO_URI = os.getenv("MONGO_URI")
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
EMBED_MODEL = "voyage-code-2"

client = MongoClient(MONGO_URI)
db = client["adaptive_mongo"]
logs = db["mongo_error_logs"]
knowledge = db["mongo_error_knowledge"]
metrics = db["retrieval_metrics"]

vo = voyageai.Client(api_key=VOYAGE_API_KEY)

# --------------------------------------------------
# Embeddings (Voyage AI)
# --------------------------------------------------
def embed(text: str):
    if not text.strip():
        return []
    res = vo.embed(
        texts=[text],
        model=EMBED_MODEL
    )
    return res.embeddings[0]

# --------------------------------------------------
# Ensure Embeddings Exist
# --------------------------------------------------
def ensure_embeddings(col, text_fields):
    missing = col.find({
        "$or": [
            {"embedding": {"$exists": False}},
            {"embedding": {"$size": 0}}
        ]
    })

    count = 0
    for doc in missing:
        text = " ".join(doc.get(f, "") for f in text_fields).strip()
        if not text:
            continue

        emb = embed(text)
        col.update_one(
            {"_id": doc["_id"]},
            {"$set": {
                "embedding": emb,
                "embedding_model": EMBED_MODEL,
                "embedding_ts": datetime.now(UTC)
            }}
        )
        count += 1

    if count:
        print(f"[INFO] Added embeddings to {count} docs in {col.name}")

# --------------------------------------------------
# MongoDB Atlas Vector Search
# --------------------------------------------------
def atlas_vector_search(col, query, index_name="vector_index", top_k=5):
    qv = embed(query)
    #print(type(qv), len(qv))

    pipeline = [
        {
            "$vectorSearch": {
                "index": index_name,
                "queryVector": qv,
                "path": "embedding",
                "numCandidates": 100,
                "limit": top_k
            }
        },
        {
            "$project": {
                "_id": 1,
                "raw_log": 1,
                "content": 1,
                "component": 1,
                "error_code": 1,
                "normalized_message": 1,
                "score": {"$meta": "vectorSearchScore"}
            }
        }
    ]

    return list(col.aggregate(pipeline))

# --------------------------------------------------
# Hybrid Search (Vector + Keyword)
# --------------------------------------------------
def hybrid_search(col, query, top_k=5):
    vec = atlas_vector_search(col, query, top_k=top_k)
    kw = list(col.find(
        {"$text": {"$search": query}},
        {"score": {"$meta": "textScore"}}
    ).limit(top_k))

    combined = {str(d["_id"]): d for d in vec + kw}
    return list(combined.values())

# --------------------------------------------------
# Deterministic Answer Generator
# --------------------------------------------------
def generate_answer(query, docs):
    if not docs:
        return None

    d = docs[0]
    return f"""
MongoDB Issue Detected
---------------------
Component: {d.get("component", "UNKNOWN")}
Error Code: {d.get("error_code", "N/A")}

Likely Cause:
{d.get("normalized_message") or d.get("raw_log") or d.get("content")}

Suggested Fix:
Review connection pool, retry logic, and MongoDB Atlas configuration.
"""

# --------------------------------------------------
# Judge / Scorer
# --------------------------------------------------
def judge(answer):
    score = 0.0
    if "Component:" in answer:
        score += 0.4
    if "Likely Cause" in answer:
        score += 0.3
    if "Suggested Fix" in answer:
        score += 0.3
    return score

# --------------------------------------------------
# Query Rewrite
# --------------------------------------------------
def rewrite_query(q):
    return q + " mongodb production error root cause"

# --------------------------------------------------
# Adaptive Agent Loop
# --------------------------------------------------
def adaptive_retrieval(query):
    ensure_embeddings(logs, ["raw_log"])
    ensure_embeddings(knowledge, ["content"])

    strategies = [
        ("logs_vector", 5),
        ("logs_hybrid", 5),
        ("knowledge_vector", 5),
        ("rewrite_knowledge", 10)
    ]

    strategy_scores = {}
    strategy_answers = {}

    for agent_pass in range(2):
        print(f"\n=== Agent Pass {agent_pass+1} ===")

        for strat, k in strategies:
            if strat == "logs_vector":
                docs = atlas_vector_search(logs, query, index_name="vector_index", top_k=k)
            elif strat == "logs_hybrid":
                docs = hybrid_search(logs, query, top_k=k)
            elif strat == "knowledge_vector":
                docs = atlas_vector_search(knowledge, query, index_name="vector_index",top_k=k)
            elif strat == "rewrite_knowledge":
                rq = rewrite_query(query)
                docs = atlas_vector_search(knowledge, rq, top_k=k)

            answer = generate_answer(query, docs)
            score = judge(answer) if answer else 0.0

            metrics.insert_one({
                "query": query,
                "strategy": strat,
                "score": score,
                "pass": agent_pass,
                "ts": datetime.now(UTC)
            })

            strategy_scores[strat] = max(strategy_scores.get(strat, 0), score)
            strategy_answers[strat] = answer

            print(f"{strat}: score={score:.2f}")

        # Sort strategies for next pass by score
        strategies.sort(key=lambda s: strategy_scores.get(s[0], 0), reverse=True)

    # Pick the best strategy at the end
    if strategy_scores:
        winning_strategy = max(strategy_scores, key=strategy_scores.get)
        return {
            "answer": strategy_answers.get(winning_strategy),
            "winning_strategy": winning_strategy,
            "confidence": strategy_scores[winning_strategy]
        }

    return {"answer": "No confident answer found."}

def pretty_print_result(result):
    if not result or "answer" not in result:
        print("\n No confident answer found.")
        return

    print("\n" + "=" * 60)
    print(" ADAPTIVE RETRIEVAL RESULT")
    print("=" * 60)

    print(f"\n Winning Strategy : {result.get('winning_strategy')}")
    print(f" Confidence       : {result.get('confidence'):.2f}")

    print("\n" + "-" * 60)
    print(result["answer"].strip())
    print("=" * 60)


# --------------------------------------------------
# Main
# --------------------------------------------------
if __name__ == "__main__":
    import sys
    query = " ".join(sys.argv[1:])
    result = adaptive_retrieval(query)
    pretty_print_result(result)


