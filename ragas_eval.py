from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision
)
from datasets import Dataset
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
import os

def evaluate_rag_quality(questions, answers, contexts, ground_truths=None):
    """
    Evaluates RAG pipeline quality using RAGAS metrics.

    Metrics:
    - Faithfulness: Is answer grounded in context? (0-1)
    - Answer Relevancy: Does answer address question? (0-1)
    - Context Precision: Are retrieved chunks relevant? (0-1)
    - Context Recall: Did we find all relevant info? (0-1)
    """
    print("📊 Running RAGAS evaluation...")

    # Prepare dataset
    if ground_truths is None:
        ground_truths = answers  # use answers as proxy if no ground truth

    data = {
        "question": questions,
        "answer": answers,
        "contexts": [[c] if isinstance(c, str) else c for c in contexts],
        "ground_truth": ground_truths
    }

    dataset = Dataset.from_dict(data)

    try:
        # Run evaluation
        results = evaluate(
            dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall
            ]
        )

        scores = {
            "faithfulness": round(float(results["faithfulness"]), 3),
            "answer_relevancy": round(float(results["answer_relevancy"]), 3),
            "context_precision": round(float(results["context_precision"]), 3),
            "context_recall": round(float(results["context_recall"]), 3),
        }

        overall = sum(scores.values()) / len(scores)
        scores["overall"] = round(overall, 3)

        print(f"✅ RAGAS evaluation complete: {scores}")
        return scores

    except Exception as e:
        print(f"⚠️ RAGAS evaluation failed: {e}")
        return None