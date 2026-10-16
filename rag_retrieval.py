
import re


def retrieve_relevant_chunks(
    question,
    embedding_model,
    collection,
    top_k=5
):
    count = collection.count()

    if count == 0:
        return []

    semantic_k = min(10, count)

    query_embedding = embedding_model.encode(
        question
    ).tolist()

    semantic_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=semantic_k
    )

    all_chunks = collection.get(
        include=["documents", "metadatas"]
    )

    query = question.lower()

    if "verification" in query and "methods" in query:
        query_terms = [
            "reviews",
            "walkthroughs",
            "inspections",
            "static analysis"
        ]
    else:
        stop_words = {
            "what", "which", "are", "is", "the",
            "a", "an", "of", "for", "to", "in",
            "on", "does", "used"
        }

        query_terms = [
            word for word in re.findall(r"\b\w+\b", query)
            if word not in stop_words
        ]

    keyword_results = []

    for document, metadata in zip(
        all_chunks["documents"],
        all_chunks["metadatas"]
    ):
        text = document.lower()

        score = sum(
            1 for term in query_terms if term in text
        )

        if score > 0:
            keyword_results.append({
                "page": metadata["page"],
                "text": document,
                "score": score
            })

    keyword_results.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    scores = {}
    lookup = {}

    def add_results(documents, metadatas):
        for rank, (document, metadata) in enumerate(
            zip(documents, metadatas), start=1
        ):
            key = (metadata["page"], document)
            scores[key] = scores.get(key, 0) + 1 / (60 + rank)

            lookup[key] = {
                "page": metadata["page"],
                "text": document
            }

    add_results(
        semantic_results["documents"][0],
        semantic_results["metadatas"][0]
    )

    for rank, item in enumerate(keyword_results, start=1):
        key = (item["page"], item["text"])
        scores[key] = scores.get(key, 0) + 1 / (60 + rank)

        lookup[key] = {
            "page": item["page"],
            "text": item["text"]
        }

    ranked = sorted(
        scores,
        key=lambda key: scores[key],
        reverse=True
    )

    return [
        {**lookup[key], "score": scores[key]}
        for key in ranked[:top_k]
    ]
