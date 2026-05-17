from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from typing import List, Any


def topic_modelling(docs: List[Any], strategy: str = "lda", n_topics: int = 5) -> List[Any]:
    """Apply topic modelling to a collection of documents.

    Assigns a topic ID and top keywords to each document's metadata.

    Args:
        docs: List of documents (strings or LangChain Document objects).
        strategy: Topic modelling strategy ("lda" for Latent Dirichlet Allocation,
                                            "lsa" for Latent Semantic Analysis).
        n_topics: Number of topics to extract.

    Returns:
        List of documents with topic information in metadata.

    Raises:
        ValueError: If an unknown topic modelling strategy is specified.
    """
    # Extract text from documents (handles strings and LangChain Documents)
    texts = [doc.page_content if hasattr(doc, "page_content") else str(doc) for doc in docs]

    if not texts:
        return docs

    if strategy == "lda":
        # LDA typically works best with raw counts
        vectorizer = CountVectorizer(stop_words='english')
        data_vectorized = vectorizer.fit_transform(texts)
        model = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        doc_topic_matrix = model.fit_transform(data_vectorized)
    elif strategy == "lsa":
        # LSA (Truncated SVD) typically works best with TF-IDF
        vectorizer = TfidfVectorizer(stop_words='english')
        data_vectorized = vectorizer.fit_transform(texts)
        model = TruncatedSVD(n_components=n_topics, random_state=42)
        doc_topic_matrix = model.fit_transform(data_vectorized)
    else:
        raise ValueError(f"Unknown topic modelling strategy: {strategy}")

    # Identify the dominant topic for each document
    top_topics = doc_topic_matrix.argmax(axis=1)

    # Extract top keywords for each topic to enrich metadata
    feature_names = vectorizer.get_feature_names_out()
    topic_keywords = []
    for _, topic in enumerate(model.components_):
        # Get top 5 words for this topic
        top_word_indices = topic.argsort()[:-6:-1]
        top_words = [feature_names[i] for i in top_word_indices]
        topic_keywords.append(", ".join(top_words))

    # Update document metadata with topic information
    for i, doc in enumerate(docs):
        topic_id = int(top_topics[i])
        keywords = topic_keywords[topic_id]

        if hasattr(doc, "metadata"):
            doc.metadata["topic_id"] = topic_id
            doc.metadata["topic_keywords"] = keywords
        elif isinstance(doc, dict):
            doc["topic_id"] = topic_id
            doc["topic_keywords"] = keywords

    return docs
