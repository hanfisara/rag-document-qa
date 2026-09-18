from sentence_transformers import util

def find_similar_chunks(query, chunks, embeddings, model, top_k=3):
    query_embedding = model.encode(query)

    scores = util.cos_sim(query_embedding, embeddings)[0]

    # best_index = scores.argmax() this is for top best 1 chunk

    top_indices =  scores.argsort(descending=True)[:top_k]

    retrieved_chunks = [chunks[i] for i in top_indices]

    return retrieved_chunks