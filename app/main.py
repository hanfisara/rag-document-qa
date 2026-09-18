from document_loader import load_pdf
from chunking import create_chunks
from embedding import create_embeddings, model
from retrieval import find_similar_chunks
from generation import generate_answer


text = load_pdf("data/FAQ_s_for_Application_for_POHW_20230513.pdf")

chunks = create_chunks(text)

embeddings = create_embeddings(chunks)

print(f"Total chunks: {len(chunks)}")
print("Total embeddings:", len(embeddings))


query = "How is pension calculated?"

retrieved_chunks = find_similar_chunks(
    query,
    chunks,
    embeddings,
    model,
    top_k=3
)


# print("\n--- RETRIEVED CHUNKS ---")

# for i, chunk in enumerate(retrieved_chunks, 1):
#     print(f"\n--- RESULT {i} ---")
#     print(chunk)

# 6. Create context
context = "\n\n".join(retrieved_chunks)

# print("\n--- CONTEXT ---")
# print(context)    

# 7. Create prompt
prompt = f"""
Answer the question using ONLY the information provided in the context.

If the answer is not present in the context, say:
"I don't have enough information in the provided documents."

Do not add information from your own knowledge.

Context:
{context}

Question:
{query}

Answer:
"""

# print("\n--- PROMPT ---")
# print(prompt)

#8 LLM
answer = generate_answer(prompt)
print("\n--- FINAL ANSWER ---")
print(answer)

# for i, chunks in enumerate(chunks,1):
#     print(f"\n--- CHUNK {i} ---")
#     print(chunks)

# print("Total chunks:", len(chunks))
# print("Total embeddings:", len(embeddings))
# print("First embedding:", embeddings[0])