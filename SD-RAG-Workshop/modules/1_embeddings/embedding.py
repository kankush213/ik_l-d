# -*- coding: utf-8 -*-
"""
================================================================================
MODULE 1: Embeddings & Similarity Search Demo (Gemini Version)
================================================================================
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from google import genai
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

load_dotenv()

print("Initializing Gemini client...")
client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))

# Retrieve embedding model and defensively strip 'models/' prefix if present
embedding_model = os.getenv('GEMINI_EMBEDDING_MODEL', 'gemini-embedding-001')
if embedding_model.startswith('models/'):
    embedding_model = embedding_model.replace('models/', '', 1)

embedding_dim = 768  

print(f"Using Gemini model: {embedding_model}")
print(f"Embedding dimension: {embedding_dim}")

print("\nLoading support tickets...")
with open('../../data/synthetic_tickets.json', 'r') as f:
    tickets = json.load(f)
print(f"Loaded {len(tickets)} support tickets")

print("\n" + "="*80)
print("SAMPLE TICKET:")
print("="*80)
sample = tickets[0]
print(f"ID: {sample['ticket_id']}")
print(f"Title: {sample['title']}")
print(f"Description: {sample['description'][:200]}...")
print(f"Category: {sample['category']}")
print(f"Priority: {sample['priority']}")

print("\n" + "="*80)
print("PART 1: Generating Embeddings")
print("="*80)

ticket_texts = [
    f"{ticket['title']}. {ticket['description']}" 
    for ticket in tickets
]

print("\nGenerating embeddings for all tickets...")

# Embed texts one by one to avoid list payload 404 errors
embeddings_list = []
for text in ticket_texts:
    res = client.models.embed_content(
        model=embedding_model,
        contents=text
    )
    embeddings_list.append(res.embeddings[0].values)

embeddings = np.array(embeddings_list)
print(f"✓ Generated embeddings with shape: {embeddings.shape}")
print(f"  ({len(tickets)} tickets × {embeddings.shape[1]} dimensions)")

print(f"\nFirst 10 values of embedding for ticket 1:")
print(embeddings[0][:10])

print("\n" + "="*80)
print("PART 2: Computing Similarity Scores")
print("="*80)

query = "Users can't login after changing password"
print(f"\nSearch Query: '{query}'")

query_response = client.models.embed_content(
    model=embedding_model,
    contents=query
)
query_embedding = np.array([query_response.embeddings[0].values])
print(f"Query embedding shape: {query_embedding.shape}")

similarities = cosine_similarity(query_embedding, embeddings)[0]
print(f"\nComputed similarity scores for {len(similarities)} tickets")
print(f"Similarity range: [{similarities.min():.4f}, {similarities.max():.4f}]")

print("\n" + "="*80)
print("PART 3: Finding Most Similar Tickets")
print("="*80)

top_k = 5
top_indices = np.argsort(similarities)[::-1][:top_k]

print(f"\nTop {top_k} most similar tickets to query: '{query}'")
print("-" * 80)

for rank, idx in enumerate(top_indices, 1):
    ticket = tickets[idx]
    score = similarities[idx]
    
    print(f"\n#{rank} - Similarity: {score:.4f}")
    print(f"Ticket ID: {ticket['ticket_id']}")
    print(f"Title: {ticket['title']}")
    print(f"Category: {ticket['category']} | Priority: {ticket['priority']}")
    print(f"Description: {ticket['description'][:150]}...")

print("\n" + "="*80)
print("PART 4: Visualizing Similarity Relationships")
print("="*80)

print("Creating similarity heatmap...")

selected_indices = list(top_indices[:5]) + list(np.random.choice(
    [i for i in range(len(tickets)) if i not in top_indices[:5]], 
    size=min(5, len(tickets) - 5), 
    replace=False
))

selected_embeddings = embeddings[selected_indices]
similarity_matrix = cosine_similarity(selected_embeddings)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

im = ax1.imshow(similarity_matrix, cmap='RdYlGn', vmin=0, vmax=1)
ax1.set_xticks(range(len(selected_indices)))
ax1.set_yticks(range(len(selected_indices)))

labels = [f"{tickets[i]['ticket_id']}\n({tickets[i]['category']})" 
          for i in selected_indices]
ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
ax1.set_yticklabels(labels, fontsize=8)

for i in range(len(selected_indices)):
    for j in range(len(selected_indices)):
        ax1.text(j, i, f'{similarity_matrix[i, j]:.2f}',
                 ha="center", va="center", color="black", fontsize=9)

ax1.set_title('Similarity Heatmap: What Embeddings Actually Measure\n' + 
             '(Top 5 matches + random others)', fontweight='bold', fontsize=11)
plt.colorbar(im, ax=ax1, label='Cosine Similarity')

query_similarities = [similarities[i] for i in selected_indices]
colors_bar = ['green' if i < 5 else 'gray' for i in range(len(selected_indices))]

ax2.barh(range(len(selected_indices)), query_similarities, color=colors_bar, alpha=0.7)
ax2.set_yticks(range(len(selected_indices)))
ax2.set_yticklabels([f"{tickets[i]['ticket_id']}" for i in selected_indices], fontsize=9)
ax2.set_xlabel('Similarity to Query', fontweight='bold')
ax2.set_title(f'Similarity Scores for Query:\n"{query}"\n(Green = Top 5 matches)', 
             fontweight='bold', fontsize=11)
ax2.set_xlim(0, 1)
ax2.grid(axis='x', alpha=0.3)

for i, score in enumerate(query_similarities):
    ax2.text(score + 0.02, i, f'{score:.3f}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('embeddings_similarity_analysis.png', dpi=150, bbox_inches='tight')
print("✓ Visualization saved as 'embeddings_similarity_analysis.png'")

print("\n" + "="*80)
print("PART 5: Try Different Queries")
print("="*80)

test_queries = [
    "Database is timing out",
    "Payment not working for foreign customers",
    "App crashes on iPhone",
    "Emails are not being sent"
]

print("\nTesting semantic search with different queries:")
for test_query in test_queries:
    query_resp = client.models.embed_content(
        model=embedding_model,
        contents=test_query
    )
    query_emb = np.array([query_resp.embeddings[0].values])
    
    sims = cosine_similarity(query_emb, embeddings)[0]
    top_idx = np.argmax(sims)
    
    print(f"\nQuery: '{test_query}'")
    print(f"  → Best match: {tickets[top_idx]['title']}")
    print(f"  → Similarity: {sims[top_idx]:.4f}")

print("\n" + "="*80)
print("DEMO COMPLETE!")
print("="*80)