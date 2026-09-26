# -*- coding: utf-8 -*-
"""
Module 2: Chunking & Vector Stores Demo (Gemini Version)
========================================================

This demo teaches:
1. Different chunking strategies for long documents
2. Building a Chroma vector store
3. Comparing retrieval quality across strategies
4. Metadata filtering for targeted search
"""

import json
import os
import time
from langchain_text_splitters import (  # Various splitting strategies
    RecursiveCharacterTextSplitter,  # Best general-purpose splitter
    CharacterTextSplitter,  # Simple split by character count
    MarkdownHeaderTextSplitter,  # Splits based on markdown headers
    HTMLHeaderTextSplitter  # Splits based on HTML tags
)
from langchain_experimental.text_splitter import SemanticChunker  # AI-powered semantic chunking
from langchain_chroma import Chroma  # Standalone Chroma package
from langchain_google_genai import GoogleGenerativeAIEmbeddings  # Gemini embedding function
from langchain_core.documents import Document  # Document abstraction
from dotenv import load_dotenv

# ============================================================================
# SETUP: Load environment and data
# ============================================================================

load_dotenv()

print("="*80)
print("MODULE 2: CHUNKING & VECTOR STORES (GEMINI VERSION)")
print("="*80)

raw_model = os.getenv('GEMINI_EMBEDDING_MODEL', 'text-embedding-004')
emb_model_name = raw_model.replace('models/', '', 1) if raw_model.startswith('models/') else raw_model

embeddings_model = GoogleGenerativeAIEmbeddings(
    model=f"models/{emb_model_name}",
    google_api_key=os.getenv('GEMINI_API_KEY')
)

with open('../../data/synthetic_tickets.json', 'r') as f:
    tickets = json.load(f)
print(f"\nLoaded {len(tickets)} support tickets")

# Helper function to safely add documents to Chroma with automatic 429 retry
def add_documents_safely(vector_store, docs, batch_size=10, delay_between_batches=3):
    """Ingests documents in small batches with automatic backoff for Gemini free-tier rate limits."""
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        for attempt in range(5):
            try:
                vector_store.add_documents(batch)
                break
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    print(f"  ⚠️ Quota limit reached. Pausing for 15s (Attempt {attempt+1}/5)...")
                    time.sleep(15)
                else:
                    raise e
        time.sleep(delay_between_batches)

# ============================================================================
# PART 1: Chunking Strategies
# ============================================================================
print("\n" + "="*80)
print("PART 1: Chunking Strategies")
print("="*80)

documents = []
for ticket in tickets:
    full_text = f"""
Ticket ID: {ticket['ticket_id']}
Title: {ticket['title']}
Category: {ticket['category']}
Priority: {ticket['priority']}
Description: {ticket['description']}
Resolution: {ticket['resolution']}
    """.strip()
    
    doc = Document(
        page_content=full_text,
        metadata={
            'ticket_id': ticket['ticket_id'],
            'category': ticket['category'],
            'priority': ticket['priority']
        }
    )
    documents.append(doc)

print(f"Created {len(documents)} documents")
print(f"\nSample document length: {len(documents[0].page_content)} characters")

# STRATEGY 1: Fixed-Size Chunking
print("\n--- Strategy 1: Fixed-Size Chunking ---")
fixed_splitter = CharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20,
    separator="\n"
)
fixed_chunks = fixed_splitter.split_documents(documents)
print(f"✓ Created {len(fixed_chunks)} chunks")

# STRATEGY 2: Recursive Character Splitting
print("\n--- Strategy 2: Recursive Character Splitting ---")
recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " "]
)
recursive_chunks = recursive_splitter.split_documents(documents)
print(f"✓ Created {len(recursive_chunks)} chunks")

# STRATEGY 3: Semantic Chunking
print("\n--- Strategy 3: Semantic Chunking ---")
demo_paragraph = """
The Mars rover collected soil samples from the Jezero crater last week. Scientists believe these rocks may contain signs of ancient microbial life. NASA plans to retrieve these samples in a future mission. The discovery could reshape our understanding of life in the solar system.

Grandma's apple pie recipe starts with peeling six large Granny Smith apples. Mix flour, sugar, and cinnamon for the filling. Roll the dough thin and crimp the edges carefully. Bake at 375 degrees for 45 minutes until golden brown.

The defendant was charged with breach of contract under Section 12. The plaintiff seeks damages of fifty thousand dollars plus legal fees. Both parties agreed to mediation before proceeding to trial. The judge scheduled the preliminary hearing for next month.
"""

semantic_splitter = SemanticChunker(
    embeddings=embeddings_model,
    breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=1.0
)
demo_doc = Document(page_content=demo_paragraph.strip())
semantic_chunks = semantic_splitter.split_documents([demo_doc])
print(f"✓ Created {len(semantic_chunks)} chunks")

# STRATEGY 4: Markdown Header Splitting
print("\n--- Strategy 4: Markdown Header Splitting ---")
markdown_doc = """
# Database Troubleshooting Guide
## Connection Issues
### Timeout Errors
If you encounter timeout errors, check the connection string and ensure the database server is reachable.
"""
markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")],
    strip_headers=False
)
md_chunks = markdown_splitter.split_text(markdown_doc)
print(f"✓ Created {len(md_chunks)} chunks from markdown")

# STRATEGY 5: HTML Header Splitting
print("\n--- Strategy 5: HTML Header Splitting ---")
html_doc = "<h1>Email Guide</h1><h2>SMTP</h2><p>Server details</p>"
html_splitter = HTMLHeaderTextSplitter(headers_to_split_on=[("h1", "Header 1"), ("h2", "Header 2")])
html_chunks = html_splitter.split_text(html_doc)
print(f"✓ Created {len(html_chunks)} chunks from HTML")

# STRATEGY 6: Whole Documents
print("\n--- Strategy 6: Whole Documents (No Chunking) ---")
print(f"✓ Using {len(documents)} whole documents")

# ============================================================================
# PART 2: Chroma Vector Store with Gemini Embeddings
# ============================================================================
print("\n" + "="*80)
print("PART 2: Chroma Vector Store")
print("="*80)

query = "Authentication problems after password reset"

existing_store = Chroma(
    collection_name="support_tickets",
    persist_directory="./chroma_db"
)
existing_store.delete_collection()

chroma_store = Chroma(
    collection_name="support_tickets",
    embedding_function=embeddings_model,
    persist_directory="./chroma_db"
)
add_documents_safely(chroma_store, documents)
print("✓ Chroma store created and persisted with Gemini Embeddings")

chroma_results = chroma_store.similarity_search(query, k=3)
print(f"\nTop {len(chroma_results)} results:")
for i, doc in enumerate(chroma_results, 1):
    print(f"#{i} - Ticket: {doc.metadata['ticket_id']} ({doc.metadata['category']})")

# ============================================================================
# PART 3: Metadata Filtering
# ============================================================================
print("\n" + "="*80)
print("PART 3: Metadata Filtering")
print("="*80)

filtered_results = chroma_store.similarity_search(
    query,
    k=3,
    filter={"category": "Authentication"}
)
print(f"Filtered results count: {len(filtered_results)}")

# ============================================================================
# PART 4: Comparing Chunking Strategies (With Rate Limit Protection)
# ============================================================================
print("\n" + "="*80)
print("PART 4: Evaluating Chunking Strategies")
print("="*80)

print("\nBuilding vector stores with different chunking strategies...")

# Whole documents store reuses Part 2
store_whole = chroma_store

# Create Fixed Chunks Store
print("Ingesting Fixed Chunks...")
store_fixed = Chroma(collection_name="fixed_chunks", embedding_function=embeddings_model)
add_documents_safely(store_fixed, fixed_chunks)

# Create Recursive Chunks Store
print("Ingesting Recursive Chunks...")
store_recursive = Chroma(collection_name="recursive_chunks", embedding_function=embeddings_model)
add_documents_safely(store_recursive, recursive_chunks)

test_query = "Database connection failures"
print(f"\nTest query: '{test_query}'")

stores = [
    ("Whole Documents", store_whole),
    ("Fixed Chunks", store_fixed),
    ("Recursive Chunks", store_recursive)
]

for name, store in stores:
    results = store.similarity_search(test_query, k=1)
    print(f"\n{name}:")
    if results:
        print(f"  Top result: {results[0].page_content[:100]}...")
        print(f"  Length: {len(results[0].page_content)} chars")

print("\n" + "="*80)
print("DEMO COMPLETE!")
print("="*80)