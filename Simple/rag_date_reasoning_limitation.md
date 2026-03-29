# RAG Limitation: Identifying the Starting Date from a CV

## Problem
When asking a Retrieval-Augmented Generation (RAG) system:

> "What is the starting date of the first job role?"

The system sometimes returns an incorrect date, even though the CV clearly lists job roles in chronological order with dates. The earliest job role is visible, but the model does not always pick it.

## Why This Happens
RAG splits the document into chunks, converts them into vectors, retrieves top-k chunks by similarity, and then sends them to an LLM. Retrieval is based on **semantic similarity**, not logical reasoning:

- No chronological understanding
- No date comparison
- Only similarity matching

### Diagram of RAG Flow
```
CV PDF --> Chunking --> Vector Embeddings --> Top-k Retrieval --> LLM --> Answer
```

For example, a CV may have:
```
Jan 2024 – Present | Software Engineer
Jun 2022 – Dec 2023 | Associate Developer
Mar 2021 – May 2022 | Intern
```
The retriever may return the latest job chunk first.

## Fix with Better Prompt
Explicitly instruct the model to reason:
```text
Extract all job roles with dates from the CV.
Sort them chronologically and return the earliest starting date.
```
This converts the task from pure retrieval to **retrieval + reason