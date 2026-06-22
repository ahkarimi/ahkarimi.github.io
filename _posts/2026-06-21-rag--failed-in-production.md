---
layout: post
title: RAG Failed Me in Production — And Then I Understood Why Claude Code Abandoned It Too
date: 2026-06-21 12:00:00
description: RAG failed in production due to chunking, semantic gaps, and stale indexes. Even Claude Code abandoned it for agentic search.
tags: rag production
categories: AI
thumbnail: assets/img/blog/2025/rag-vs-agentic-search.png
---

<div class="row mt-3">
    <div class="col-12 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/blog/2026/rag-vs-agentic-search.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    RAG vs Agentic Search
</div>

# RAG Failed Me in Production — And Then I Understood Why Claude Code Abandoned It Too

I was confident. I had embeddings, a vector database, a clean retrieval pipeline, and a working demo. The system felt smart. Users asked questions in natural language, and it returned relevant chunks from a large document base.

Then we went to production.

---

## What Actually Happened

The first cracks showed up quietly. Users would ask something completely reasonable — using a slightly different word, or an industry term we hadn't seen in our test set — and the retrieval would come back with something adjacent but wrong. Close enough to look right, but not actually what they needed.

Then came the documents with mixed formats: tables, code snippets, dense paragraphs. Our chunking strategy — which worked beautifully on clean prose — started cutting things in the worst possible places. A table split in half. A code block separated from its explanation. The retrieval system had no idea something was broken.

And then the knowledge base started changing. New versions of documents came in. Old ones were updated. But the index? It remembered the past. Users were getting answers based on information that was months out of date.

Three problems, all painful:

**Chunking is a decision you make once and pay for forever.** A simple fixed-size strategy that works on your test data will quietly fall apart when real-world documents arrive in formats you didn't anticipate.

**Semantic gap is real and hard to close.** Embedding models are good, but they're not mind readers. When a user's vocabulary doesn't match the vocabulary in your documents, similarity search returns things that feel related but aren't.

**RAG is static by nature.** It's excellent for stable knowledge. But the moment your data is alive — changing, updating, expanding — your index starts drifting from reality.

---

## Then I Saw What Claude Code Did

Around the time I was debugging all of this, I came across a Hacker News comment from Boris Cherny, the creator of Claude Code. He mentioned that early versions of Claude Code actually used RAG with a local vector database. They tried it. They tested it. And then they dropped it.

His explanation was refreshingly honest: agentic search — using tools like `grep`, `glob`, and `find` to explore the codebase on demand — outperformed RAG. By a lot. And it surprised them too.

The reasons made immediate sense to me:

- `grep` finds exactly what's there. Embeddings introduce fuzzy matches that sometimes help and sometimes mislead.
- A pre-built index drifts. Code changes constantly during active development, so any index built yesterday is already slightly wrong today.
- No index means no setup, no maintenance, and no data leaving the machine for embedding computation.

Reading that, I felt strangely relieved. Not because my failures were excused, but because even a team at Anthropic — with far more resources than I had — hit the same wall and made the same pivot.

---

## But It's Worth Being Precise Here

There's a framing thing worth noting. If you go back to the original definition of RAG — a mechanism that retrieves external information and uses it to generate a response — then agentic search *is* a form of RAG. Claude Code retrieves context; it just does it through filesystem exploration instead of vector similarity.

What was actually abandoned was what most engineers mean when they say "RAG" in practice: embeddings, vector database, pre-built index, similarity search. The retrieve-then-generate philosophy stayed. The infrastructure around it was replaced.

And agentic search has its own limits. Ask Claude Code a conceptual question — "where does authentication happen in this codebase?" — and it's limited to literal string matching and file-by-file exploration. It can struggle with questions that require semantic understanding rather than exact lookup. Cursor, for instance, still uses vector embeddings precisely because of this gap.

---

## What I'm Still Thinking About

I don't think RAG is dead. I think most of us — myself included — underestimated how much the surrounding system matters. Chunking strategy, index freshness, evaluation loops, feedback mechanisms. The model is often the least of your problems.

What I'm more interested in now is when each approach actually fits. Static knowledge with stable documents and high query volume? Standard RAG still makes sense. Active codebases, changing data, precision-critical queries? Agentic search starts to look much more attractive. Somewhere in between? Probably a hybrid, with a lot of careful thought about what you're optimizing for.

I'm still figuring it out. But I've stopped assuming that a working demo means a working product.

---

*If you've hit similar walls with RAG in production, I'd genuinely like to hear what you found. My contact is always open.*
