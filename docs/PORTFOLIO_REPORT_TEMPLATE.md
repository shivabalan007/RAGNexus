# RAGNexus Retrieval Engineering — Portfolio Report

> Replace all bracketed placeholders with measured results. Do not publish guessed or fabricated metrics.

## 1. Executive summary

- System: RAGNexus — workspace-scoped retrieval-augmented assistant.
- Corpus: [number of workspaces, documents, current versions and chunks].
- Evaluation set: [number of reviewed questions; dataset revision/hash].
- Best configuration: [name, with measured evidence].
- Main trade-off: [quality gain vs latency/cost].

## 2. System architecture

Describe FastAPI, PostgreSQL/pgvector, the ingestion worker, Streamlit, standard RAG, LangGraph Agentic RAG, Tavily, Redis and OpenTelemetry. Include the architecture diagram from `ARCHITECTURE_ADVANCED_ENGINEERING.md` or create a measured final diagram.

## 3. Evaluation protocol

- Dataset creation and human-review procedure: [details].
- Relevant-unit labels: [chunk-level or document-level].
- Workspace/document access rules: [details].
- Candidate K / final K: [values].
- Embedding/reranker/model versions: [exact IDs and revisions].
- Generation parameters: [temperature, token cap, provider].
- Runs and hardware: [number of repeats and system specs].

## 4. Retrieval comparison

Paste the generated `evals/reports/retrieval/comparison.md` table. Explain changes in Recall@K, Precision@K, MRR and nDCG configuration by configuration.

## 5. Answer and citation quality

Report normalized exact match, token F1, lexical expected-fact coverage, answerability/abstention accuracy, citation precision, citation coverage and LLM-as-judge faithfulness. Manually review false positives and unsupported answers; do not rely on an automated judge alone.

## 6. Latency and cost

Report retrieval and end-to-end p50/p95 latency, embedding and reranking cold-start times, input/output tokens, estimated provider costs and Tavily usage. Distinguish warm and cold runs.

## 7. Ablations and error analysis

Explain which techniques improved results (structure-aware chunking, parent expansion, multi-query, compression, reranker changes) and which did not. Include failure examples: contextual follow-ups, conflicting versions, insufficient evidence, wrong citations, and deleted-document cases.

## 8. Security and reliability

Describe tested workspace isolation, selected-document scope, role restrictions, document lifecycle, ingestion worker lease recovery, approval controls, and data retention. State exactly which checks were manual and which are automated.

## 9. Limitations

List corpus coverage, language/domain limitations, embedding dimension constraints, external provider dependency, evaluation-set bias, and remaining unverified integrations.

## 10. Conclusion

Summarize the chosen production configuration and the evidence that justifies it. Include a link or checksum for the dataset and generated reports where appropriate.
