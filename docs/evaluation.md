# Evaluation

Metrics staged in the public export:
- Recall@K for K in {1, 3, 5, 10}
- image-to-report retrieval
- report-to-image retrieval
- intramodal retrieval summaries
- majority-vote summaries
- global benchmarking dashboards

Public-safe evaluation assets included in this repository:

- dashboard figures derived from aggregated metrics;
- the notebook template used to build the final comparative dashboards;
- the reporting script that reads precomputed metrics without regenerating embeddings.

The public version documents the reporting layer, not the full private execution environment used in the thesis.
