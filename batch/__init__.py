"""ChemReX batch — large-scale PDF extraction with resident models.

Independent of the core pipeline/ package: the core keeps handling single
text segments and single images; this package adds

- a SQLite task queue (add / list / retry / status),
- a resident MinerU parsing service (model loaded once, many PDFs),
- a resident OCR/OCSR image worker (models loaded once, many documents),
- a scheduler that runs documents concurrently with per-chunk LLM workers.

Entry point: ``python run_batch.py <command>`` (see batch/cli.py).
"""
