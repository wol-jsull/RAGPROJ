"""Patent RAG: retrieval-augmented question answering over a small US patent corpus.

The notebooks in ``notebooks/`` document how the corpus was built (PDF extraction,
chunking, embedding). This package turns that pipeline into reusable, testable
modules that the Streamlit app (``app.py``) and the scripts in ``scripts/`` share.
"""

__version__ = "1.1.0"
