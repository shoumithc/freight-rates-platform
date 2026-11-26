# Runs at image build time so the container starts without downloading anything.
import json
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer

cfg = json.load(open("rag_index/rag_config.json"))
SentenceTransformer(cfg["embed_model"], device="cpu")
print("Embedding model cached:", cfg["embed_model"])
hf_hub_download(cfg["llm_repo"], cfg["llm_file"])
print("LLM cached:", cfg["llm_repo"], cfg["llm_file"])
