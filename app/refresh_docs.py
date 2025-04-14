import torch
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
import json
import numpy as np
from app.embeddings import Embeddings

class RefreshDocs:
    def __init__(self, model_name="microsoft/Phi-3.5-mini-instruct"):
        self.model_name = model_name

    def refresh(self, file, action="RAG", max_cache_tokens=int(128000 / 10),):
        if action == "CACHE":
            return self.recalculate_cached_docs(file, max_cache_tokens)
        else:
            self.recalculate_rag_embeddings(file)

    def load_model(self):
        # Load the tokenizer and model with 4-bit quantization to reduce memory usage.
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quant_config,
            device_map='auto'
        )
        self.device = self.model.model.embed_tokens.weight.device

    def recalculate_cached_docs(self, fname, max_cache_tokens):
        # c
        data_embed = Embeddings(fname, import_emb=True)
        sorted_count_idx = np.argsort(data_embed.access_counts)[::-1]
        self.sorted_docs = [data_embed.documents[i] for i in sorted_count_idx]
        self.sorted_embed = [data_embed.embeddings[i] for i in sorted_count_idx]
        self.load_model()

        total = 0
        cache_indices = []
        for idx in sorted_count_idx:
            doc_input_ids = self.tokenizer.encode(data_embed.documents[idx], return_tensors="pt").to(self.device)
            if (total + doc_input_ids.shape[1]) < max_cache_tokens:
                cache_indices.append(idx)
                total += doc_input_ids.shape[1]
            else:
                break
        # print results
        #print('Tokens for Cache Docs', total)
        #print('Number of Docs for Cache', len(cache_indices))

        # open all docs file and add cache indices
        with open(fname, "r", encoding="utf-8") as file:
            ogs_docs = json.load(file)
        with open(fname, "w", encoding="utf-8") as f:
            json.dump({'embeddings': ogs_docs['embeddings'],
                       'documents': ogs_docs['documents'],
                       'urls': ogs_docs['urls'],
                       'cache_idxs': [int(item) for item in cache_indices],
                       'access_count': ogs_docs['access_count']
                       }, f, indent=4)
        return {'cache_tok': total, 'cache_doc_idx': len(cache_indices)}

    def recalculate_rag_embeddings(self, fname='documents_rag.json'):
        # recalculate search embeddings, on some recurring frequency run:
        self.data_embed = Embeddings(fname)  # default arg import_emb=False so that this embeds documents

        # generate fake counts for testing, this needs some replacement for prod
        counts_log = np.clip(np.round(np.random.exponential(scale=5, size=len(self.data_embed.data))).astype(int), 0, 30)

        with open('app/all_OGS_embedded_docs.json', "w", encoding="utf-8") as f:
            json.dump({'embeddings': self.data_embed.embeddings.tolist(),
                       'documents': self.data_embed.documents,
                       'urls': self.data_embed.urls,
                       'access_count': counts_log.tolist()  # replace with some matching against the old list
                       }, f, indent=4)