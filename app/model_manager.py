import torch
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM, pipeline
from sklearn.metrics.pairwise import cosine_similarity
import json
import numpy as np
from app.cache_generator import CacheGenerator
from app.embeddings import Embeddings
from app.retrieval import Retrieval
from app.rag_generator import RAGGenerator
import time 

class ModelManager:
    def __init__(self, model_name="microsoft/Phi-3.5-mini-instruct"):
        # load model
        self.model_name = model_name
        self.load_model()

        # get cache files and setup CAG attributes
        cache_file = self.load_file('app/all_OGS_embedded_docs.json')
        self.cache_docs = [cache_file['documents'][i] for i in cache_file['cache_idxs']]
        self.cache_embed = np.array([cache_file['embeddings'][i] for i in cache_file['cache_idxs']])
        self.cache_urls = [cache_file['urls'][i] for i in cache_file['cache_idxs']]
        del cache_file
        self.cag = CacheGenerator(self.cache_docs, self.model, self.tokenizer, self.device)

        # setup RAG attributes
        self.data_embed = Embeddings('app/all_OGS_embedded_docs.json', import_emb=True)  # load saved embeddings
        self.setup_pipeline()
        self.retriever = Retrieval(self.data_embed)
        self.rag = RAGGenerator(self.rag_pipeline)


    def load_file(self, fname):
        try:
            with open(fname, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Error: The file '{fname}' was not found.")
        except json.JSONDecodeError:
            print(f"Error: The file '{fname}' is not a valid JSON file.")
        except Exception as e:
            print(f"An unexpected error occurred while loading '{fname}': {e}")
        return None

    def load_model(self):
        """ load tokenizer and model using 4-bit quantization to speed up inference and reduce memor
        """
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

    def setup_pipeline(self):
        # Create a text-generation pipeline for RAG.
        self.rag_pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map='auto'
        )

    def query_handler(self, query, threshold=0.57, max_tokens=150, gen_method='Greedy', temperature=0.7, top_k=50):
        start_time = time.time()
        # calculate embedding for query and reshape for cosine sim
        query_embedding = self.data_embed.get_embedding(query)
        query_embedding = query_embedding.reshape(1, -1)

        similarity = cosine_similarity(query_embedding, self.cache_embed)

        # Find indices where the similarity exceeds the threshold
        above_threshold_indices = np.where(similarity[0] > threshold)[0]
        end_time = time.time()
        if np.max(similarity) > threshold:
            method = 'CAG'
            output = self.cag.query_responder(query, self.tokenizer, self.model, self.device, max_tokens, gen_method, temperature, top_k)
            output_urls = [self.cache_urls[idx] for idx in above_threshold_indices]
            response = {'method': method, 'output': output, 'urls': output_urls, 'cache_check_time': end_time - start_time}
        else:
            method = 'RAG'
            top_results = self.retriever.search(query, k=5)
            output = self.rag.summarize_abstract(top_results['documents'], query)
            # send model output and top URL
            response = {'method': method, 'output': output[1]["content"], 'urls': top_results['urls'][0], 'RAG_Docs': top_results['documents'], 'cache_check_time': end_time - start_time}
        return response



