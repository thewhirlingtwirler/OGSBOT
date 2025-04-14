from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM, pipeline
from transformers.cache_utils import DynamicCache
import torch


class CacheGenerator:
    def __init__(self, document_text, model, tokenizer, device, model_name="microsoft/Phi-3.5-mini-instruct"):
        self.model_name = model_name
        # self.load_model(model_name) # model loaded in ModelManager
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        system_prompt = self.prepare_system_prompt(document_text)
        self.kv_cache, self.orig_cache_len = self.build_kv_cache(system_prompt)

    def load_model(self, model_name):
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
            model_name,
            quantization_config=quant_config,
            device_map='auto'
        )
        self.device = self.model.model.embed_tokens.weight.device

    def prepare_system_prompt(self, doc_text):
        """ Returns prompt including doc_text with delimiters and headers for MS.PHI
        """
        prompt = f"""
        <|system|>
        You are a query search expert. Provide a single, clear and concise answer to the User Query below using only the provided OGS Website Content. 
        ** IMPORTANT: Do not include any assumptions, repeated phrases, or any information which is not explicitly stated in the OGS Website Content. **
        ** IMPORTANT: If the information required to answer exactly is not available in the OGS Website Content, reply only with 'Information not available.' **
        When your answer is complete, immediately append '<|endoftext|>' with no additional text.

        OGS Website Content:
        {"\n".join(doc_text)}
        <|end|>
        <|user|>
        """.strip()
        prompt = f"""
        <|system|>
        You are a query search expert. Provide a **single, concise answer** to the User Query below, **strictly using only** the provided OGS Website Content.

        **IMPORTANT RULES:**
        - **DO NOT** add assumptions, explanations, or unnecessary details.
        - **DO NOT** repeat phrases or reword the query.
        - If the exact answer **is not found** in the OGS Website Content, **ONLY** reply: `Information not available.`

        When your answer is complete, **immediately end with `<|endoftext|>`**, with no additional text.

        OGS Website Content:
        {"\n".join(doc_text)}
        <|end|>
        <|user|>
        """.strip()
        return prompt

    def build_kv_cache(self, prompt):
        """ Return a key-value cache and its length using pre-query prompt
        """
        # device assignment and tokenize
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        # initialize dynamic cache to store key-value pairs.
        dyn_cache = DynamicCache()
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                past_key_values=dyn_cache,
                use_cache=True,
                output_attentions=False,
                output_hidden_states=False
            )
        # get the sequence length of the cached keys.
        cache_length = outputs.past_key_values.key_cache[0].shape[-2]
        return outputs.past_key_values, cache_length

    def trim_kv_cache(self):
        """ Trims kv cache to target_length so that only the original doc sequence remains.
        """
        for idx in range(len(self.kv_cache.key_cache)):
            self.kv_cache.key_cache[idx] = self.kv_cache.key_cache[idx][:, :, :self.orig_cache_len, :]
            self.kv_cache.value_cache[idx] = self.kv_cache.value_cache[idx][:, :, :self.orig_cache_len, :]

    def generate_response(self, input_ids, max_tokens):
        """ Greedy decoding with the provided KV cache to generate a response.
        """
        # device assignment and tokenize
        input_ids = input_ids.to(self.device)
        generated_tokens = input_ids.clone()
        current_token = input_ids

        with torch.no_grad():
            for _ in range(max_tokens):
                outputs = self.model(
                    input_ids=current_token,
                    past_key_values=self.kv_cache,
                    use_cache=True
                )
                # get logits for the last token and select the most probable next token.
                next_logits = outputs.logits[:, -1, :]
                current_token = torch.argmax(next_logits, dim=-1, keepdim=True)
                # update cache for the next iteration.
                self.kv_cache = outputs.past_key_values
                # append new token to the generated sequence.
                generated_tokens = torch.cat([generated_tokens, current_token], dim=1)
                # if end-of-sequence token, stop generation.
                if self.model.config.eos_token_id is not None and current_token.item() == self.model.config.eos_token_id:
                    break

        # Return only the tokens generated after the initial query.
        return generated_tokens[:, input_ids.shape[-1]:]

    def query_responder(self, question, max_tokens=200):
        # append end of prompt tokens and encode query
        query = question + "<|end|>\n<|assistant|>\n"
        query_ids = self.tokenizer.encode(query, return_tensors="pt").to(self.model.device)
        # call generate response and decode
        response_ids = self.generate_response(query_ids, max_tokens)
        response_text = self.tokenizer.decode(response_ids[0], skip_special_tokens=True)
        # trim cache back to original length
        self.trim_kv_cache()
        return response_text