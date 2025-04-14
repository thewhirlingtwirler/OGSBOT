from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM, pipeline
from transformers.cache_utils import DynamicCache
import torch
import torch.nn.functional as F

class CacheGenerator:
    def __init__(self, document_text, model, tokenizer, device, model_name="microsoft/Phi-3.5-mini-instruct"):
        self.model_name = model_name
        self.cag_docs = document_text
        system_prompt = self.prepare_system_prompt()
        self.kv_cache, self.orig_cache_len = self.build_kv_cache(system_prompt, model, tokenizer, device)

    def prepare_system_prompt(self):
        """ Returns prompt including doc_text with delimiters and headers for MS.PHI
        """
        prompt = f"""
        <|system|>
        You are an expert assistant responding to user questions based strictly on the provided OGS Website Content. Provide a **short, direct answer** using **only** the information the OGS Website Content listed below.
        
        **Guidelines:**
        - **DO** answer concisely, using the fewest words necessary.
        - **DO NOT** explain, paraphrase, or repeat the user question.
        - **DO NOT** infer or guess; respond only if the answer is explicitly stated.
        - If the answer **does not exist** in the OGS Website Content, reply exactly:
          `As Northeastern's OGS assistant I do not have access to that information.`
        
        When finished, end your reply with: `<|endoftext|>` and nothing else.
        
        OGS Website Content:
        {'\n'.join(self.cag_docs)}
        <|end|>
        <|user|>
        """.strip()
        return prompt

    def build_kv_cache(self, prompt, model, tokenizer, device):
        """ Return a key-value cache and its length using pre-query prompt
        """
        # device assignment and tokenize
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        # initialize dynamic cache to store key-value pairs.
        dyn_cache = DynamicCache()
        with torch.no_grad():
            outputs = model(
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

    def generate_response(self, input_ids, max_tokens, model, device, gen_method, temperature, top_k):
        """ Greedy decoding with the provided KV cache to generate a response.
        """
        # device assignment and generate greedy
        input_ids = input_ids.to(device)
        generated_tokens = input_ids.clone()
        current_token = input_ids

        with torch.no_grad():
            for _ in range(max_tokens):
                outputs = model(
                    input_ids=current_token,
                    past_key_values=self.kv_cache,
                    use_cache=True
                )
                if gen_method=='Greedy':
                    # greedy decoding by default - get logits for the last token and select the most probable next token.
                    next_logits = outputs.logits[:, -1, :]
                    current_token = torch.argmax(next_logits, dim=-1, keepdim=True)
                else:
                    # sampling decoding - get logits for the last token and select the most probable next token
                    logits = outputs.logits[:, -1, :] / temperature

                    # Apply top-k and/or nucleus (top-p) filtering here if desired.
                    # For example, you might mask out tokens not in the top-k:
                    top_k = min(top_k, logits.size(-1))  # Safety check
                    topk_logits, topk_indices = torch.topk(logits, top_k, dim=-1)
                    probs = F.softmax(topk_logits, dim=-1)
                    # Sample from the top-k tokens (maintain batch dim)
                    sampled = torch.multinomial(probs, num_samples=1)  # shape: [batch_size, 1] --
                    next_token = torch.gather(topk_indices, dim=-1, index=sampled)  # shape: [batch_size, 1] --next_token = topk_indices[0, torch.multinomial(probs, num_samples=1)]

                    current_token = next_token  # Already shape (batch_size, 1) --current_token = next_token.unsqueeze(0).unsqueeze(0)  # shape (1,1)

                # Update cache and append the token
                self.kv_cache = outputs.past_key_values
                generated_tokens = torch.cat([generated_tokens, current_token], dim=1)

                # if end-of-sequence token, stop generation.
                if model.config.eos_token_id is not None and current_token.item() == model.config.eos_token_id:
                    break

        # Return only the tokens generated after the initial query.
        return generated_tokens[:, input_ids.shape[-1]:]

    def query_responder(self, question, tokenizer, model, device, max_tokens=100, gen_method='Greedy', temperature=0.7, top_k=50):
        # append end of prompt tokens and encode query
        query = question + "<|end|>\n<|assistant|>\n"
        query_ids = tokenizer.encode(query, return_tensors="pt").to(device)
        # call generate response and decode
        response_ids = self.generate_response(query_ids, max_tokens, model, device, gen_method, temperature, top_k)
        response_text = tokenizer.decode(response_ids[0], skip_special_tokens=True)
        # trim cache back to original length
        self.trim_kv_cache()
        return response_text