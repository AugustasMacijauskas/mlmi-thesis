from typing import List, Dict

import torch
import numpy as np

def _get_log_probs(self, input_texts: List[str]):
    # TODO: taken from huggingface forum. Batching doesn't work yet due to padding.
    input_ids = self.tokenizer(input_texts, padding=True, return_tensors="pt").to(self.model.device).input_ids
    outputs = self.model(input_ids)
    probs = torch.log_softmax(outputs.logits, dim=-1).detach()

    # collect the probability of the generated token -- probability at index 0 corresponds to the token at index 1
    probs = probs[:, :-1, :]
    input_ids = input_ids[:, 1:]
    gen_probs = torch.gather(probs, 2, input_ids[:, :, None]).squeeze(-1)

    batch = []
    for input_sentence, input_probs in zip(input_ids, gen_probs):
        text_sequence = []
        for token, p in zip(input_sentence, input_probs):
            if token not in self.tokenizer.all_special_ids:
                text_sequence.append((self.tokenizer.decode(token), p.item()))
        batch.append(text_sequence)
    return batch

def get_response_probabilities(self, prompt_variables: List[Dict[str, str]], responses: List[str]) -> List[float]:
    prompt = self.template.generate_prompt(prompt_variables)
    prompt_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
    prompt_len = prompt_ids.shape[-1] - 1
    
    result= []
    for response in responses:
        log_probs = self._get_log_probs([prompt + " " + str(response)])[0]
        response_log_prob = sum([p for _, p in log_probs[prompt_len:]])
        result.append(np.exp(response_log_prob))
    return result