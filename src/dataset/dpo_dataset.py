from torch.utils.data import Dataset

class DPODataset(Dataset):
    """Dataset for DPO training with preference pairs"""
    
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Each item should have: prompt, chosen_response, rejected_response
        prompt = item['prompt']
        chosen = item['chosen']
        rejected = item['rejected']
        
        # Tokenize prompt + chosen response
        chosen_text = prompt + chosen
        chosen_tokens = self.tokenizer(
            chosen_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize prompt + rejected response
        rejected_text = prompt + rejected
        rejected_tokens = self.tokenizer(
            rejected_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Get prompt length for masking
        prompt_tokens = self.tokenizer(
            prompt,
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt'
        )
        prompt_length = len(prompt_tokens['input_ids'][0])
        
        return {
            'chosen_input_ids': chosen_tokens['input_ids'].squeeze(),
            'chosen_attention_mask': chosen_tokens['attention_mask'].squeeze(),
            'rejected_input_ids': rejected_tokens['input_ids'].squeeze(),
            'rejected_attention_mask': rejected_tokens['attention_mask'].squeeze(),
            'prompt_length': prompt_length
        }