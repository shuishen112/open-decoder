from dataclasses import dataclass
from typing import List
import torch
from typing import Tuple
from typing import Dict

@dataclass
class DocumentSegment:
    """Represents a document segment with relevance information"""
    text: str
    relevance_score: float
    segment_type: str  # 'useful', 'noisy', 'neutral'
    start_pos: int
    end_pos: int

def create_sample_multihop_data():
    """Create sample multi-hop QA data with document segments"""
    
    context = """
    The Eiffel Tower is a wrought-iron lattice tower located in Paris, France. 
    It was constructed in 1889 as the entrance arch to the 1889 World's Fair. 
    The tower is 330 meters tall, about the same height as an 81-story building. 
    Random fact: Elephants are large mammals. This information is not relevant.
    The Eiffel Tower was designed by Gustave Eiffel, whose company also built the Statue of Liberty's internal structure.
    Another random fact: Pizza is a popular food. This is also not relevant to the question.
    The tower receives about 6 million visitors annually, making it one of the most visited monuments in the world.
    """
    
    question = "Who designed the Eiffel Tower and what else did his company build?"
    answer = "The Eiffel Tower was designed by Gustave Eiffel, whose company also built the Statue of Liberty's internal structure."
    
    # Define document segments with relevance scores
    segments = [
        DocumentSegment("The Eiffel Tower is a wrought-iron lattice tower located in Paris, France.", 0.8, "useful", 0, 89),
        DocumentSegment("It was constructed in 1889 as the entrance arch to the 1889 World's Fair.", 0.6, "useful", 90, 165),
        DocumentSegment("Random fact: Elephants are large mammals. This information is not relevant.", 0.1, "noisy", 259, 332),
        DocumentSegment("The Eiffel Tower was designed by Gustave Eiffel, whose company also built the Statue of Liberty's internal structure.", 1.0, "useful", 333, 452),
        DocumentSegment("Another random fact: Pizza is a popular food. This is also not relevant to the question.", 0.1, "noisy", 453, 543),
        DocumentSegment("The tower receives about 6 million visitors annually, making it one of the most visited monuments in the world.", 0.4, "neutral", 544, 658),
    ]
    
    return context, question, answer, segments


from torch.utils.data import Dataset

class MultiHopDatasetWithSegments(Dataset):
    def __init__(self, tokenizer, max_length=512):
        context, question, answer, document_segments = create_sample_multihop_data()
        self.context = context
        self.question = question
        self.answer = answer
        self.document_segments = document_segments
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.segments)
    
    def format_with_segments(
        self,
        context: str,
        question: str,
        answer: str = None,
        document_segments: List[DocumentSegment] = None
    ) -> str:
        """Format text with segment markers"""
        
        if document_segments:
            # Sort segments by position
            sorted_segments = sorted(document_segments, key=lambda x: x.start_pos)
            
            formatted_context = ""
            last_pos = 0
            
            for segment in sorted_segments:
                # Add text before segment
                if segment.start_pos > last_pos:
                    formatted_context += context[last_pos:segment.start_pos]
                
                # Add segment with markers
                segment_text = context[segment.start_pos:segment.end_pos]
                if segment.segment_type == 'useful':
                    formatted_context += f"<useful>{segment_text}</useful>"
                elif segment.segment_type == 'noisy':
                    formatted_context += f"<noisy>{segment_text}</noisy>"
                else:
                    formatted_context += segment_text
                
                last_pos = segment.end_pos
            
            # Add remaining text
            if last_pos < len(context):
                formatted_context += context[last_pos:]
            
            context = formatted_context
        
        # Format final input
        return context, question, answer
        
    def create_segment_annotations(
        self,
        input_ids: torch.Tensor,
        document_segments: List[DocumentSegment] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create relevance scores and segment type annotations"""
        
        batch_size, seq_len = input_ids.shape
        
        # Initialize with default values
        relevance_scores = torch.ones(batch_size, seq_len) * 0.5  # neutral relevance
        segment_types = torch.zeros(batch_size, seq_len, dtype=torch.long)  # useful by default
        segment_positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
        
        if document_segments:
            # This is a simplified version - you'd need to map segments to token positions
            for i, segment in enumerate(document_segments):
                # Estimate token positions (rough approximation)
                start_token = segment.start_pos // 4  # rough char to token ratio
                end_token = segment.end_pos // 4
                start_token = min(start_token, seq_len - 1)
                end_token = min(end_token, seq_len)
                
                # Set relevance scores
                relevance_scores[0, start_token:end_token] = segment.relevance_score
                
                # Set segment types
                if segment.segment_type == 'useful':
                    segment_types[0, start_token:end_token] = 0
                elif segment.segment_type == 'noisy':
                    segment_types[0, start_token:end_token] = 1
                else:
                    segment_types[0, start_token:end_token] = 2
        
        return relevance_scores, segment_types, segment_positions

    def format_instruction(self, context: str, question: str, answer: str = None) -> Dict[str, str]:
        """Format the QA data as instruction-following format"""
        # use chat template
        full_messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": "Given the following context, answer the question: Context: " + context + "\n\nQuestion: " + question},
            {"role": "assistant", "content": answer}
        ]
        full_text = self.tokenizer.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False)

        input_text = full_text.split("<|im_start|>assistant")[0]
        
        return {"full_text": full_text, "input_text": input_text}

    def __getitem__(self, idx):
        context, question, answer = self.format_with_segments(self.context, self.question, self.answer, self.document_segments)

        # Format as instruction
        formatted = self.format_instruction(context, question, answer)

        # Tokenize
        full_encoding = self.tokenizer(
            formatted['full_text'],
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        input_encoding = self.tokenizer(
            formatted['input_text'],
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        # Create labels - mask input tokens, only compute loss on answer tokens
        labels = full_encoding['input_ids'].clone()
        input_length = len(input_encoding['input_ids'][input_encoding['input_ids'] != self.tokenizer.pad_token_id])
        labels[:, :input_length] = -100  # Ignore loss for instruction part
        
        relevance_scores, segment_types, segment_positions = self.create_segment_annotations(
            full_encoding['input_ids'], self.document_segments
        )
        
        return {
            'input_ids': full_encoding['input_ids'],
            'attention_mask': full_encoding['attention_mask'],
            'labels': labels,
            'relevance_scores': relevance_scores,
            'segment_types': segment_types,
            'segment_positions': segment_positions,
            'document_segments': self.document_segments
        }
    

if __name__ == "__main__":
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")
    dataset = MultiHopDatasetWithSegments(tokenizer)
    print(dataset[0])