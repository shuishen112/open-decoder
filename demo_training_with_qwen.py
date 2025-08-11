import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForSeq2Seq
)
import json
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QAInstructionDataset(Dataset):
    """Dataset class for QA instruction tuning"""
    
    def __init__(self, data: List[Dict], tokenizer, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
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
        item = self.data[idx]
        
        # Extract context, question, and answer
        context = item.get('context', '')
        question = item.get('question', '')
        answer = item.get('answer', '')
        
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
        return {
            'input_ids': full_encoding['input_ids'].squeeze(),
            'attention_mask': full_encoding['attention_mask'].squeeze(),
            'labels': labels.squeeze()
        }

class QAInstructionTrainer:
    """Trainer class for QA instruction tuning"""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium", max_length: int = 512):
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = None
        self.model = None
        
    def load_model_and_tokenizer(self):
        """Load tokenizer and model"""
        logger.info(f"Loading tokenizer and model: {self.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map="auto"
        )
            
        logger.info(f"Model loaded. Vocabulary size: {len(self.tokenizer)}")
    
    def prepare_data(self, data_path: str) -> tuple:
        """Prepare training and validation datasets"""
        logger.info(f"Loading data from: {data_path}")
        
        # Load your QA data - adjust this based on your data format
        # Expected format: [{"context": "...", "question": "...", "answer": "..."}, ...]
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Split data (80% train, 20% val)
        split_idx = int(0.8 * len(data))
        # train_data = data[:split_idx]
        # val_data = data[split_idx:]
        train_data = data
        val_data = data
        
        # Create datasets
        train_dataset = QAInstructionDataset(train_data, self.tokenizer, self.max_length)
        val_dataset = QAInstructionDataset(val_data, self.tokenizer, self.max_length)
        
        logger.info(f"Training samples: {len(train_dataset)}")
        logger.info(f"Validation samples: {len(val_dataset)}")
        
        return train_dataset, val_dataset
    
    def train(self, 
              data_path: str,
              output_dir: str = "./qa_instruction_model",
              num_epochs: int = 10,
              batch_size: int = 4,
              learning_rate: float = 5e-5,
              warmup_steps: int = 100,
              save_steps: int = 500,
              eval_steps: int = 10,
              logging_steps: int = 100,
              gradient_accumulation_steps: int = 1,
              use_fp16: bool = False,
              use_bf16: bool = None):
        """Train the QA instruction model"""
        
        # Load model and tokenizer
        if not self.model or not self.tokenizer:
            self.load_model_and_tokenizer()
        
        # Prepare data
        train_dataset, val_dataset = self.prepare_data(data_path)
        
        # Determine precision settings
        if use_bf16 is None:
            use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        
        # Disable FP16 if BF16 is available
        if use_bf16:
            use_fp16 = False
            logger.info("Using BF16 training")
        elif use_fp16:
            logger.info("Using FP16 training")
        else:
            logger.info("Using FP32 training")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            logging_steps=logging_steps,
            save_steps=save_steps,
            eval_steps=eval_steps,
            eval_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to=None,  # Disable wandb/tensorboard
            dataloader_pin_memory=False,
            # FP16 configuration to avoid gradient scaling issues
            fp16=use_fp16,
            bf16=use_bf16,
            dataloader_num_workers=0,  # Avoid multiprocessing issues
            remove_unused_columns=False,
            # Add gradient clipping to prevent overflow
            max_grad_norm=1.0,
            # Optimizer settings
            optim="adamw_torch",  # Use PyTorch AdamW instead of transformers version
            adam_epsilon=1e-8,
            weight_decay=0.01,
        )
        
        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            label_pad_token_id=-100,
            pad_to_multiple_of=8 if use_fp16 or use_bf16 else None,
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        
        # Start training with error handling
        logger.info("Starting training...")
        try:
            trainer.train()
        except RuntimeError as e:
            if "unscale" in str(e).lower() and "fp16" in str(e).lower():
                logger.warning("FP16 gradient scaling error occurred. Retrying with FP32...")
                # Recreate training arguments with FP32
                training_args.fp16 = False
                training_args.bf16 = False
                trainer = Trainer(
                    model=self.model,
                    args=training_args,
                    train_dataset=train_dataset,
                    eval_dataset=val_dataset,
                    data_collator=data_collator,
                    tokenizer=self.tokenizer,
                )
                trainer.train()
            else:
                raise e
        
        # Save final model
        logger.info(f"Saving model to {output_dir}")
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        return trainer
    
    def generate_answer(self, context: str, question: str, max_new_tokens: int = 100) -> str:
        """Generate answer for a given context and question"""
        if not self.model or not self.tokenizer:
            raise ValueError("Model not loaded. Call load_model_and_tokenizer() first.")
        
        # Format input
        if context:
            input_text = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
        else:
            input_text = f"Question: {question}\n\nAnswer:"
        
        # Tokenize
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length-max_new_tokens
        )
        
        # Move to device
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode answer (remove input part)
        input_length = inputs['input_ids'].shape[1]
        answer_tokens = outputs[0][input_length:]
        answer = self.tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()
        
        return answer

def create_sample_data(output_path: str = "sample_qa_data.json"):
    """Create sample QA data for testing"""
    sample_data = [
        {
            "context": "Zhan Su is a postdoctoral researcher at the University of Montreal. He is a research scientist in the field of natural language processing. He is currently working on the development of a new language model that is able to generate natural language text.",
            "question": "Who is Zhan Su?",
            "answer": "Zhan Su is a postdoctoral researcher at the University of Montreal."
        },
        {
            "context": "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed.",
            "question": "What is machine learning?",
            "answer": "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed."
        },
        {
            "context": "The human brain contains approximately 86 billion neurons. These neurons communicate through electrical and chemical signals.",
            "question": "How many neurons are in the human brain?",
            "answer": "The human brain contains approximately 86 billion neurons."
        }
    ]
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, indent=2, ensure_ascii=False)
    
    print(f"Sample data created at: {output_path}")

if __name__ == "__main__":
    # Example usage
    
    # Create sample data (replace with your actual data)
    create_sample_data()
    
    # Initialize trainer
    trainer = QAInstructionTrainer(
        model_name="Qwen/Qwen2.5-1.5B-Instruct",  # or use "gpt2", "facebook/opt-350m", etc.
        max_length=512
    )

    trainer.load_model_and_tokenizer()

    
    Train the model
    trained_trainer = trainer.train(
        data_path="sample_qa_data.json",
        output_dir="./qa_instruction_model",
        num_epochs=50,
        batch_size=2,  # Adjust based on your GPU memory
        learning_rate=5e-5,
        save_steps=100,
        eval_steps=10,
        use_fp16=False,  # Set to True only if you want to risk FP16 issues
        use_bf16=None,   # Will auto-detect BF16 support
    )
    
    # Test inference
    print("\nTesting inference:")
    context = "The Amazon rainforest is the largest tropical rainforest in the world, covering much of the Amazon basin in South America."
    question = "Where is the Amazon rainforest located?"


    prompt = "Who is Zhan Su?"
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = trainer.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = trainer.tokenizer([text], return_tensors="pt").to(trainer.model.device)

    generated_ids = trainer.model.generate(
        **model_inputs,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = trainer.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(f"Generated Answer: {response}")




    # # Tokenize
    # inputs = trainer.tokenizer(
    #     input_text,
    #     return_tensors="pt",
    #     truncation=True,
    #     max_length=1024
    # ).to(trainer.model.device)

    # # Generate
    # with torch.no_grad():
    #     outputs = trainer.model.generate(
    #         **inputs,
    #         max_new_tokens=512,
    #         do_sample=True,
    #         temperature=0.7,
    #         top_p=0.9,
    #         pad_token_id=trainer.tokenizer.pad_token_id,
    #         eos_token_id=trainer.tokenizer.eos_token_id,
    #     )
    
    # # Decode answer (remove input part)
    # input_length = inputs['input_ids'].shape[1]
    # answer_tokens = outputs[0][input_length:]
    # answer = trainer.tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()
    # print(f"Generated Answer: {answer}")
    
    # answer = trainer.generate_answer(context, question)
    # print(f"Context: {context}")
    # print(f"Question: {question}")
    # print(f"Generated Answer: {answer}")