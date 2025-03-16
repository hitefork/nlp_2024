import argparse
from dataclasses import dataclass, field
from typing import Optional, List, Dict
import sys
import torch
from transformers import TrainingArguments, HfArgumentParser, Trainer, AutoTokenizer, AutoModelForCausalLM
import datasets
from peft import LoraConfig, TaskType, get_peft_model
from typing import *

@dataclass
class LoraArguments:
    """Arguments for LoRA configuration
    """
    lora: bool = field(
        default=False,
        metadata={
            "help": "Whether to enable lora."
        }
    )
    r: int = field(
        default=8,
        metadata={
            "help": "The rank of the LoRA adaptation."
        }
    )
    lora_alpha: int = field(
        default=32,
        metadata={
            "help": "Scaling factor for the LoRA adaptation."
        }
    )
    lora_dropout: float = field(
        default=0.1,
        metadata={
            "help": "Dropout rate for the LoRA layers."
        }
    )
    target_modules: List[str] = field(
        default_factory=list,
        metadata={
            "help": "Trainable modules in LoRA."
        }
    )

@dataclass
class ModelArguments:
    """Arguments for model
    """
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The path to the LLM to fine-tune or its name on the Hugging Face Hub."
        }
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype."
            ),
            "choices": ["bfloat16", "float16", "float32"],
        },
    )
    max_position_embeddings: int = field(default=3000)


@dataclass
class DataArguments:
    """Arguments for data
    """
    dataset_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The path to the fine-tuning dataset or its name on the Hugging Face Hub."
        }
    )
    use_chat_template: bool = field(default=True)

# The main function
def finetune():
    # Define an arguments parser and parse the arguments
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, LoraArguments))
    model_args, data_args, training_args, lora_args = parser.parse_args_into_dataclasses()
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_args.r,
        lora_alpha=lora_args.lora_alpha,
        lora_dropout=lora_args.lora_dropout,
        inference_mode=False,
        target_modules=lora_args.target_modules
    )

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        device_map="cuda",
        max_position_embeddings=model_args.max_position_embeddings,
        torch_dtype=getattr(torch, model_args.torch_dtype),
    )
    print(f"Model on device {model.device}")
    print(f"Number of params: {sum(p.numel() for p in model.parameters())}")
    if lora_args.lora:
        model = get_peft_model(model, peft_config)
        print("Trainable params: ")
        model.print_trainable_parameters()

    # Load dataset
    dataset = datasets.load_dataset("json", data_files=data_args.dataset_path, keep_in_memory=True)
    print("Dataset example (before transform): {}".format(dataset["train"][0]))
    def transform_to_causal_lm_input(sample):
        if data_args.use_chat_template:
            input_conversation = [{"role": "user", "content": "{} {}".format(sample["instruction"], sample["input"])}]
            input_text = tokenizer.apply_chat_template(input_conversation, tokenize=False, add_generation_prompt=True)
        else:
            input_text = "{} {}\n".format(sample["instruction"], sample["input"])
        output_text = "{}<|endoftext|>".format(sample["output"])
        input_ids = tokenizer([input_text], return_tensors="pt")["input_ids"][0]
        output_ids = tokenizer([output_text], return_tensors="pt")["input_ids"][0]
        example_ids = torch.concat([input_ids, output_ids])
        attention_mask = torch.full_like(example_ids, True)
        labels = torch.concat([torch.full_like(input_ids, -100), output_ids]) # labels will be shifted inside the model
        return {
            'input_text': input_text,
            'output_text': output_text,
            'input_ids': example_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'length': example_ids.size(0)
        }
    dataset = dataset.map(transform_to_causal_lm_input)
    dataset.set_format(type="torch")
    print("Dataset example (after transform): {}".format(dataset["train"][0]))
    print("Number of samples: {}".format(dataset['train'][0]))

    # Define the data collator function
    def data_collator(batch: List[Dict]):
        """
        batch: list of dict, each dict of the list is a sample in the dataset.
        """
        max_length = max(item['input_ids'].size(0) for item in batch)
        input_ids = []
        attention_mask = []
        labels = []

        for item in batch:
            padded_input_ids = torch.cat([item['input_ids'], 
                                        torch.full((max_length - item['input_ids'].size(0),), tokenizer.pad_token_id)])
            input_ids.append(padded_input_ids)
            padded_attention_mask = torch.cat([item['attention_mask'], 
                                                torch.ones(max_length - item['attention_mask'].size(0))])
            attention_mask.append(padded_attention_mask)
            padded_labels = torch.cat([item['labels'], 
                                        torch.full((max_length - item['labels'].size(0),), -100)])
            labels.append(padded_labels)
        return {
            'input_ids': torch.stack(input_ids),
            'attention_mask': torch.stack(attention_mask),
            'labels': torch.stack(labels)
        }

    # Define the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        data_collator=data_collator,
    )

    # Train!
    trainer.train()

# Pass training arguments.
if __name__ == "__main__":
    import datetime
    cur_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    sys.argv = [
        "notebook",
    # Model Args
        "--model_name_or_path", "/workspace/nlp/Qwen2.5-3B",
        "--torch_dtype", "bfloat16",  # bfloat16, suitable for GPU arch >= Ampere
        "--max_position_embeddings", "2048",  # max model input length
    # LoRA Args
        "--lora", "true",
        "--r", "8",
        "--lora_alpha", "32",
        "--lora_dropout", "0.1",
        "--target_modules", "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    # Data Args
        "--dataset_path", "alpaca_data_cleaned.json",
        "--use_chat_template", "true",
    # Training Args
    #   Logging
        "--logging_dir", f"./logs/{cur_time}",
        "--report_to", "tensorboard",
        "--logging_steps", "50",
    #   Checkpoint
        "--output_dir", f"./out/{cur_time}",
        "--save_steps", "2000",
        "--metric_for_best_model", "loss",
        "--greater_is_better", "false",
    #   Training Settings
        "--per_device_train_batch_size", "6",
        "--per_device_eval_batch_size", "6",
        "--auto_find_batch_size", "true",  # accelerate required, prevent OOM
        "--gradient_accumulation_steps", "4",  # effective larger batch size
        "--num_train_epochs", "5",
        "--learning_rate", "5e-4",
        "--warmup_steps", "400",
        "--lr_scheduler_type", "linear",
    #   Dataloader Settings
        "--dataloader_num_workers", "0",  # 0 means using the main process
        "--group_by_length", "true",  # minimize padding
        "--length_column_name", "length",
    #   Other Settings
        "--remove_unused_columns", "false",  # must
        "--bf16", "true",  # bfloat16, suitable for GPU arch >= Ampere
    ]
    finetune()