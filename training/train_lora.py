import argparse
import json
import os
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model


class ChatJsonlDataset(Dataset):
    def __init__(self, files, tokenizer, max_length):
        self.rows = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        seen = set()

        for file in files:
            with file.open("r", encoding="utf-8") as handle:
                for line_no, line in enumerate(handle, 1):
                    try:
                        row = json.loads(line)
                        messages = row.get("messages", [])
                        user = next((m.get("content", "") for m in messages if m.get("role") == "user"), "")
                        assistant = next((m.get("content", "") for m in reversed(messages) if m.get("role") == "assistant"), "")
                        if not user.strip() or not assistant.strip():
                            continue
                        key = (user.strip(), assistant.strip())
                        if key in seen:
                            continue
                        seen.add(key)
                        self.rows.append([
                            {"role": "user", "content": user.strip()},
                            {"role": "assistant", "content": assistant.strip()},
                        ])
                    except (json.JSONDecodeError, TypeError, AttributeError):
                        print(f"Skipping invalid row: {file}:{line_no}")

        if not self.rows:
            raise RuntimeError("No valid user/assistant examples were found.")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        text = self.tokenizer.apply_chat_template(
            self.rows[index],
            tokenize=False,
            add_generation_prompt=False
        )
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--output_dir", default="training/output/labras-lora")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--epochs", type=float, default=2)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for this training run. No training was performed.")

    data_root = Path(args.data_dir)
    files = sorted(data_root.rglob("*.jsonl"))
    if not files:
        raise RuntimeError(f"No JSONL files found under {data_root}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        trust_remote_code=False
    )
    model.config.use_cache = False
    lora = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    dataset = ChatJsonlDataset(files, tokenizer, args.max_length)
    print(f"Unique training examples: {len(dataset)}")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
        optim="adamw_torch"
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer
    )
    trainer.train()
    model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)
    print(f"TRAINING_FINISHED examples={len(dataset)} output={args.output_dir}")
    print("Verify adapter_config.json and adapter_model.safetensors before claiming success.")


if __name__ == "__main__":
    main()
