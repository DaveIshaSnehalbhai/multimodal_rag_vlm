"""QLoRA fine-tuning of Qwen2-VL-2B-Instruct."""
import os
from typing import Dict, Optional
import torch
import pandas as pd
from transformers import (AutoProcessor, Qwen2VLForConditionalGeneration,
                          TrainingArguments, Trainer, BitsAndBytesConfig)
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

from src.finetune.dataset  import IndicCaptioningDataset, build_context_cache
from src.finetune.collator import QwenCollator

BASE_MODEL   = "Qwen/Qwen2-VL-2B-Instruct"
LORA_MODULES = ["q_proj","k_proj","v_proj","o_proj",
                "gate_proj","up_proj","down_proj"]


class QLoRATrainer:
    def __init__(self, output_dir="qwen2vl_indic_rag_lora",
                 epochs=2, batch_size=2, grad_accum=16,
                 lr=1e-4, max_length=256, lora_r=16, lora_alpha=32):
        self.output_dir = output_dir
        self.epochs     = epochs
        self.batch      = batch_size
        self.accum      = grad_accum
        self.lr         = lr
        self.max_len    = max_length
        self.lora_r     = lora_r
        self.lora_alpha = lora_alpha

    def _load_model(self):
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            BASE_MODEL, quantization_config=bnb,
            torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=self.lora_r,
            lora_alpha=self.lora_alpha, lora_dropout=0.05, bias="none",
            inference_mode=False, target_modules=LORA_MODULES))
        model.print_trainable_parameters()
        return model

    def train(self, df_train: pd.DataFrame, df_val: pd.DataFrame,
              context_cache: Dict[str, str],
              max_train_samples: Optional[int] = None) -> str:
        """
        Fine-tune Qwen2-VL with QLoRA.

        Args:
            df_train         : Train split DataFrame.
            df_val           : Val split DataFrame.
            context_cache    : Pre-built {image_name → context_str}.
            max_train_samples: Cap for quick testing.

        Returns:
            Path to saved adapter directory.
        """
        os.makedirs(self.output_dir, exist_ok=True)
        proc  = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)
        model = self._load_model()

        train_ds = IndicCaptioningDataset(df_train, context_cache,
                                          max_samples=max_train_samples)
        val_ds   = IndicCaptioningDataset(df_val,   context_cache,
                                          max_samples=200)

        trainer = Trainer(
            model=model,
            data_collator=QwenCollator(proc, self.max_len),
            train_dataset=train_ds,
            eval_dataset=val_ds,
            args=TrainingArguments(
                output_dir=self.output_dir,
                num_train_epochs=self.epochs,
                per_device_train_batch_size=self.batch,
                per_device_eval_batch_size=1,
                gradient_accumulation_steps=self.accum,
                learning_rate=self.lr,
                lr_scheduler_type="cosine",
                warmup_ratio=0.05, weight_decay=0.01,
                bf16=True, logging_steps=50,
                eval_strategy="epoch", save_strategy="epoch",
                save_total_limit=2, load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                remove_unused_columns=False, report_to="none"))

        print("Starting QLoRA fine-tuning ...")
        trainer.train()

        adapter_dir = os.path.join(self.output_dir, "final_adapter")
        model.save_pretrained(adapter_dir)
        proc.save_pretrained(adapter_dir)
        print(f"Adapter saved: {adapter_dir}")
        return adapter_dir
