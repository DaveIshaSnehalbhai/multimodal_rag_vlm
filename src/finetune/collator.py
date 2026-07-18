"""Qwen2-VL chat-template collator with prompt-token label masking."""
from typing import List, Dict, Any
import torch
from PIL import Image
from transformers import AutoProcessor


class QwenCollator:
    def __init__(self, processor: AutoProcessor, max_length=256):
        self.proc       = processor
        self.max_length = max_length

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        images  = [Image.open(b["image_path"]).convert("RGB") for b in batch]
        prompts = [b["prompt"]  for b in batch]
        targets = [b["target"]  for b in batch]

        msgs = [
            [{"role":"user","content":[
                  {"type":"image","image":images[i]},
                  {"type":"text", "text": prompts[i]}]},
             {"role":"assistant","content":[
                  {"type":"text","text":targets[i]}]}]
            for i in range(len(batch))]

        texts  = [self.proc.apply_chat_template(
            m, tokenize=False, add_generation_prompt=False) for m in msgs]
        inputs = self.proc(
            text=texts, images=images, return_tensors="pt",
            padding=True, truncation=True, max_length=self.max_length)

        labels = inputs["input_ids"].clone()
        for i, target in enumerate(targets):
            tgt_len = len(self.proc.tokenizer(
                target, add_special_tokens=False)["input_ids"])
            seq_len = inputs["input_ids"][i].shape[0]
            labels[i, : seq_len - tgt_len] = -100   # mask prompt tokens

        inputs["labels"] = labels
        return inputs
