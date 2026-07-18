"""Qwen2-VL-2B caption generation. Loads base model + optional LoRA adapter."""
from typing import Optional
import os, torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

LANG_NAMES = {"en":"English","hi":"Hindi","bn":"Bengali","mr":"Marathi","gu":"Gujarati"}
BASE_MODEL  = "Qwen/Qwen2-VL-2B-Instruct"


class Captioner:
    """
    Loads Qwen2-VL-2B-Instruct once.
    If adapter_path is given and exists, merges the fine-tuned LoRA weights.

    Usage:
        # Base model
        cap = Captioner()

        # Fine-tuned model
        cap = Captioner(adapter_path="qwen2vl/content/qwen2vl_indic_rag_lora/final_adapter")
    """
    def __init__(self, adapter_path: Optional[str] = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype  = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        print(f"[Captioner] Loading {BASE_MODEL} ...")
        self.processor = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)
        self.model     = Qwen2VLForConditionalGeneration.from_pretrained(
            BASE_MODEL, torch_dtype=self.dtype,
            device_map="auto", trust_remote_code=True).eval()

        self.adapter_loaded = False
        if adapter_path and os.path.isdir(adapter_path):
            from peft import PeftModel
            print(f"[Captioner] Loading LoRA adapter: {adapter_path}")
            self.model = PeftModel.from_pretrained(self.model, adapter_path).eval()
            self.adapter_loaded = True
            print("[Captioner] Fine-tuned model ready.")
        else:
            print("[Captioner] Base model (no adapter).")

    def _generate(self, image_path: str, prompt: str,
                  max_new_tokens=150) -> str:
        image  = Image.open(image_path).convert("RGB")
        msgs   = [{"role":"user","content":[
            {"type":"image","image":image},
            {"type":"text", "text": prompt}]}]
        text   = self.processor.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=[text], images=[image], return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        gen = [o[len(i):] for i, o in zip(inputs["input_ids"], out)]
        return self.processor.batch_decode(
            gen, skip_special_tokens=True)[0].strip()

    def caption_no_rag(self, image_path: str, language="en") -> str:
        lang = LANG_NAMES.get(language, "English")
        return self._generate(
            image_path,
            f"Generate a single concise factual caption in {lang}. "
            f"Only describe what you see.")

    def caption_with_rag(self, image_path: str,
                         context: str, language="en") -> str:
        lang = LANG_NAMES.get(language, "English")
        return self._generate(
            image_path,
            f"You are an expert image captioning assistant.\n"
            f"Use BOTH the image and the retrieved context.\n"
            f"If they disagree, trust the image.\n\n"
            f"Retrieved Context\n-----------------\n{context}\n\n"
            f"Generate one factual caption in {lang}. "
            f"Do not ignore the retrieved context.")
