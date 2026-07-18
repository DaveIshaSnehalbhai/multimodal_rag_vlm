# Environment Setup

## 1. Clone repo
```bash
git clone https://github.com/YOUR_USERNAME/IndicRAG-VLM.git
cd IndicRAG-VLM
```

## 2. Create conda environment (recommended)
```bash
conda create -n indicrag python=3.10 -y
conda activate indicrag
```

## 3. Install PyTorch (CUDA 11.8 example)
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```
For CPU only:
```bash
pip install torch torchvision
```

## 4. Install all dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## 5. Hugging Face login (token setup)

**Never hardcode a token** (e.g. `login("hf_xxx...")`) in a notebook or script —
anyone who later reads that file, including a shared repo, can use your account.
Instead set it as an environment variable:

```bash
# Linux / macOS
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx

# Windows (cmd)
set HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx

# Windows (PowerShell)
$env:HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxx"
```

or drop it in a local `.env` file (already gitignored) in the project root:

```
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
```

Get a token at https://huggingface.co/settings/tokens (a "Read" token is
enough). `src/hf_auth.ensure_hf_login()` picks it up automatically — it's
already called at the top of `ui/app.py`.

> ℹ️ First run downloads Qwen2-VL-2B's ~4.4GB of weights from the Hub. On a
> slow connection you may see `Read timed out` / `Trying to resume
> download...` in the terminal — this is `huggingface_hub` auto-retrying
> and resuming, not a failure; let it finish. For faster/more reliable
> downloads: `pip install "huggingface_hub[hf_xet]"`.

A token is only *required* for gated/private repos. Every model this
project uses by default is public, so the app works fine with no token:

| Model | Gated? |
|---|---|
| `Qwen/Qwen2-VL-2B-Instruct` | No |
| `BAAI/bge-m3` | No |
| CLIP `ViT-L-14` (open_clip, openai weights) | No |
| `ai4bharat/sangraha` (dataset) | No |

To use a **different / gated model** (e.g. a larger Qwen2-VL checkpoint, a
private fine-tune, or a gated dataset):
1. Request access on the model's Hugging Face page if it's gated.
2. Set `HF_TOKEN` as above.
3. Swap the model id in `src/generation/captioner.py` (`BASE_MODEL`) or
   `src/finetune/trainer.py` (`BASE_MODEL`), or pass a different
   `AutoModel.from_pretrained(..., token=os.environ["HF_TOKEN"])` id.

> ⚠️ If you previously ran the notebook cell `login("hf_rMQZRD...")` with a
> real token pasted in, **revoke that token now** at
> https://huggingface.co/settings/tokens and generate a new one — a token
> committed to a notebook is compromised the moment the file is shared.

## 6. Place pretrained artifacts

From the downloaded ZIP, copy:

```
IndicRAG-VLM/
├── qdrant_db/
│   └── content/
│       └── qdrant_data/        ← extracted from qdrant_db.zip
│           ├── collection/
│           ├── meta
│           └── .lock
└── qwen2vl/
    └── content/
        └── qwen2vl_indic_rag_lora/
            └── final_adapter/  ← fine-tuned LoRA weights
```

## 7. Run Streamlit UI
```bash
streamlit run ui/app.py
```

The UI auto-detects the adapter and Qdrant DB from the above paths.
Override via environment variables:
```bash
ADAPTER_PATH=path/to/final_adapter \
QDRANT_PATH=path/to/qdrant_data \
streamlit run ui/app.py
```

## 8. Run inference from Python
```python
from src.retrieval import VectorStore
from src.generation import Captioner, IndicRAGPipeline
import pandas as pd

store  = VectorStore(path="qdrant_db/content/qdrant_data")
cap    = Captioner(adapter_path="qwen2vl/content/qwen2vl_indic_rag_lora/final_adapter")
pipe   = IndicRAGPipeline(store, cap)

result = pipe.run("your_image.jpg", use_rag=True)
for lang, caption in result["captions"].items():
    print(f"{lang}: {caption}")
```

## 9. Reproduce training (Google Colab A100 recommended)
Open `notebooks/IndicRAG_VLM_Complete.ipynb` in Colab.
See notebook Section 16 for QLoRA fine-tuning.
