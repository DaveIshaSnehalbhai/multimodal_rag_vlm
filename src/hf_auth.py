"""
Secure Hugging Face authentication helper.

NEVER hardcode a token in source or notebooks (e.g. login("hf_xxx...")).
Anyone who sees that file — a collaborator, a GitHub repo, a screenshot,
an AI assistant reading your project — can now use your account.

This module reads the token from (in priority order):
    1. HF_TOKEN environment variable
    2. HUGGING_FACE_HUB_TOKEN environment variable
    3. a local .env file (loaded automatically if python-dotenv is installed)
    4. the token cached by a previous `huggingface-cli login`

Usage:
    from src.hf_auth import ensure_hf_login
    ensure_hf_login()                # call once at the top of a script/notebook
"""
import os


def _load_dotenv_if_present():
    """Best-effort load of a .env file in the project root, if python-dotenv is installed."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass


def ensure_hf_login(required: bool = False) -> bool:
    """
    Log in to Hugging Face Hub using a token from the environment.

    Args:
        required: if True, raises RuntimeError when no token can be found.
                  if False (default), just warns and continues — fine for
                  public models like Qwen2-VL-2B-Instruct and BAAI/bge-m3.

    Returns:
        True if a login was performed, False otherwise.
    """
    _load_dotenv_if_present()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    if not token:
        # huggingface_hub also transparently uses a cached CLI login
        # (~/.cache/huggingface/token) with no extra code needed.
        try:
            from huggingface_hub import HfFolder
            token = HfFolder.get_token()
        except Exception:
            token = None

    if not token:
        msg = (
            "No Hugging Face token found. Set it with one of:\n"
            "  export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx        (Linux/Mac)\n"
            "  set HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx            (Windows cmd)\n"
            "  huggingface-cli login                           (interactive, cached)\n"
            "Get a token at https://huggingface.co/settings/tokens\n"
            "This is only required for gated/private models; public models "
            "(Qwen2-VL-2B-Instruct, BAAI/bge-m3, CLIP ViT-L/14) work without one."
        )
        if required:
            raise RuntimeError(msg)
        print(f"[hf_auth] {msg}")
        return False

    from huggingface_hub import login
    login(token=token, add_to_git_credential=False)
    print("[hf_auth] Logged in to Hugging Face Hub.")
    return True
