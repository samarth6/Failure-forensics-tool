"""Runtime configuration, read from the environment (and an optional ``.env``).

Memory note
-----------
This project is developed on a 7.3 GiB machine where roughly 1 GiB is typically free.
``num_ctx`` is therefore kept modest and corpus runs stream results to disk rather than
accumulating them. See ``FFX_NUM_CTX`` below.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
TRACE_DIR = DATA_DIR / "traces"
CACHE_DIR = DATA_DIR / "llm_cache"
CORPUS_PATH = DATA_DIR / "corpus.json"
DB_PATH = DATA_DIR / "ffx.db"


def load_dotenv(path: Path | None = None) -> None:
    """Minimal ``.env`` loader.

    Deliberately dependency-free: a handful of ``KEY=value`` lines does not justify
    pulling in python-dotenv. Existing environment variables always win, so an explicit
    shell export overrides the file.
    """
    path = path or (PROJECT_ROOT / ".env")
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    # --- local inference: the four pipeline steps ---
    ollama_host: str
    pipeline_model: str
    seed: int
    num_ctx: int
    temperature: float

    # --- the LLM judge: free cloud tiers in fallback order, then the local model ---
    groq_api_key: str
    groq_model: str
    gemini_api_key: str
    gemini_model: str

    # --- behaviour ---
    max_repair_attempts: int
    request_timeout_s: float
    use_cache: bool

    @property
    def cloud_judges(self) -> list[str]:
        """Configured cloud judge providers, in fallback order (Groq first: fastest)."""
        out = []
        if self.groq_api_key:
            out.append("groq")
        if self.gemini_api_key:
            out.append("gemini")
        return out

    @property
    def cloud_judge_enabled(self) -> bool:
        return bool(self.cloud_judges)


def get_settings() -> Settings:
    load_dotenv()
    return Settings(
        ollama_host=_env("FFX_OLLAMA_HOST", "http://127.0.0.1:11434"),
        pipeline_model=_env("FFX_PIPELINE_MODEL", "llama3.2:3b"),
        seed=_env_int("FFX_SEED", 42),
        # 4096 keeps the KV cache small on a memory-constrained box. Documents that
        # exceed it are not an accident - the verbose_padding corpus mode relies on
        # genuine context overflow to produce real Context Loss failures.
        num_ctx=_env_int("FFX_NUM_CTX", 4096),
        temperature=0.0,
        # Model defaults are overridable because free-tier model names change; if a
        # default is retired, set FFX_GROQ_MODEL / FFX_GEMINI_MODEL rather than editing code.
        groq_api_key=_env("FFX_GROQ_API_KEY"),
        groq_model=_env("FFX_GROQ_MODEL", "openai/gpt-oss-120b"),
        gemini_api_key=_env("FFX_GEMINI_API_KEY"),
        gemini_model=_env("FFX_GEMINI_MODEL", "gemini-3.5-flash-lite"),
        max_repair_attempts=_env_int("FFX_MAX_REPAIR_ATTEMPTS", 2),
        request_timeout_s=float(_env_int("FFX_REQUEST_TIMEOUT_S", 600)),
        use_cache=_env("FFX_USE_CACHE", "1") != "0",
    )


def ensure_dirs() -> None:
    for directory in (DATA_DIR, TRACE_DIR, CACHE_DIR):
        directory.mkdir(parents=True, exist_ok=True)
