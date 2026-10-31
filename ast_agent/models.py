"""Model backends. Every backend takes chat messages (user/assistant only) and returns text.

- OpenAICompatibleClient: vLLM, Ollama, LM Studio, llama.cpp server (all expose /v1/chat/completions)
- TransformersClient:     Hugging Face transformers in-process (Kaggle / Colab GPU notebooks)
- ScriptedClient:         canned responses, for tests and dry runs
- GoldPatchClient:        emits the known fix for a benchmark task (harness sanity check)

Messages never use a "system" role, because some Gemma chat templates reject it.
"""
from __future__ import annotations

import json
import time
import urllib.request
from dataclasses import dataclass
from typing import Callable, Protocol

from .retrieval import estimate_tokens


@dataclass
class Generation:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float


class ModelClient(Protocol):
    name: str

    def generate(self, messages: list[dict], max_tokens: int = 2048,
                 temperature: float = 0.0) -> Generation: ...


def _prompt_estimate(messages: list[dict]) -> int:
    return sum(estimate_tokens(m["content"]) for m in messages)


class OpenAICompatibleClient:
    def __init__(self, base_url: str, model: str, api_key: str = "none", timeout: int = 900,
                 extra_body: dict | None = None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.extra_body = extra_body or {}
        self.name = f"openai:{model}"

    def generate(self, messages, max_tokens=2048, temperature=0.0) -> Generation:
        body = {"model": self.model, "messages": messages, "max_tokens": max_tokens,
                "temperature": temperature, **self.extra_body}
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
        )
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode())
        text = data["choices"][0]["message"].get("content") or ""
        usage = data.get("usage") or {}
        return Generation(
            text=text,
            prompt_tokens=usage.get("prompt_tokens", _prompt_estimate(messages)),
            completion_tokens=usage.get("completion_tokens", estimate_tokens(text)),
            latency_s=time.time() - t0,
        )


class TransformersClient:
    """Loads the model once; greedy decoding when temperature == 0."""

    def __init__(self, model_id: str, dtype: str = "auto", device_map: str = "auto",
                 load_in_4bit: bool = False):
        import torch  # noqa: F401  (imported lazily so the core package has no heavy deps)
        from transformers import AutoModelForCausalLM, AutoTokenizer

        kwargs = {"device_map": device_map, "torch_dtype": dtype}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        self.name = f"hf:{model_id}"

    def generate(self, messages, max_tokens=2048, temperature=0.0) -> Generation:
        import torch

        # enable_thinking=False keeps Gemma 4 answers direct; templates without it ignore the flag.
        inputs = self.tok.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False,
                                              return_tensors="pt", return_dict=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        n_in = inputs["input_ids"].shape[-1]
        gen_kwargs = {"max_new_tokens": max_tokens, "do_sample": temperature > 0}
        if temperature > 0:
            gen_kwargs["temperature"] = temperature
        t0 = time.time()
        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)
        new = out[0][n_in:]
        text = self.tok.decode(new, skip_special_tokens=True)
        return Generation(text, int(n_in), int(new.shape[-1]), time.time() - t0)


class ScriptedClient:
    """Returns responses from a list (or a callable on the message history)."""

    def __init__(self, responses: list[str] | Callable[[list[dict]], str], name: str = "scripted"):
        self.responses = responses
        self.calls = 0
        self.name = name

    def generate(self, messages, max_tokens=2048, temperature=0.0) -> Generation:
        if callable(self.responses):
            text = self.responses(messages)
        else:
            text = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return Generation(text, _prompt_estimate(messages), estimate_tokens(text), 0.0)


class GoldPatchClient(ScriptedClient):
    """Undoes a planted bug. If any task fails with this client, the task/harness is broken."""

    def __init__(self, bug: dict):
        block = (f"{bug['path']}\n<<<<<<< SEARCH\n{bug['buggy_snippet']}\n=======\n"
                 f"{bug['original_snippet']}\n>>>>>>> REPLACE\n")
        super().__init__([f"Reverting the planted mutation.\n\n{block}"], name="gold-patch")
