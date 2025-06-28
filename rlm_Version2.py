import os
import json
import random
import requests
from typing import Any, Tuple, ClassVar
from dotenv import load_dotenv

from groq import Groq
from google import genai

from llama_index.core.llms import (
    CustomLLM,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata
)
from llama_index.core.llms.callbacks import llm_completion_callback

load_dotenv()

GROQ_API_KEY = os.getenv('GROQ_API_KEY')
POLLINATIONS_API_KEY = os.getenv('POLLINATIONS_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# Initialize GeminAI client if key present
GENAI_CLIENT = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

MODEL_DATA_FILE = "model_data.json"
POLLINATIONS_URL = "https://text.pollinations.ai/openai"

def load_model_data(file_path: str = MODEL_DATA_FILE) -> dict:
    """Load model/provider pairs from a JSON file."""
    try:
        with open(file_path, 'r') as f:
            content = json.load(f)
        # Normalize providers to lowercase
        return {model: provider.lower() for model, provider in content.get('models', {}).items()}
    except FileNotFoundError:
        print(f"[Model Data] {file_path} not found. Using empty defaults.")
    except json.JSONDecodeError as e:
        print(f"[Model Data] JSON decode error: {e}")
    except Exception as e:
        print(f"[Model Data] Unexpected error: {e}")
    return {}

def rand_model() -> Tuple[str, str]:
    """
    Randomly select a provider and its model name from model_data.json.
    Returns (provider, model_name).
    """
    models = load_model_data()
    if not models:
        print("[rand_model] No models configured; defaulting to groq llama-3.1-8b-instant")
        return 'groq', 'llama-3.1-8b-instant'

    choices = list(models.items())  # (model_name, provider)
    model_name, provider = random.choice(choices)
    print(f"[rand_model] Selected provider={provider}, model={model_name}")
    return provider, model_name

class rlm(CustomLLM):
    """Randomized LLM dispatcher across multiple providers."""
    context_window: int = 7024
    num_output: int = 7024
    provider: str = 'groq'
    model: str = 'llama-3.1-8b-instant'
    use_random: ClassVar[bool] = True

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=self.model or "random",
            provider=self.provider or "random",
            model=self.model or "random",
            use_random=self.use_random
        )

    def _call_pollinations(self, model: str, prompt: str) -> str:
        headers = {}
        if POLLINATIONS_API_KEY:
            headers['Authorization'] = f"Bearer {POLLINATIONS_API_KEY}"
        try:
            r = requests.post(POLLINATIONS_URL, headers=headers, json={
                'model': model,
                'messages': [{'role': 'user', 'content': prompt}]
            })
            r.raise_for_status()
            return r.json()['choices'][0]['message']['content']
        except Exception as e:
            print(f"[Pollinations] Request failed: {e}")
            return ''

    def _call_groq(self, model: str, prompt: str) -> str:
        try:
            client = Groq(api_key=GROQ_API_KEY)
            result = client.chat.completions.create(
                model=model,
                messages=[{'role': 'user', 'content': prompt}]
            )
            return result.choices[0].message.content
        except Exception as e:
            print(f"[Groq] API error: {e}")
            return ''

    def _call_gemini(self, model: str, prompt: str) -> str:
        if not GENAI_CLIENT:
            print("[Gemini] API key not configured.")
            return ''
        try:
            response = GENAI_CLIENT.models.generate_content(
                model=model,
                contents=[prompt]
            )
            return response.text
        except Exception as e:
            print(f"[Gemini] API error: {e}")
            return ''

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        # Optional: allow override of provider/model in kwargs
        provider = kwargs.get("provider", None)
        model = kwargs.get("model", None)
        if not provider or not model:
            provider, model = (rand_model() if self.use_random else (self.provider, self.model))

        func = {
            'pollinations': self._call_pollinations,
            'groq': self._call_groq,
            'gemini': self._call_gemini
        }.get(provider)

        if not func:
            raise ValueError(f"Unsupported provider: {provider}")

        text = func(model, prompt)
        return CompletionResponse(text=text)

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        provider = kwargs.get("provider", None)
        model = kwargs.get("model", None)
        if not provider or not model:
            provider, model = (rand_model() if self.use_random else (self.provider, self.model))

        func = {
            'pollinations': self._call_pollinations,
            'groq': self._call_groq,
            'gemini': self._call_gemini
        }.get(provider)

        if not func:
            raise ValueError(f"Unsupported provider: {provider}")

        full_text = func(model, prompt)

        buffer = ''
        for token in full_text.split():
            buffer += token + ' '
            yield CompletionResponse(text=buffer, delta=token + ' ')

def test():
    prompt = (
        "You have to craft a compelling story that blends historical facts with engaging narrative techniques. "
        "The text is meant for 5th and 6th grade school level, it needs to be simple, but it has to be also very detailed, so don't cut it short. "
        "Topic is music. Separate paragraphs by new line (important)."
    )
    llm = rlm()
    resp = llm.complete(prompt)
    print("Generated response:\n", resp.text)

if __name__ == '__main__':
    test()