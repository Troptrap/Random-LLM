import os
import re
import json
import time
import requests
from datetime import datetime, timedelta, UTC
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from google import genai
from tqdm import tqdm

load_dotenv()

# Config
OUTPUT_FILE = "model_data.json"
POLLINATIONS_API_KEY = os.environ.get("POLLINATIONS_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GENAI_CLIENT = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

POLLINATIONS_URL = "https://text.pollinations.ai/models"
POLLINATIONS_CHAT_URL = "https://text.pollinations.ai/openai"

GROQ_MODELS_URL = "https://api.groq.com/openai/v1/models"
GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_HEADERS = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json"
}

RPD_THRESHOLD = 8000
CONTEXT_WINDOW_THRESHOLD = 8000

def is_cache_fresh(file_path, max_age_hours=24):
    """
    Check if the cache file at file_path is fresh (not older than max_age_hours).
    
    Args:
        file_path (str): Path to the cache file.
        max_age_hours (int): Maximum age in hours for the cache to be considered fresh.
        
    Returns:
        bool: True if fresh, False otherwise.
    """
    if not os.path.exists(file_path):
        return False
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
            timestamp = data.get("timestamp")
            if not timestamp:
                return False
            saved_time = datetime.fromisoformat(timestamp)
            return datetime.now(UTC) - saved_time < timedelta(hours=max_age_hours)
    except Exception as e:
        print(f"[Cache] Error reading cache: {e}")
        return False

def get_pollinations_models():
    """
    Fetch and validate available Pollinations models for text-to-text chat.
    
    Returns:
        dict: Dictionary mapping valid model names to the string "pollinations".
    """
    headers = {}
    if POLLINATIONS_API_KEY:
        headers["Authorization"] = f"Bearer {POLLINATIONS_API_KEY}"

    try:
        response = requests.get(POLLINATIONS_URL, headers=headers, timeout=10)
        response.raise_for_status()
        models = response.json()

        valid_models = {}
        for model in tqdm(models, desc="Validating Pollinations models"):
            if "text" in model.get("input_modalities", []) and "text" in model.get("output_modalities", []):
                model_name = model["name"]
                try:
                    r = requests.post(
                        POLLINATIONS_CHAT_URL,
                        headers=headers,
                        json={
                            "model": model_name,
                            "messages": [{"role": "user", "content": "Hello"}]
                        },
                        timeout=10
                    )
                    if r.status_code == 200:
                        data = r.json()
                        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                        if content.strip():
                            valid_models[model_name] = "pollinations"
                except Exception as e:
                    print(f"[Pollinations] Model '{model_name}' failed: {e}")

        return valid_models

    except Exception as e:
        print(f"[Pollinations] Error fetching models: {e}")
        return {}

def get_groq_models():
    """
    Fetch and filter Groq models based on context window and rate limits.
    
    Returns:
        dict: Dictionary mapping valid model IDs to the string "groq".
    """
    try:
        response = requests.get(GROQ_MODELS_URL, headers=GROQ_HEADERS)
        response.raise_for_status()
        models = response.json().get("data", [])
    except Exception as e:
        print(f"[Groq] Error fetching models: {e}")
        return {}

    valid_models = {}
    for model in tqdm(models, desc="Filtering Groq models"):
        if model.get("context_window", 0) <= CONTEXT_WINDOW_THRESHOLD or "guard" in model["id"].lower():
            continue
        try:
            payload = {
                "model": model["id"],
                "messages": [{"role": "user", "content": "Test"}],
                "max_tokens": 10
            }
            resp = requests.post(GROQ_CHAT_URL, headers=GROQ_HEADERS, json=payload)
            resp.raise_for_status()
            rpd = int(resp.headers.get("x-ratelimit-limit-requests", 0))
            if rpd > RPD_THRESHOLD:
                valid_models[model["id"]] = "groq"
        except Exception as e:
            continue
        time.sleep(2)
    return valid_models

def parse_limit_value(val):
    """
    Parse a rate limit value from the Gemini documentation table.
    
    Args:
        val (str): The value string to parse.
        
    Returns:
        int: Parsed integer value, or 0 if not available.
    """
    if val == "--" or not val:
        return 0
    match = re.search(r"[\d,]+", val.replace("\n", " "))
    return int(match.group(0).replace(",", "")) if match else None

def fetch_free_tier_limits():
    """
    Scrape Gemini API free tier limits from the official documentation.
    
    Returns:
        dict: Mapping of model documentation names to their rate limits.
    """
    url = "https://ai.google.dev/gemini-api/docs/rate-limits"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    table = soup.find("h3", id="free-tier").find_next("table")
    rows = table.find("tbody").find_all("tr")
    limits = {}
    for row in rows:
        cols = [td.get_text(strip=True).replace("\u202f", " ") for td in row.find_all("td")]
        if len(cols) < 4:
            continue
        name, rpm, tpm, rpd = cols
        limits[name] = {
            "rpm": parse_limit_value(rpm),
            "tpm": parse_limit_value(tpm),
            "rpd": parse_limit_value(rpd)
        }
    return limits

def fetch_model_ids():
    """
    Fetch the list of available Gemini API model IDs.
    
    Returns:
        list: List of Gemini API model IDs.
    """
    return [m.name for m in GENAI_CLIENT.models.list()]

def map_with_llm(scraped_names, model_ids, llm_model="gemini-2.0-flash"):
    """
    Use an LLM to map documentation model names to Gemini API model IDs.
    
    Args:
        scraped_names (list): List of names from documentation.
        model_ids (list): List of available API model IDs.
        llm_model (str): The LLM model ID to use for mapping.
        
    Returns:
        dict: Mapping from documentation names to API model IDs.
    """
    system = "You are an assistant mapping documentation model names to API model IDs. Output only JSON: key = doc name, value = API model ID."
    user = (
        "Scraped names:\n" + json.dumps(scraped_names, indent=2) +
        "\n\nModel IDs:\n" + json.dumps(model_ids, indent=2) +
        "\n\nReturn a JSON mapping."
    )
    lm = genai.Client(api_key=GEMINI_API_KEY).models.generate_content(
        model=llm_model,
        contents=[system, user],
        config=genai.types.GenerateContentConfig(response_mime_type="application/json")
    )
    return json.loads(lm.text)[0]

def get_gemini_models(rpd_threshold=200):
    """
    Collect eligible Gemini models above a specified rate limit threshold.
    
    Args:
        rpd_threshold (int): Minimum requests per day allowed for inclusion.
        
    Returns:
        dict: Dictionary mapping eligible model IDs to the string "gemini".
    """
    if not GENAI_CLIENT:
        print("[Gemini] GEMINI_API_KEY not set.")
        return {}
    try:
        scraped = fetch_free_tier_limits()
        ids = fetch_model_ids()
        mapping = map_with_llm(list(scraped.keys()), ids)
        full = {
            mapping[name]: scraped[name]
            for name in mapping
            if name in scraped
        }
        high = {
            mid: "gemini"
            for mid, info in tqdm(full.items(), desc="Filtering Gemini models")
            if info.get("rpd", 0) > rpd_threshold and info.get("rpm", 0) > 10
        }
        return high
    except Exception as e:
        print(f"[Gemini] Error: {e}")
        return {}

def collect_all_models():
    """
    Collect all models from Pollinations, Groq, and Gemini, and combine them.
    
    Returns:
        dict: Dictionary with a timestamp and model source mappings.
    """
    pollinations = get_pollinations_models()
    groq = get_groq_models()
    gemini = get_gemini_models()

    all_models = {**pollinations, **groq, **gemini}
    with tqdm(total=len(all_models), desc="Total models") as pbar:
        for _ in all_models:
            pbar.update(1)

    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "models": all_models
    }

if __name__ == "__main__":
    """
    Main entry point: Checks cache freshness and updates model data if needed.
    """
    if is_cache_fresh(OUTPUT_FILE):
        print("Everything up to date.")
        exit(0)

    result = collect_all_models()

    with open(OUTPUT_FILE, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n✅ Model data updated and written to {OUTPUT_FILE}")
    
