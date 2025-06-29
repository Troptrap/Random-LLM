# Random-LLM: Because Choosing Just One AI is *So* Last Century

Tired of being loyal to a single LLM? Do you dream of a world where your text generation is as unpredictable as a cat chasing a laser pointer? Then you've come to the right place!

**Random-LLM (AB)Use free AI** is your one-stop-shop for AI model roulette. We've crammed a bunch of free AI models into a Python package and made it choose one at random every time you ask it to generate text. Because why settle for consistency when you can have surprise?

## Features

*   **Random Model Selection:** The core feature! Each call to `rlm.complete()` or `rlm.stream_complete()` picks a different model (if you let it!). Prepare for anything!
*   **Multiple Provider Support:** Currently supports Groq, Gemini, and Pollinations.  (Think of it as the Avengers, but with less coordinated teamwork).
*   **Easy to Use:** Just install and go! (See "Installation" below).
*   **`model_data.json` Configuration:**  Want to tweak the chaos?  `model_data.json` lets you specify which models and providers Random-LLM can choose from.  (But where's the fun in that?)
*   **(AB)Use free AI:** Yes, it uses free AI. That's the point of this repo.

## Installation

```bash
git clone https://github.com/Troptrap/Random-LLM.git
cd Random-LLM
pip install -r requirements.txt
# Set your API keys in .env file (see "Configuration" below)
python -m pip install -e .
```

## Configuration

You'll need API keys for the providers you want to use. Create a `.env` file in the root directory of the project and add your keys like this:

```
GROQ_API_KEY="YOUR_GROQ_API_KEY"
POLLINATIONS_API_KEY="YOUR_POLLINATIONS_API_KEY"
GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
```

**Warning:**  Make sure to actually *get* these API keys.  Otherwise, Random-LLM will just print error messages and sulk.

## Usage

```python
from rlm import rlm

llm = rlm()
prompt = "Write a haiku about a confused AI."
response = llm.complete(prompt)
print(response.text)
```

**Expected Output:**

```
(Possibly coherent haiku, possibly complete gibberish, possibly an apology from the AI for not understanding haikus.)
```
## Known behavior 
There is no message history, no context. If you want to send context, add it to prompt.

## Why?

Because life's too short to use the same LLM every day. Embrace the randomness! Live a little! Confuse your users!

## Contributing

Pull requests are welcome!  Especially if they add more AI providers or make the randomness even more unpredictable.  (Good luck with that.)

## License

MIT License - do whatever you want with it, I don't care.

## Disclaimer

Use at your own risk.  We are not responsible for any existential crises caused by the unpredictable output of Random-LLM.  You have been warned.
