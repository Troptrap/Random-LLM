import pytest
from unittest.mock import patch, MagicMock
from rlm import rlm

@pytest.fixture
def llm_instance():
    return rlm()

def test_rand_model_selection(monkeypatch):
    # Patch load_model_data to return a fixed set
    from rlm import rand_model
    monkeypatch.setattr('rlm.load_model_data', lambda: {'test-model': 'groq'})
    provider, model = rand_model()
    assert provider == 'groq'
    assert model == 'test-model'

def test_complete_groq(llm_instance):
    with patch.object(llm_instance, '_call_groq', return_value='groq response') as mock_groq:
        resp = llm_instance.complete("test prompt", provider="groq", model="llama-3.1-8b-instant")
        assert resp.text == 'groq response'
        mock_groq.assert_called_once()

def test_complete_pollinations(llm_instance):
    with patch.object(llm_instance, '_call_pollinations', return_value='pollinations response') as mock_poll:
        resp = llm_instance.complete("test prompt", provider="pollinations", model="some-model")
        assert resp.text == 'pollinations response'
        mock_poll.assert_called_once()

def test_complete_gemini(llm_instance):
    with patch.object(llm_instance, '_call_gemini', return_value='gemini response') as mock_gemini:
        resp = llm_instance.complete("test prompt", provider="gemini", model="some-model")
        assert resp.text == 'gemini response'
        mock_gemini.assert_called_once()

def test_unknown_provider(llm_instance):
    with pytest.raises(ValueError):
        llm_instance.complete("test prompt", provider="unknown", model="foo")

def test_stream_complete(llm_instance):
    with patch.object(llm_instance, '_call_groq', return_value='a b c'):
        gen = llm_instance.stream_complete("test prompt", provider="groq", model="llama-3.1-8b-instant")
        outputs = [resp.text for resp in gen]
        # Should yield incremental tokens
        assert outputs[-1].strip() == 'a b c'
