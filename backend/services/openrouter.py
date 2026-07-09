import os
import requests
from typing import Optional

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "mistralai/mistral-7b-instruct"


def _call_openrouter(messages: list, max_tokens: int = 800) -> str:
    """Call OpenRouter API and return the response text."""
    if not OPENROUTER_API_KEY:
        return (
            "OpenRouter API key not found. "
            "Please add OPENROUTER_API_KEY to your environment variables (.env file). "
            "AI explanations and report generation will not be available without it."
        )

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://datastory-ai.app",
        "X-Title": "DataStory AI",
    }

    payload = {
        "model": DEFAULT_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }

    try:
        response = requests.post(OPENROUTER_URL, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
    except requests.exceptions.Timeout:
        return "Request timed out. Please try again."
    except requests.exceptions.HTTPError as e:
        return f"API error: {e.response.status_code} - {e.response.text[:200]}"
    except Exception as e:
        return f"Unexpected error calling OpenRouter: {str(e)}"


def generate_explanation(prompt: str, context: str = "") -> str:
    """Generate a plain-language explanation using OpenRouter."""
    system = (
        "You are DataStory AI, a friendly and clear data analyst assistant. "
        "Explain things in simple, beginner-friendly language. "
        "Avoid jargon. Use short paragraphs. Be encouraging and practical."
    )
    messages = [{"role": "system", "content": system}]
    if context:
        messages.append({"role": "user", "content": f"Context:\n{context}\n\n{prompt}"})
    else:
        messages.append({"role": "user", "content": prompt})
    return _call_openrouter(messages, max_tokens=600)


def generate_report_section(prompt: str, style: str = "Beginner Friendly") -> str:
    """Generate a report section using OpenRouter."""
    style_instructions = {
        "Beginner Friendly": "Write in very simple language. Explain every term. Avoid jargon.",
        "Technical": "Use proper ML terminology. Be precise and detailed.",
        "Business Summary": "Focus on business impact and actionable insights. Keep it concise.",
    }
    instruction = style_instructions.get(style, style_instructions["Beginner Friendly"])

    system = (
        f"You are DataStory AI, generating a professional data analysis report. "
        f"Style: {style}. {instruction} "
        f"Write in clear paragraphs, no bullet points unless listing items."
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    return _call_openrouter(messages, max_tokens=500)


def answer_question(question: str, context: str) -> str:
    """Answer a user question about the dataset using OpenRouter + RAG context."""
    system = (
        "You are DataStory AI, a helpful data analyst assistant. "
        "Answer questions about the user's dataset, analysis, and models clearly. "
        "Be concise, friendly, and beginner-appropriate. "
        "If you don't know something, say so honestly."
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Dataset context:\n{context}\n\nQuestion: {question}"},
    ]
    return _call_openrouter(messages, max_tokens=700)
