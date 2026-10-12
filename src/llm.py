"""Ask the LLM to answer using ONLY the retrieved chunks, with citations.

This is the "augmented generation" half of RAG: we paste the relevant chunks
into the prompt so the model answers from your documents, not from memory.
"""

from dataclasses import dataclass

from openai import OpenAI, OpenAIError

import config
from retrieval import SearchResult

# The model must start its reply with this exact token when the sources are
# insufficient, so our code can detect it reliably.
NOT_ENOUGH_INFO = "NOT_ENOUGH_INFO"

SYSTEM_PROMPT = f"""You are a careful research assistant.
Answer the user's question using ONLY the numbered sources provided.
Rules:
- Cite sources inline with their numbers in square brackets, e.g. [1] or [2][3].
- Do not use outside knowledge. Do not guess.
- If the sources do not contain enough information to answer, reply with
  {NOT_ENOUGH_INFO} on the first line, then one sentence explaining what is missing.
- If the sources answer only part of the question, answer that part and say clearly
  what the sources do not cover."""


@dataclass
class Answer:
    text: str
    enough_info: bool


def is_configured() -> bool:
    return bool(config.OPENAI_API_KEY)


def build_user_prompt(question: str, results: list[SearchResult]) -> str:
    lines = ["Sources:"]
    for n, r in enumerate(results, start=1):
        c = r.chunk
        lines.append(f"[{n}] ({c.doc_name}, page {c.page_number})\n{c.text}\n")
    lines.append(f"Question: {question}")
    return "\n".join(lines)


def generate_answer(question: str, results: list[SearchResult]) -> Answer:
    client = OpenAI(api_key=config.OPENAI_API_KEY)
    try:
        response = client.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(question, results)},
            ],
            max_completion_tokens=800,
        )
    except OpenAIError as e:
        raise RuntimeError(f"OpenAI API call failed: {e}") from e

    text = (response.choices[0].message.content or "").strip()
    if text.startswith(NOT_ENOUGH_INFO):
        explanation = text[len(NOT_ENOUGH_INFO) :].strip()
        return Answer(text=explanation, enough_info=False)
    return Answer(text=text, enough_info=True)
