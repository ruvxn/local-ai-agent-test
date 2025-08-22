from __future__ import annotations

import sys
import textwrap
from typing import Any, Iterable, List

from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda


from vector import retriever



# Model config

model = OllamaLLM(
    model="llama3.2",
    temperature=0.2,     
    num_ctx=4096,        
)


# ----------------------------Prompt--------------------

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful expert on a pizza restaurant.\n"
            "Answer ONLY using the review snippets provided.\n"
            "If the reviews don't contain the answer, say you don't know.\n"
            "Keep responses under 100 words and be concise.",
        ),
        (
            "user",
            "Reviews:\n{reviews}\n\nQuestion: {question}",
        ),
    ]
)


# 
def _format_reviews(raw: Any, top_n: int = 6, max_chars_per: int = 350) -> str:
    """
    Accepts whatever the retriever returns and formats into a compact, numbered block.
    - Supports List[str] or List[Document] like objects.
    - Truncates very long snippets to keep the prompt tight.
    """
    if not raw:
        return ""

    items: List[str] = []
    # Handle List[str]
    if isinstance(raw, list) and raw and isinstance(raw[0], str):
        for i, s in enumerate(raw[:top_n], start=1):
            s = s.strip()
            if len(s) > max_chars_per:
                s = s[: max_chars_per - 1] + "…"
            items.append(f"{i}. {s}")
        return "\n".join(items)

    # Handle ListDocument-ish
    for i, d in enumerate(list(raw)[:top_n], start=1):
        text = getattr(d, "page_content", str(d)).strip()
        meta = getattr(d, "metadata", {}) or {}
        src = meta.get("source") or meta.get("file") or meta.get("url")
        if len(text) > max_chars_per:
            text = text[: max_chars_per - 1] + "…"
        if src:
            items.append(f"{i}. {text}\n   — source: {src}")
        else:
            items.append(f"{i}. {text}")
    return "\n".join(items)


def _nice_print(s: str) -> None:
    print("\n" + textwrap.fill(s.strip(), width=88) + "\n")


# Build LC graph
# 1) Take the users question
# 2) Fetch/format reviews with the retriever
# 3) Feed both into the prompt to model
prepare_inputs = RunnableParallel(
    {
        "question": RunnablePassthrough(),
        "reviews": RunnableLambda(lambda q: _format_reviews(retriever.invoke(q))),
    }
)

chain = prepare_inputs | prompt | model


# ----------------------------CLI loop--------------------

def main() -> None:
    print("\nPizza RAG assistant (q to quit)")
    while True:
        try:
            question = input("\n> Ask about the restaurant: ").strip()
            if not question or question.lower() in {"q", "quit", "exit"}:
                print("Bye!")
                return

            # Fetch formatted reviews to decide answer reuiqred
            formatted_reviews = _format_reviews(retriever.invoke(question))
            if not formatted_reviews:
                _nice_print(
                    "I couldn’t find relevant reviews for that. Try rephrasing or asking "
                    "about menu items, service, price, ambience, or specific dishes."
                )
                continue

            # Invoke full chain
            result = chain.invoke(question)

            
            if hasattr(result, "content"):
                result = result.content

            _nice_print(result)

        except KeyboardInterrupt:
            print("\nInterrupted. Bye!")
            sys.exit(0)
        except Exception as e:
            _nice_print(f"Oops—something went wrong: {e}")


if __name__ == "__main__":
    main()
