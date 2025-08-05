from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

model = OllamaLLM(model="llama3.2")

template = """
You are an expert in answering questions about a pizza restaurant.

Here are some relevant reviews : {reviews}

Here is the question : {question}

Please respond in under 100 words.
Only use the given reviews. Do not make up facts.
"""

prompt = ChatPromptTemplate.from_template(template)

chain = prompt | model

result = chain.invoke({
    "reviews": "The pizza at pizza1 restaurant was great! The crust was crispy and the cheese was gooey. The toppings were fresh and the sauce was delicious. I would definitely recommend this pizza to anyone. However, i would still recommend pizza2 restaurant over pizza1 restaurant.",
    "question": "What is the best pizza in town?"
})

print(result)