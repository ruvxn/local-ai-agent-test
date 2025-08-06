from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

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

while True:
    print("\n\n ------------------")
    question = input("Enter a question (Press 'q' to exit): ")
    print("\n\n")
    if question == "q":
        break
    reviews = retriever.invoke(question)
    result = chain.invoke({
    "reviews": reviews,
    "question": question,
    })

    print(result)