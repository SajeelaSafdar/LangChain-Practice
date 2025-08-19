import langchain
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = HuggingFaceEndpoint(
    # repo_id = "Qwen/Qwen3-4B-Instruct-2507",
    repo_id = "openai/gpt-oss-20b",  #tiny llama
    task = 'text-generation',
    temperature= 0.7,
    max_new_tokens=10
)
model = ChatHuggingFace(llm=llm)
template1 = PromptTemplate(
    template = "Write a detailed report on {topic}",
    input_variables = ["topic"]
)
template2 = PromptTemplate(
    template = "Write a brief summary in paragraph form of the following text.\n {text}",
    input_variables = ["text"]
)
prompt1 = template1.invoke({'topic':'Black Hole'})
result = model.invoke(prompt1)

prompt2 = template2.invoke({'text':result.content})
summary = model.invoke(prompt2)

print(summary.content)