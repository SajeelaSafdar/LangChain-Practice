from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnableParallel

load_dotenv()
llm = HuggingFaceEndpoint(
    repo_id = "openai/gpt-oss-20b",
    task = 'text-generation'
)
llm2 = HuggingFaceEndpoint(
    repo_id = "Qwen/Qwen3-Coder-30B-A3B-Instruct",
    task = 'text-generation'
)
model1 = ChatHuggingFace(llm=llm)
model2 = ChatHuggingFace(llm=llm2)

prompt1 = PromptTemplate(
    template = "Make brief notes on the following {topic}",
    input_variables = ["topic"]
)
prompt2 = PromptTemplate(
    template = "Make 5 fill in the blanks in the following {topic}",
    input_variables = ["topic"]
)
prompt3 = PromptTemplate(
    template = "Combine the following notes {notes} and fill in the blanks {blanks} and return them in a good format",
    input_variables = ['notes', 'blanks']
)
parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes' : prompt1 | model1 | parser,
    'blanks' : prompt2 | model2 | parser
})
merged_chain = prompt3 | model1 | parser
chain = parallel_chain | merged_chain
result = chain.invoke({'topic':'Linear Regression'})
# print(result)
chain.get_graph().print_ascii()