from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()
llm = HuggingFaceEndpoint(
    repo_id = "openai/gpt-oss-20b",
    task = 'text-generation'
)
model = ChatHuggingFace(llm=llm)
prompt = PromptTemplate(
    template = "Write 5 points on {topic}",
    input_variables = ["topic"]
)
parser = StrOutputParser()

chain = prompt | model | parser
result = chain.invoke({'topic' : 'Black Hole'})
# print(result)

chain.get_graph().print_ascii()