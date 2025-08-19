from langchain_core.output_parsers import PydanticOutputParser
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

load_dotenv()
llm = HuggingFaceEndpoint(
    # repo_id = "Qwen/Qwen3-4B-Instruct-2507",
    repo_id = "Qwen/Qwen3-4B-Instruct-2507",
    task = 'text-generation',
    temperature= 1.5
)
model = ChatHuggingFace(llm=llm)

class Person(BaseModel):
    name: str = Field(..., description="The name of the person.")
    age: int = Field(gt=18, description="The age of the person.")
    net_worth: float = Field(..., description="The estimated net worth of the person.")

parser = PydanticOutputParser(pydantic_object=Person)
template = PromptTemplate(
    template = "Give me name, age and estimated net worth of {place} person.\n {format_instruction}",
    input_variables = ['place'],
    partial_variables= {'format_instruction' : parser.get_format_instructions()}
)

chain = template | model | parser
result = chain.invoke({'place':'Pakistani'})
print(result)


