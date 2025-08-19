from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnableBranch
from pydantic import BaseModel, Field
from typing import Literal


load_dotenv()
llm = HuggingFaceEndpoint(
    repo_id = "Qwen/Qwen3-Coder-30B-A3B-Instruct",
    task = 'text-generation'
)
model = ChatHuggingFace(llm=llm)

class Feedback(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(..., description="The sentiment of the feedback.")

parser2 = PydanticOutputParser(pydantic_object=Feedback)
parser = StrOutputParser()

prompt1 = PromptTemplate(
    template = "Classify the following feedback into positive or negative. Feedback is: \n {feedback} \n {format_instruction}",
    input_variables = ["feedback"],
    partial_variables= {'format_instruction' : parser2.get_format_instructions()}
)
prompt2 = PromptTemplate(
    template = "Say thank you on the positive feedback \n {feedback}",
    input_variables = ['feedback']
)
prompt3 = PromptTemplate(
    template = "Say Sorry on the negative feedback \n {feedback}",
    input_variables = ['feedback']
)

classifier = prompt1 | model | parser2
branch_chain = RunnableBranch(
    (lambda x: x.sentiment == "positive", prompt2 | model | parser),
    (lambda x: x.sentiment == "negative", prompt3 | model | parser),
    RunnableLambda(lambda x: "No sentiment detected")
)
chain = classifier | branch_chain
print(chain.invoke({'feedback':"I love this phone"}))
chain.get_graph().print_ascii()