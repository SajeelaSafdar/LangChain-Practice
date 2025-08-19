from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()
llm = HuggingFaceEndpoint(
    # repo_id = "Qwen/Qwen3-4B-Instruct-2507",
    repo_id = "Qwen/Qwen3-4B-Instruct-2507",
    task = 'text-generation',
    temperature= 1.5
)
model = ChatHuggingFace(llm=llm)
schema = [
    ResponseSchema(name="Fact 1", description="Fact 1 about the topic"),
    ResponseSchema(name="Fact 2", description="Fact 2 about the topic"),
    ResponseSchema(name="Fact 3", description="Fact 3 about the topic"),
]
parser = StructuredOutputParser.from_response_schemas(schema)
template = PromptTemplate(
    template = "Give me facts about the {topic} \n {format_instruction}",
    input_variables = ['topic'],
    partial_variables= {'format_instruction' : parser.get_format_instructions()}
)

chain = template | model | parser
result = chain.invoke({'topic':'Black Hole'})
print(result)
print(type(result))

#con -> No data validation