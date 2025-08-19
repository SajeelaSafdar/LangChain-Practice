from langchain_core.output_parsers import JsonOutputParser
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
parser = JsonOutputParser()
template = PromptTemplate(
    template = "Give me name, age and estimated net worth of Bollywood King.\n {format_instruction}",
    input_variables = [],
    partial_variables= {'format_instruction' : parser.get_format_instructions()}
)
# prompt = template.format()
# result = model.invoke(prompt)
# #print(type(result.content))
# final_result = parser.parse(result.content)

#shorter way
chain = template | model | parser
result = chain.invoke({})
print(result)
print(type(result))


#con -> does not enforce schema