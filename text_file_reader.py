from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompt_values import PromptValue
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub
from dotenv import load_dotenv
load_dotenv()

# 1- Load the document
loader = TextLoader(r"C:\Users\T480S\Machine Learning\LangChain\script")
document = loader.load()

# 2- Split the document
splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=50)
chunks = splitter.split_documents(document)

# 3- Create embeddings
vectorstore = FAISS.from_documents(chunks, HuggingFaceEmbeddings())

# 4- Retriever
retriever = vectorstore.as_retriever()

query = input("Enter a query: ")
retrieved_docs = retriever.invoke(query)

#combine retrieved_text in a single prompt
retrieved_text = "\n".join(doc.page_content for doc in retrieved_docs)

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN
)
model = ChatHuggingFace(llm=llm)
prompt = PromptTemplate(
    template = "You have to give all answers using the provided text \n {retrieved_text} \n and question is {query} \n Answers should be within 5 lines.",
    input_variables = ["query", "retrieved_text"]
)
prompt_formatted = prompt.format(query=query, retrieved_text=retrieved_text)
answer = model.invoke(prompt_formatted)

print(answer.content)
