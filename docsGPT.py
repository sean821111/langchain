from langchain.llms import OpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

import os, json

api_key = os.getenv("OPENAI_API_KEY")


def _doc_embeddings(pdf_path):
    loader = PyPDFLoader(pdf_path)
    doc = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=1000)
    pages = splitter.split_documents(doc)
    
    embeddings = OpenAIEmbeddings()
    db = Chroma.from_documents(documents=pages, embedding= embeddings)
    return db
    
def chat_document(pdf_path):
    db = _doc_embeddings(pdf_path)
    
    prompt_template = """Use the following pieces of context to answer the question, 
                    if you don't know the answer, leave it blank don't try to make up an answer.
                {context}
                Question: {question}
                Answer in JSON representations
                """
                
                
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    chain_type_kwargs = {"prompt": PROMPT}

        
    chain_type_kwargs = {"prompt": PROMPT}
    rqa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=db.as_retriever(), chain_type_kwargs=chain_type_kwargs)   
    
    query = """ What is the display specifications?
                include: company_name, 
                product_name, size_inch,
                resolution,contrast,
                operation_temperature,
                sunlight_readable,
                antiglare,
        """
        
    res = rqa.run(query)
    return res

if __name__ == '__main__':
    example_data1 = "example_data/rxi-web-panel-emerson.pdf"
    example_data2 = "example_data/HEM-6232T Manual - Omron Healthcare.pdf"
    example_data3 = "example_data/10-04-580-SPC - Wall-Smart.pdf"
    
    result = chat_document(example_data3)
    print(result)