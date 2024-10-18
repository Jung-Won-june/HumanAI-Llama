!nvidia-smi
!pip install langchain openai chromadb langchainhub tiktoken pypdf
!pip install langchain_huggingface
!pip install accelerate
!pip install --upgrade transformers
!pip install langchain-community

from langchain.document_loaders import PyPDFLoader
from google.colab import files

# Upload multiple PDF files
uploaded = files.upload()

# Initialize an empty list to hold all the pages
all_pages = []

# Iterate over the uploaded files
for file_name in uploaded.keys():
    file_path = "/content/" + file_name
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()
    all_pages.extend(pages)  # Append the pages to the all_pages list

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0)
splits = text_splitter.split_documents(pages)

from langchain.embeddings import HuggingFaceEmbeddings
!pip install sentence-transformers

model_name = "jhgan/ko-sroberta-multitask"
model_kwargs = {'device' : 'cpu'}
encode_kwargs = {'normalize_embeddings' : False}
embedding_model = HuggingFaceEmbeddings(
    model_name = model_name,
    model_kwargs = model_kwargs,
    encode_kwargs = encode_kwargs
)

from langchain.vectorstores import Chroma
vectorstore = Chroma.from_documents(documents=splits, embedding=embedding_model)
retriever = vectorstore.as_retriever(search_kwargs={"k" : 10})

import os
os.environ['HF_TOKEN']="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"

#llama컨텐츠를 colab환경에 다운받는다.(미)
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

#불러온 tokenizer, model을 langchain과 연동하기 위해 HuggingFacePipeline이용해 text-generation으로 묶어준다.
from langchain_huggingface.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=100)
hf = HuggingFacePipeline(pipeline=pipe)

#프롬프트 설정
from langchain.prompts import PromptTemplate

template = """다음과 같은 맥락을 사용하여 마지막 질문에 대답하십시오.
{context}
질문: {question}
상세하고 풍부한 답변을 제공해 주십시오:"""
#필요한 부분은 답변 부분이니까 템플릿을 나눈 후 스플릿하기 위해 parser에서 다룬다.
rag_prompt_custom = PromptTemplate.from_template(template)

#output Parser설정
from langchain.schema import BaseOutputParser

class CustomOutputParser(BaseOutputParser):
  def parse(self, text: str) -> str:
    if not text:
      raise ValueError("입력된 텍스트가 없습니다.")
    #도움이 되는 답변: 이후의 텍스트를 추출
    split_text = text.split('상세하고 풍부한 답변을 제공해 주십시오:', 1)
    if len(split_text) >1:
      return ' '.join(split_text[1:]).strip()
    else:
      return text #도움이 되는 답변이 없다면 원본 텍스트 반환.

output_parser = CustomOutputParser()

from langchain.schema.runnable import RunnablePassthrough
rag_chain = {"context": retriever, "question": RunnablePassthrough()} | rag_prompt_custom | hf | output_parser

import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt')

def post_process_output(output: str) -> str:
    sentences = sent_tokenize(output)
    return ' '.join(sentences)

# RAG 체인 결과 후처리
response = rag_chain.invoke("이걸 더 자세하게 한글로 설명해줄 수 있어?")
processed_response = post_process_output(response)
print(processed_response)
