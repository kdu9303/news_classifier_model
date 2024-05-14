import asyncio
import os
import time
import glob
import warnings
import aiocsv
import aiofiles
import tiktoken
import traceback
import pandas as pd
from rich import print
from functools import wraps
from typing import Any, Coroutine, Generator, AsyncGenerator
from dotenv import load_dotenv

# text 전처리
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
)
from langchain_community.document_loaders.excel import UnstructuredExcelLoader
from langchain_community.document_loaders.csv_loader import CSVLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter

# embeddings
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_core.prompts import PromptTemplate

# LLM
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.chains import (
    create_retrieval_chain,
)


load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings('ignore')


def async_timer(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.perf_counter()

        # call the decorated function
        result = await func(*args, **kwargs)

        end_time = time.perf_counter()
        execution_time = end_time - start_time

        if func.__name__ == "main":
            print(f"----------------------------------------")
            print(f"---------- Main 함수 실행 --------------")
            print(f"총 실행 시간: {execution_time:.3f}초")
            print(f"----------------------------------------")
        else:
            print(f"----------------------------------------")
            print(f"실행 시간: {execution_time:.3f}초")
            print(f"----------------------------------------")
        return result

    return wrapper


class LlmArticleClassifier:
    def __init__(self, train_dir_path: str, model: str = "gpt-3.5-turbo"):
        self.train_dir_path = train_dir_path

        self.model = model

    async def async_get_input_data_from_files(
        self, input_dir_path
    ) -> AsyncGenerator[str, None]:
        for file_name in glob.glob(input_dir_path + "/*"):
            try:
                if ".xlsx" in file_name:
                    data = pd.read_excel(file_name)
                elif ".csv" in file_name:
                    data = pd.read_csv(file_name)

                for _, row in data.iterrows():
                    row_dict = row.to_dict()
                    row_str = "\n".join(f'"{k}": "{v}"' for k, v in row_dict.items())
                    yield row_str

            except Exception:
                print(traceback.format_exc())

    def sync_get_input_data_from_files(self) -> Generator[str, None, None]:
        # formatted_data_strings = []

        for file_name in glob.glob(self.input_dir_path + "/*"):
            try:
                if ".xlsx" in file_name:
                    data = pd.read_excel(file_name)
                elif ".csv" in file_name:
                    data = pd.read_csv(file_name)
                else:
                    continue

                for _, row in data.iterrows():
                    row_dict = row.to_dict()
                    row_str = "\n".join(f'"{k}": "{v}"' for k, v in row_dict.items())
                    # formatted_data_strings.append(row_str)
                    yield row_str

            except Exception as e:
                print(traceback.format_exc())

        # return formatted_data_strings

    def get_train_data_from_files(self) -> list[Document]:
        doc_list = []

        for file_name in glob.glob(self.train_dir_path + "/*"):
            print(f"Uploaded {file_name}")

            if ".pdf" in file_name:
                loader = PyPDFLoader(file_name)
                documents = loader.load_and_split()
            elif ".docx" in file_name:
                loader = Docx2txtLoader(file_name)
                documents = loader.load_and_split()
            elif ".txt" in file_name:
                loader = TextLoader(file_name)
                documents = loader.load_and_split()
            elif ".xlsx" in file_name:
                loader = UnstructuredExcelLoader(file_name, mode="elements")
                documents = loader.load_and_split()
            elif ".csv" in file_name:
                loader = CSVLoader(file_name)
                documents = loader.load_and_split()
            else:
                continue

            doc_list.extend(documents)

        return doc_list

    def tiktoken_len(self, text: str) -> int:
        tokenizer = tiktoken.get_encoding("cl100k_base")
        tokens = tokenizer.encode(text)
        return len(tokens)

    def get_text_chunks(self) -> list[Document]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024, chunk_overlap=100, length_function=self.tiktoken_len
        )
        text = self.get_train_data_from_files()
        chunks = text_splitter.split_documents(text)
        return chunks

    async def get_vectorstore(self) -> Coroutine[Any, Any, FAISS]:
        embeddings = HuggingFaceEmbeddings(
            model_name="jhgan/ko-sroberta-multitask",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        text_chunks = self.get_text_chunks()
        vectordb = await FAISS.afrom_documents(text_chunks, embeddings)
        return vectordb

    def set_templetes(self):
        document_template = f"""
            너의 역할은 주어진 뉴스 기사나 리포트를 분석하고 제공된 Output 형식의 카테고리 분류 설명에 따라 
            입력 기사를 카테고리별로 분류하는 역할이야.
            결과를 출력할때는 Markdown에서 글씨를 보기 쉽게 문장 혹은 단위별로 prettify를 적용해서 출력해줘.
            
            리포트 혹은 뉴스의 내용과 Output 형식은 백틱(```)안에 주어진다.
            예제 리포트에 포함된 내용은 결과에 Return하지 않는다.
            
            예제 리포트:
            {{context}}
            
            ```
            {{input}}
            ```
            """
        document_prompt = PromptTemplate(
            template=document_template, input_variables=["input"]
        )

        return document_prompt

    async def get_create_retrieval_chain(self, temperature: float = 0.0):
        """
        모델 선택
        "gpt-3.5-turbo",
        "gpt-4-turbo",
        "gpt-4o",
        "claude-2.1",
        "claude-3-haiku-20240307",
        """

        if "gpt" in self.model:
            llm = ChatOpenAI(
                api_key=os.getenv("open_ai_key"),
                model_name=self.model,
                temperature=temperature,
            )
        elif "claude" in self.model:
            llm = ChatAnthropic(
                api_key=os.getenv("antrophic_key"),
                model_name=self.model,
                temperature=temperature,
            )

        """
        Retriever 옵션
        as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.5})
        as_retriever(search_kwargs={"k": 3})
        as_retriever(search_type="mmr")
        """
        vectorstore = await self.get_vectorstore()
        retriever = vectorstore.as_retriever()
        document_prompt = self.set_templetes()

        combine_docs_chain = create_stuff_documents_chain(llm, document_prompt)
        retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

        return retrieval_chain

    @async_timer
    async def process_llm(self, query: str) -> None:
        chain = await self.get_create_retrieval_chain()
        result = await chain.ainvoke({"input": query})
        print(result["answer"])
        
        return result

    @async_timer
    async def process_llm_batch(self, queries: list[str]) -> list[dict[str, str]]:
        chain = await self.get_create_retrieval_chain()
        results = await chain.abatch([{"input": query} for query in queries])
        for result in results:
            # print(f"Query: {result['input']}")
            print(result['answer'])
            print()
        return results    

async def write_to_csv(results: list[dict[str, str]]):

    output_dir = "./output_data"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "results.csv")
    
    async with aiofiles.open(output_file, mode="w", encoding="utf-8-sig", newline="") as afp:
        # header (excel, unix)
        writer = aiocsv.AsyncWriter(afp, dialect="unix")
        await writer.writerow(["query", "answer"])

        # rows
        for result in results:
            await writer.writerow([result["input"], result["answer"]])

    print(f"결과가 '{output_file}'에 저장되었습니다.")


@async_timer
async def main():
    llm = LlmArticleClassifier(train_dir_path="./train_data", model="gpt-4o")

    data_generator = llm.async_get_input_data_from_files(
        input_dir_path="./input_data",
    )

    # tasks = []
    results = []
    chunk_size = 4  # 한 번에 처리할 chunk 크기
    batch_queries = []
    async for query in data_generator:
        # tasks.append(asyncio.create_task(llm.process_llm(query)))
        batch_queries.append(query)
        if len(batch_queries) == chunk_size:
            results.extend(await llm.process_llm_batch(batch_queries))
            batch_queries.clear()
        
        # if len(tasks) == chunk_size:
        #     results.extend(await asyncio.gather(*tasks))
        #     tasks.clear()

    # if tasks:
    #     results.extend(await asyncio.gather(*tasks))

    if batch_queries:
        results.extend(await llm.process_llm_batch(batch_queries))
    
    # await write_to_csv(results)

if __name__ == "__main__":
    asyncio.run(main())
