import asyncio
import os
import re
from collections.abc import Iterator, Sequence
from typing import TypeVar

import certifi
import requests
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyExtract
from langchain_text_splitters import RecursiveCharacterTextSplitter

from logger import log_error, log_header, log_info, log_success, log_warning


LLMS_TXT = "https://docs.langchain.com/llms.txt"
INDEX_NAME = "rag-langchain-documentation-retrieval"

URL_BATCH_SIZE = 20
INDEX_BATCH_SIZE = 500
TEXT_CHUNK_SIZE = 4000
TEXT_CHUNK_OVERLAP = 200

T = TypeVar("T")


def batched(items: Sequence[T], size: int) -> Iterator[Sequence[T]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def get_text(url: str) -> str:
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.text


def get_documentation_urls() -> list[str]:
    main_index = get_text(LLMS_TXT)
    python_index = re.search(
        r"\[[^\]\n]*Build / Python[^\]\n]*\]\((https://[^\s)]+\.txt)\)",
        main_index,
    )
    if python_index is None:
        raise ValueError("Could not find the Build / Python link in llms.txt")

    python_index_text = get_text(python_index.group(1))
    urls = re.findall(
        r"\((https://docs\.langchain\.com/[^\s)]+\.md)\)",
        python_index_text,
    )
    return [url for url in urls if "/oss/python/" in url]


async def extract_batch(
    extractor: TavilyExtract, urls: Sequence[str], batch_number: int
) -> list[Document]:
    log_info(f"Tavily: Batch {batch_number} — requesting {len(urls)} pages")
    response = await extractor.ainvoke({"urls": list(urls)})

    documents: list[Document] = []
    for result in response.get("results", []):
        url = result.get("url")
        content = result.get("raw_content")
        if not url or not content:
            log_warning(f"Tavily: Skipping incomplete result for {url}")
            continue

        documents.append(
            Document(page_content=content, metadata={"source": url})
        )

    log_info(f"Tavily: Batch {batch_number} — extracted {len(documents)} pages")
    return documents


async def extract_documents(urls: list[str]) -> list[Document]:
    extractor = TavilyExtract()
    results = await asyncio.gather(
        *(
            extract_batch(extractor, batch, number)
            for number, batch in enumerate(batched(urls, URL_BATCH_SIZE), start=1)
        ),
        return_exceptions=True,
    )

    documents: list[Document] = []
    for number, result in enumerate(results, start=1):
        if isinstance(result, BaseException):
            if not isinstance(result, Exception):
                raise result
            log_error(f"Tavily: Batch {number} failed — {result}")
            continue

        documents.extend(result)

    return documents


def index_documents(chunks: list[Document]) -> int:
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        show_progress_bar=True,
        chunk_size=50,
        retry_min_seconds=10,
    )
    vector_store = PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=embeddings,
    )

    indexed = 0
    for number, batch in enumerate(batched(chunks, INDEX_BATCH_SIZE), start=1):
        try:
            vector_store.add_documents(list(batch))
        except Exception as error:
            log_error(f"Pinecone: Batch {number} failed — {error}")
            continue

        indexed += len(batch)
        log_success(f"Pinecone: Batch {number} — indexed {len(batch)} chunks")

    return indexed


def main() -> None:
    load_dotenv()
    ca_bundle = certifi.where()
    os.environ["SSL_CERT_FILE"] = ca_bundle
    os.environ["REQUEST_CA_BUNDLE"] = ca_bundle

    log_header("DOCUMENTATION INGESTION PIPELINE")
    urls = get_documentation_urls()
    log_info(f"Found {len(urls)} documentation URLs")
    if not urls:
        log_warning("No matching documentation URLs found. Nothing to index.")
        return

    log_header("DOCUMENT EXTRACTION")
    documents = asyncio.run(extract_documents(urls))
    log_info(f"Extracted {len(documents)}/{len(urls)} pages")
    if not documents:
        log_warning("No pages extracted. Nothing to index.")
        return

    log_header("DOCUMENT CHUNKING")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=TEXT_CHUNK_SIZE,
        chunk_overlap=TEXT_CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)
    log_info(f"Created {len(chunks)} text chunks")
    if not chunks:
        log_warning("No text chunks created. Nothing to index.")
        return

    log_header("VECTOR STORAGE")
    indexed = index_documents(chunks)

    summary = (
        f"Extracted {len(documents)}/{len(urls)} pages; "
        f"indexed {indexed}/{len(chunks)} chunks in successful batches."
    )
    if len(documents) == len(urls) and indexed == len(chunks):
        log_success(summary)
    else:
        log_warning(summary)


if __name__ == "__main__":
    main()