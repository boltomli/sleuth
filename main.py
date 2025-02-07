import asyncio

import chromadb
from llama_index.core import (SimpleDirectoryReader, StorageContext,
                              VectorStoreIndex)
from llama_index.core.postprocessor.llm_rerank import LLMRerank
from llama_index.core.response_synthesizers import CompactAndRefine
from llama_index.core.schema import NodeWithScore
from llama_index.core.workflow import (Context, Event, StartEvent, StopEvent,
                                       Workflow, step)
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore
from loguru import logger

embed_model = OllamaEmbedding(model_name="qwen2.5:7b")
llm_model = Ollama(model="qwen2.5:7b", request_timeout=1000.0)
rerank_model = Ollama(model="deepseek-r1:8b", request_timeout=1000.0)


class RetrieverEvent(Event):
    """Result of running retrieval"""

    nodes: list[NodeWithScore]


class RerankEvent(Event):
    """Result of running reranking on retrieved nodes"""

    nodes: list[NodeWithScore]


class RAGWorkflow(Workflow):
    @step
    async def ingest(self, ctx: Context, ev: StartEvent) -> StopEvent | None:
        """Entry point to ingest a document, triggered by a StartEvent with `dirname`."""
        dirname = ev.get("dirname")
        if not dirname:
            return None

        documents = SimpleDirectoryReader(dirname).load_data()
        db = chromadb.PersistentClient(path="./db/chroma_db")
        col = f"roottree_{len(documents)}"
        chroma_collection = db.get_or_create_collection(col)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

        if chroma_collection.count() == 0:
            logger.info("save to disk")
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            index = VectorStoreIndex.from_documents(
                documents, storage_context=storage_context, embed_model=embed_model
            )
        else:
            logger.info("load from disk")
            index = VectorStoreIndex.from_vector_store(
                vector_store,
                embed_model=embed_model,
            )

        return StopEvent(result=index)

    @step
    async def retrieve(self, ctx: Context, ev: StartEvent) -> RetrieverEvent | None:
        "Entry point for RAG, triggered by a StartEvent with `query`."
        query = ev.get("query")
        index = ev.get("index")

        if not query:
            return None

        logger.info(f"Query the database with: {query}")

        # store the query in the global context
        await ctx.set("query", query)

        # get the index from the global context
        if index is None:
            logger.info("Index is empty, load some documents before querying!")
            return None

        retriever = index.as_retriever(similarity_top_k=10)
        nodes = await retriever.aretrieve(query)
        logger.info(f"Retrieved {len(nodes)} nodes.")
        return RetrieverEvent(nodes=nodes)

    @step
    async def rerank(self, ctx: Context, ev: RetrieverEvent) -> RerankEvent:
        # Rerank the nodes
        ranker = LLMRerank(choice_batch_size=5, top_n=3, llm=rerank_model)
        logger.info(await ctx.get("query", default=None), flush=True)
        new_nodes = ranker.postprocess_nodes(
            ev.nodes, query_str=await ctx.get("query", default=None)
        )
        logger.info(f"Reranked nodes to {len(new_nodes)}")
        return RerankEvent(nodes=new_nodes)

    @step
    async def synthesize(self, ctx: Context, ev: RerankEvent) -> StopEvent:
        """Return a streaming response using reranked nodes."""
        summarizer = CompactAndRefine(llm=llm_model, streaming=True, verbose=True)
        query = await ctx.get("query", default=None)

        response = await summarizer.asynthesize(query, nodes=ev.nodes)
        return StopEvent(result=response)


w = RAGWorkflow(timeout=100.0)


async def run(dirname, query):
    index = await w.run(dirname=dirname)
    result = await w.run(query=query, index=index)
    async for chunk in result.async_response_gen():
        print(chunk, end="", flush=True)


if __name__ == "__main__":
    src = "roottree"
    query = "给出Roottree家族的家族树"
    asyncio.run(run(src, query))
