from typing import List
import os

from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationChain
from langchain.llms import HuggingFacePipeline
from langchain.document_loaders import PDFPlumberLoader
from langchain.text_splitter import CharacterTextSplitter, TokenTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

from torch import cuda, eq, device
from torch import float16, tensor, long, LongTensor, FloatTensor

from transformers import pipeline, BitsAndBytesConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import StoppingCriteria, StoppingCriteriaList

EMB_SBERT_MPNET_BASE = "sentence-transformers/all-mpnet-base-v2"

# This is a quick way to stop rambling
class StopGenerationCriteria(StoppingCriteria):
    def __init__(
        self, tokens: List[List[str]], tokenizer: AutoTokenizer, device: device
    ):
        stop_token_ids = [tokenizer.convert_tokens_to_ids(t) for t in tokens]
        self.stop_token_ids = [
            tensor(x, dtype=long, device=device) for x in stop_token_ids
        ]
 
    def __call__(
        self, input_ids: LongTensor, scores: FloatTensor, **kwargs
    ) -> bool:
        for stop_ids in self.stop_token_ids:
            if eq(input_ids[0][-len(stop_ids) :], stop_ids).all():
                return True
        return False


def create_model(model_src : str, cache_dir : dir):

    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    # Loading model
    model_4bit = AutoModelForCausalLM.from_pretrained(
        model_src, 
        device_map="auto",
        quantization_config=quantization_config,
        cache_dir=cache_dir,

        # Change to true to use model from remote, and avoid downloading 20Gb
        trust_remote_code=True
    )

    return model_4bit

def create_conv_chain(template : str, num_saved_mes : int,pipe) -> ConversationChain:
    # Wrapping pipeline
    llm = HuggingFacePipeline(pipeline=pipe)

    
    prompt = PromptTemplate(input_variables=["history", "input"], template=template)


    # Configuring conversation chain
    memory = ConversationBufferWindowMemory(
        memory_key="history", k=num_saved_mes, return_only_outputs=True
    )

    # Making conversation chain
    chain = ConversationChain(
        llm=llm,
        memory=memory,
        prompt=prompt,
        verbose=True,
    )

    return chain

def create_emb():
    
    device = "cuda" if cuda.is_available() else "cpu"
    return HuggingFaceEmbeddings(model_name=EMB_SBERT_MPNET_BASE, model_kwargs={"device": device})

def create_vectordb() -> Chroma:
    """Creates vectordb """

    ######### Vector Data ##########
    embedding = create_emb()

    # Load the pdf
    test_dir = "data"
    test_file = "lecture06-processes.pdf"
    pdf_path = os.path.join("..", test_dir, test_file)
    print(f"This is current directory: {os.curdir()}\nThis is dir of supposed file {pdf_path}")
    loader = PDFPlumberLoader(pdf_path)
    documents = loader.load()

    # Split documents and create text snippets
    text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
     
    # This the encoding for text-embedding-ada-002
    text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=10, encoding_name="cl100k_base")
    texts = text_splitter.split_documents(texts)

    persist_directory = "chroma_db"
    vectordb = Chroma.from_documents(documents=texts, embedding=embedding, persist_directory=persist_directory)
    
    return vectordb