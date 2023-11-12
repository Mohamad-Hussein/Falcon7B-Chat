from typing import List
import chromadb 
import os

from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationChain
from langchain.llms import HuggingFacePipeline
from langchain.document_loaders import PDFPlumberLoader
from langchain.text_splitter import CharacterTextSplitter, TokenTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

import torch

from transformers import pipeline, BitsAndBytesConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import StoppingCriteria, StoppingCriteriaList

# Parameters to tune
num_saved_mes = 6
max_length = 2048

# This is a quick way to stop rambling
class StopGenerationCriteria(StoppingCriteria):
    def __init__(
        self, tokens: List[List[str]], tokenizer: AutoTokenizer, device: torch.device
    ):
        stop_token_ids = [tokenizer.convert_tokens_to_ids(t) for t in tokens]
        self.stop_token_ids = [
            torch.tensor(x, dtype=torch.long, device=device) for x in stop_token_ids
        ]
 
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        for stop_ids in self.stop_token_ids:
            if torch.eq(input_ids[0][-len(stop_ids) :], stop_ids).all():
                return True
        return False
    
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

# Choose model
model_src = "tiiuae/falcon-7b-instruct"
cache_dir = "models/"

# Loading model
model_4bit = AutoModelForCausalLM.from_pretrained(
        model_src, 
        device_map="auto",
        quantization_config=quantization_config,
        cache_dir=cache_dir,

        # Change to true to use model from remote, and avoid downloading 20Gb
        trust_remote_code=True
        )

# model_4bit.to_bettertransformer()
# model_4bit = model_4bit.eval()

# Making tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_src, cache_dir=cache_dir)

# Making criteria for stopping model from rambling or imagining
stop_tokens = [["Human", ":"], ["AI", ":"], ["User", ":"]]
stopping_criteria = StoppingCriteriaList(
    [StopGenerationCriteria(stop_tokens, tokenizer, model_4bit.device)]
)

# Pipeline for models
pipe = pipeline(
        "text-generation",
        model=model_4bit,
        tokenizer=tokenizer,
        use_cache=True,
        device_map="auto",
        stopping_criteria=stopping_criteria, # Criteria
        max_length=max_length,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
)

# Wrapping pipeline
llm = HuggingFacePipeline(pipeline=pipe)

# Custom Prompt
template = """
You are a friendly AI assistant currently nicknamed "Falcon" who is helping the user accomplish his tasks,
and answers his questions informatively.

Current conversation:
{history}
Human: {input}
AI:""".strip()
 
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
EMB_SBERT_MPNET_BASE = "sentence-transformers/all-mpnet-base-v2"

def create_emb():
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return HuggingFaceEmbeddings(model_name=EMB_SBERT_MPNET_BASE, model_kwargs={"device": device})

######### Vector Data ##########
embedding = create_emb()

# Load the pdf
pdf_path = os.path.join("data", "lecture06-processes.pdf")
loader = PDFPlumberLoader(pdf_path)
documents = loader.load()

# Split documents and create text snippets
text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=10, encoding_name="cl100k_base")  # This the encoding for text-embedding-ada-002
texts = text_splitter.split_documents(texts)

persist_directory = "chroma_db"
vectordb = Chroma.from_documents(documents=texts, embedding=embedding, persist_directory=persist_directory)

print(f"\nConversation started with Falcon. Type 'quit' to stop conversation.\n")
# Entering chat with Falcon
while True:
    user_input = input("\nYou: ")
    if user_input.lower() == "quit":
        break

    response = chain(user_input)["response"]

    # Processing response to remove stop word
    response = response.replace("\nUser","").replace("\nHuman:","")
    print("Falcon: ", response)

# Extract the conversation history from the chain object
conversation_history = chain.memory.buffer

# Save the conversation history to a text file
with open("conversation_history.txt", "w") as file:
    file.writelines(conversation_history)

print("\nSession ended")