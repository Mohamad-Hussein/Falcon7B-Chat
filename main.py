from src.funcs import create_model, create_vectordb, create_conv_chain
from src.funcs import StopGenerationCriteria

from transformers import pipeline
from transformers import AutoTokenizer
from transformers import StoppingCriteriaList

# Parameters to tune
NUM_SAVED_MESSAGES = 6
MAX_LENGTH = 2048


# Choose model
model_src = "tiiuae/falcon-7b-instruct"
cache_dir = "models/"

# Loading model
model_4bit = create_model(model_src, cache_dir)

# Making tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_src, cache_dir=cache_dir)

# Making criteria for stopping model from rambling or imagining
stop_tokens = [["Human", ":"], ["AI", ":"], ["User", ":"]]
stopping_criteria = StoppingCriteriaList(
    [StopGenerationCriteria(stop_tokens, tokenizer, model_4bit.device)]
)

# Custom Prompt
template = """
You are a friendly AI assistant currently nicknamed "Falcon" who is helping the user accomplish his tasks,
and answers his questions informatively.

Current conversation:
{history}
Human: {input}
AI:""".strip()

# Pipeline for models
pipe = pipeline(
        "text-generation",
        model=model_4bit,
        tokenizer=tokenizer,
        use_cache=True,
        device_map="auto",
        stopping_criteria=stopping_criteria, # Criteria
        max_length=MAX_LENGTH,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
)

# Making conversation chain
chain = create_conv_chain(
    template=template,
    num_saved_mes=NUM_SAVED_MESSAGES,
    pipe=pipe
)

# Making Vector Database
vectordb = create_vectordb()

print(f"\nConversation started with Falcon. Type 'quit' to stop conversation.\n")

# Entering chat with Falcon
while True:
    user_input = input("\nYou: ")
    if user_input.lower() == "quit":
        break
    
    db_search = vectordb.similarity_search_with_relevance_scores(user_input)
    search = db_search[0][0]
    score = db_search[0][1]
    
    print(f"This is the doc: {search}\n This is the relevance score {score}")

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