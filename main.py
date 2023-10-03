from transformers import pipeline, AutoTokenizer, FalconForQuestionAnswering, AutoModelForCausalLM
from transformers import AutoConfig
import torch

from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

# Choose model
model_src = "tiiuae/falcon-7b-instruct"
cache_dir = "models/"

model_4bit = AutoModelForCausalLM.from_pretrained(
        model_src, 
        device_map="auto",
        quantization_config=quantization_config,
        cache_dir=cache_dir
        )
model_4bit.to_bettertransformer()

tokenizer = AutoTokenizer.from_pretrained(model_src, cache_dir=cache_dir)


pipe = pipeline(
        "text-generation",
        model=model_4bit,
        tokenizer=tokenizer,
        use_cache=True,
        device_map="auto",
        max_length=296,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
)

while(1):
    user_input = input("\nType your query: ")
    if user_input.lower() == "quit": break
    response = pipe(user_input)
    print(f"Falcon: {response[0]['generated_text']}\n")

print(f"\nSession ended")