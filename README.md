# Falcon7B-Chat
This is an attempt to configure a chatbot using the falcon-7b-instruct parameter model to run locally on a machine with &lt;8Gb VRAM. Using 4 bit quantization to reduce memory load. Source to [guide](https://www.mlexpert.io/prompt-engineering/chatbot-with-local-llm-using-langchain).

## Set up
1. Download docker
2. docker build -t chatbot-image:latest -f docker/Dockerfile .
3. docker run -it --gpus all -v $(pwd):/workspace chatbot-image:latest bash
4. ...
