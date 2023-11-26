# Falcon7B-Chat
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/dwyl/esta/issues)

![Python](https://img.shields.io/badge/Python-3.11-3776AB.svg?style=flat&logo=python&logoColor=white) [![pytorch](https://img.shields.io/badge/PyTorch-2.1.1-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)

This is an attempt to configure a chatbot using the falcon-7b-instruct parameter model to run locally on a machine with &lt;8Gb VRAM. Using 4 bit quantization to reduce memory load. Source to [guide](https://www.mlexpert.io/prompt-engineering/chatbot-with-local-llm-using-langchain).

## Set up
1. Download docker
2. docker build -t chatbot-image:latest -f docker/Dockerfile .
3. docker run -it --gpus all -v $(pwd):/workspace chatbot-image:latest bash
4. ...
