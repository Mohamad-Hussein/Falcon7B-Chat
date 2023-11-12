FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu18.04

WORKDIR /home/chatbot
ADD main.py /home/chatbot
ADD env.yml /home/chatbot

RUN apt-get update && apt-get install -y wget

# Install conda
RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O Miniconda.sh && \
	/bin/bash Miniconda.sh -b -p /opt/conda && \
	rm Miniconda.sh

ENV PATH /opt/conda/bin:$PATH

RUN apt-get update && apt-get -y install gcc 

# Installing require packages in conda
RUN conda env create -f env.yml

# Download docker gpu compatibility packages
RUN apt-get install -y ca-certificates curl
RUN distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
RUN curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
 tee /etc/apt/sources.list.d/nvidia-container-toolkit.list \
  && apt-get update


SHELL ["conda", "run", "-n", "ChatBot-env", "/bin/bash", "-c"]

RUN echo "Make sure PyTorch is installed"
RUN python -c "import torch"

EXPOSE 5000


# Uncomment to run the file
# ENTRYPOINT ["conda", "run", "-n", "ChatBot-env", "python", "main.py"]

# I also downloaded nvidia-container-toolkit with

# sudo apt-get install -y nvidia-container-toolkit
# sudo nvidia-ctk runtime configure --runtime=docker
# sudo systemctl restart docker