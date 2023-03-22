# Optimus
ML from scratch in python and CUDA for educational purposes.

## Setup Dev Env

We use docker to setup and use dev environment. Please make sure you have docker CLI, engine and compose installed. You can refer this [link](https://docs.docker.com/get-docker/) to install everything you need. 

Build optimus dev image using 
```
docker compose build
```
Then start the docker container in interactive mode with bash
```
sudo nvidia-docker run --cap-add=SYS_ADMIN -v ~/.ssh:/root/.ssh:Z -v ~/optimus/:/workspace/host/optimus --ipc=host -ti optimus:dev bash
```