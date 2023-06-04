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
docker compose run optimus bash
```

## Building Optimus 

Once you start the docker container, you will be put in `/workspace/host/optimus` by default. 
Do the following to genrate the make files.

```
mkdir build && cd build 
cmake ..
```

Now build Optimus with 
```
make all
```

## Running Tests

Run any python unit test individually, for example to test the matmul op
```
python3 /workspace/host/optimus/tests/ops/test_matmul.py
```