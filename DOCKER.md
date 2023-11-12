## Local
For local testing:
1. `docker build -t all-in-one -f Dockerfile_tests . && docker run --rm -it all-in-one bash`
2. wait for the container to build and run (~5 minutes)
3. Once in the container command prompt: `pytest tests/`

## Deployment
For deployment, two other images are provided:
* CPU: `docker build -t all-in-one -f Dockerfile_cpu`
* CUDA: `docker build -t all-in-one -f Dockerfile_cuda`

