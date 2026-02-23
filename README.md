# MLops
findings - 

<img width="923" height="531" alt="image" src="https://github.com/user-attachments/assets/5c259a8e-9e8d-43c7-bc7c-2f3dcf1ff480" />

command for - 
Docker image building using the Dockerfile

docker build -t dl-project:v1 .

Docker image building using the Dockerfile

docker build -t m25csa024-project:v1 .

docker run -it --rm \
  --gpus all \
  --shm-size=8g \
  -v $(pwd):/workspace \
  m25csa024-project:v1



