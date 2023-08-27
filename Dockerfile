# Build: sudo docker build -t <project_name> .
# Run: sudo docker run -v $(pwd):/workspaces/project --gpus all -it --rm <project_name>

ARG PYTORCH="1.11.0"
ARG CUDA="11.5"
ARG CUDNN="8"
ARG UBUNTU="22.04"

FROM ghcr.io/pytorch/pytorch-nightly

RUN apt-get update && DEBIAN_FRONTEND=noninteractive TZ=PST apt-get install -y --no-install-recommends \
        less sudo rsync graphviz git dnsutils build-essential libsndfile1 && \
        python3 -m pip install --upgrade pip

RUN useradd -d /root -ms /bin/bash admin && \
  adduser admin sudo && \
  echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
  chown -R admin:admin /root

USER admin

ARG DEBIAN_FRONTEND="noninteractive"
ADD ./requirements.txt .
RUN python3 -m pip install -r requirements.txt

ADD . /root
WORKDIR /root

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9000"]
