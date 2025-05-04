ARG VERSION_UBUNTU_CONTAINER="24.04" \
    VERSION_CUDA_CONTAINER=12-6 \
    VERSION_CUDNN_CONTAINER=9-cuda-12 \
    VERSION_PYTHON_CONTAINER="3.12" \
    URL_KEYRING_NVIDIA_CONTAINER

FROM ubuntu:"$VERSION_UBUNTU_CONTAINER" AS base-amd64



# ========================================
FROM "base-$TARGETARCH" AS base

ARG VERSION_CUDA_CONTAINER \
    VERSION_CUDNN_CONTAINER \
    VERSION_PYTHON_CONTAINER \
    URL_KEYRING_NVIDIA_CONTAINER

SHELL ["/bin/bash", "-c"]

ARG DEBIAN_FRONTEND=noninteractive
ENV DEBCONF_NOWARNINGS=yes

# Install essentials
RUN apt-get update --quiet && \
    apt-get install --quiet --assume-yes --no-install-recommends \
    apt-utils \
    build-essential \
    ca-certificates \
    curl \
    git \
    htop \
    lsb-release \
    nano \
    software-properties-common \
    tmux \
    unzip \
    wget \
    && \
    apt-get autoremove --quiet --assume-yes && \
    apt-get clean --quiet && \
    rm /var/lib/apt/lists/* --recursive --force



# ========================================
FROM base AS system-amd64

ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=all

# Install cuda
RUN wget "$URL_KEYRING_NVIDIA_CONTAINER" --quiet --directory-prefix /tmp/ && \
    dpkg --install /tmp/cuda-keyring_1.1-1_all.deb && \
    rm /tmp/cuda*.deb
RUN apt-get update --quiet && \
    apt-get install --quiet --assume-yes --no-install-recommends \
    "cuda-toolkit-$VERSION_CUDA_CONTAINER" \
    && \
    apt-get autoremove --quiet --assume-yes && \
    apt-get clean --quiet && \
    rm /var/lib/apt/lists/* --recursive --force
ENV PATH="/usr/local/cuda/bin:$PATH"
RUN ldconfig

# Install cudnn
RUN apt-get update --quiet && \
    apt-get install --quiet --assume-yes --no-install-recommends \
    "cudnn$VERSION_CUDNN_CONTAINER" \
    && \
    apt-get autoremove --quiet --assume-yes && \
    apt-get clean --quiet && \
    rm /var/lib/apt/lists/* --recursive --force



# ========================================
FROM "system-$TARGETARCH" AS python-base

# Install python
RUN apt-get update --quiet && \
    apt-get install --quiet --assume-yes --no-install-recommends \
    python-is-python3 \
    python3 \
    python3-pip \
    python3-venv \
    && \
    apt-get autoremove --quiet --assume-yes && \
    apt-get clean --quiet && \
    rm /var/lib/apt/lists/* --recursive --force

# Install python version explicitly
RUN add-apt-repository --quiet --assume-yes ppa:deadsnakes/ppa
RUN apt-get update --quiet && \
    apt-get install --quiet --assume-yes --no-install-recommends \
    "python$VERSION_PYTHON_CONTAINER" \
    "python$VERSION_PYTHON_CONTAINER-venv" \
    && \
    apt-get autoremove --quiet --assume-yes && \
    apt-get clean --quiet && \
    rm /var/lib/apt/lists/* --recursive --force

ENV PYTHONUNBUFFERED=1 \
    PYTHONWARNINGS=ignore:::setuptools.command.install,ignore:::setuptools.command.easy_install,ignore:::setuptools.command.develop,ignore:::setuptools.dist,ignore:::pkg_resources \
    PIP_ROOT_USER_ACTION=ignore \
    PIP_DISABLE_PIP_VERSION_CHECK=on

# Setup venv
RUN eval "python$VERSION_PYTHON_CONTAINER" -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/opt/venv/lib/python$VERSION_PYTHON_CONTAINER/site-packages:$PYTHONPATH"
RUN pip install --no-cache-dir \
    pip==24.2 \
    setuptools==74.0.0 \
    wheel==0.44.0



# ========================================
FROM python-base AS python-amd64

# Install dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --requirement /tmp/requirements.txt && \
    rm -rf /tmp/*



# ========================================
FROM "python-$TARGETARCH" AS pytorch-project

# Add code
COPY . /root/repos/pytorch-project-template

WORKDIR /root/repos/pytorch-project-template
ENTRYPOINT []
CMD ["bash"]
