ARG UBUNTU_VERSION=20.04
ARG CUDA_VERSION=11.3.1
ARG CUDA=11.3
ARG CUDNN_VERSION=8

FROM nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_VERSION}-devel-ubuntu${UBUNTU_VERSION}
LABEL maintainer "http://medai.korea.ac.kr"

ARG PYTHON_VERSION=3.8
ARG CONDA_ENV_NAME=dremb

# Let us install tzdata painlessly
ENV DEBIAN_FRONTEND=noninteractive

# Needed for string substitution
SHELL ["/bin/bash", "-c"]
RUN echo "Acquire::Check-Valid-Until \"false\";\nAcquire::Check-Date \"false\";" | cat > /etc/apt/apt.conf.d/10no--check-valid-until
RUN echo "Preparing system..." \
  && apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    ccache \
    cmake \
    curl \
    git \
    libfreetype6-dev \
    libhdf5-serial-dev \
    libzmq3-dev \
    libjpeg-dev \
    libpng-dev \
    libsm6 \
    libxext6 \
    libxrender-dev \
    pkg-config \
    software-properties-common \
    ssh \
    sudo \
    unzip \
    vim \
    wget
RUN rm -rf /var/lib/apt/lists/*

# For CUDA profiling
ENV LD_LIBRARY_PATH /usr/local/cuda-${CUDA}/targets/x86_64-linux/lib:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 && \
    echo "/usr/local/cuda/lib64/stubs" > /etc/ld.so.conf.d/z-cuda-stubs.conf && \
    ldconfig

# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8
RUN curl -o /tmp/miniconda.sh -sSL http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -bfp /usr/local && \
    rm /tmp/miniconda.sh
RUN conda update -y conda

# For connecting via ssh
RUN echo "PasswordAuthentication yes" >> /etc/ssh/sshd_config && \
    echo "PermitEmptyPasswords yes" >> /etc/ssh/sshd_config && \
    echo "UsePAM no" >> /etc/ssh/sshd_config

# Create the conda environment
COPY environment.yaml .
RUN echo "Preparing Dr.Emb Appyter..." && \
  conda config --set ssl_verify false && \
  conda env create -f environment.yaml

ENV PATH /usr/local/envs/$CONDA_ENV_NAME/bin:$PATH
RUN echo "source activate ${CONDA_ENV_NAME}" >> ~/.bashrc

# Enable jupyter lab
RUN source activate ${CONDA_ENV_NAME} && \
    conda install -c conda-forge jupyterlab && \
    jupyter serverextension enable --py jupyterlab --sys-prefix

# Install the packages
RUN source activate ${CONDA_ENV_NAME} && \
    conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

RUN set -x \
  && echo "Preparing user..." \
  && useradd -ms /bin/bash -d /app app \
  && adduser app docker \
  && groupadd fuse \
  && adduser app fuse

RUN set -x \
  echo "Installing jupyter kernel..." && \
  source activate ${CONDA_ENV_NAME} && \
  pip3 install --no-cache-dir ipython_genutils ipykernel && \
  python3 -m ipykernel install

RUN mkdir /Dr.Emb_Appyter
WORKDIR /Dr.Emb_Appyter
ADD . /Dr.Emb_Appyter

RUN echo "Downloading embedding vectors..." && \
  wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1bZpepqycN9LPPLXDqX8georOCYsAj_zD' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1bZpepqycN9LPPLXDqX8georOCYsAj_zD" -O Library.zip && rm -rf /tmp/cookies.txt && \
  unzip Library.zip && \
  wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1Co4rwFTR0jPVMq_0ee5JP_1v_AufhT3Z' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1Co4rwFTR0jPVMq_0ee5JP_1v_AufhT3Z" -O methods/mol2vec/mol2vec_model_300dim.pkl && rm -rf /tmp/cookies.txt

ENV APPYTER_PREFIX=/
ENV APPYTER_HOST=0.0.0.0
ENV APPYTER_PORT=5000
ENV APPYTER_DEBUG=false
ENV APPYTER_IPYNB=dr_emb.ipynb
ENV APPYTER_PROFILE=biojupies
ENV APPYTER_EXTRAS=["toc"]
ENV APPYTER_EXTRAS=["toggle-code"]

CMD ["appyter" "flask-app"]