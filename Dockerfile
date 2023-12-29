ARG UBUNTU_VERSION=20.04
ARG CUDA_VERSION=11.3.1
ARG CUDA=11.3
ARG CUDNN_VERSION=8

FROM nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_VERSION}-devel-ubuntu${UBUNTU_VERSION}
LABEL maintainer "http://medai.korea.ac.kr"

ARG PYTHON_VERSION=3.8

ENV DEBIAN_FRONTEND "noninteractive"
ENV TZ "America/New_York"

CMD ["/bin/bash"]

SHELL ["/bin/bash", "-c"]

RUN set -x \
  && echo "Preparing system..." \
  && apt-get -y update \
  && apt-get -y install \
    ca-certificates \
    ccache \
    cmake \
    curl \
    fuse \
    git \
    nginx \
    python3-dev \
    python3-pip \
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
    wget \
  && rm -rf /var/lib/apt/lists/* \
  && pip3 install --no-cache-dir --upgrade pip

ENV LD_LIBRARY_PATH /usr/local/cuda-${CUDA}/targets/x86_64-linux/lib:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 && \
    echo "/usr/local/cuda/lib64/stubs" > /etc/ld.so.conf.d/z-cuda-stubs.conf && \
    ldconfig

RUN set -x \
  && echo "Installing jupyter kernel..." \
  && pip3 install --no-cache-dir ipython_genutils ipykernel \
  && python3 -m ipykernel install

ADD requirements.txt /app/requirements.txt
RUN echo "Preparing Dr.Emb Appyter..." && \
  pip3 install --no-cache-dir -r /app/requirements.txt && \
  pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113 && \
  rm /app/requirements.txt

ARG appyter_version=appyter[production]@git+https://github.com/Maayanlab/appyter
RUN set -x \
  && echo "Installing appyter..." \
  && pip3 install --no-cache-dir --upgrade ${appyter_version}

COPY catalog_helper.py /bin/appyter-catalog-helper
RUN set -x \
  && echo "Installing catalog helper..." \
  && chmod 755 /bin/appyter-catalog-helper

RUN set -x \
  && echo "Preparing user..." \
  && useradd -ms /bin/bash -d /app app \
  && groupadd fuse \
  && adduser app fuse \
  && mkdir -p /app /app/data /data \
  && chown -R app:app /app /data \
  && chmod og+rwx -R /var/lib/nginx /var/log/nginx

USER app
WORKDIR /app
EXPOSE 5000
VOLUME /app/data

ENV PATH "/app:$PATH"
ENV PYTHONPATH "/app:$PYTHONPATH"
ENV APPYTER_PREFIX "/"
ENV APPYTER_HOST "0.0.0.0"
ENV APPYTER_PORT "5000"
ENV APPYTER_DEBUG "false"
ENV APPYTER_PROFILE "biojupies"
ENV APPYTER_EXTRAS '["toc", "toggle-code"]'
ENV APPYTER_IPYNB "dr_emb.ipynb"

COPY --chown=app:app . /app

# BEGIN CATALOG
RUN rm /app/catalog_helper.py
RUN appyter-catalog-helper setup
# END CATALOG