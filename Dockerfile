FROM ubuntu:18.04
RUN apt-get update && apt-get install -y \
    sudo \
    wget \
    vim \
    curl \
    unzip \
    git \
    tree

EXPOSE 8501

WORKDIR /opt
RUN wget https://repo.continuum.io/archive/Anaconda3-2020.07-Linux-x86_64.sh && \
    sh /opt/Anaconda3-2020.07-Linux-x86_64.sh -b -p /opt/anaconda3 && \
    rm -f Anaconda3-2020.07-Linux-x86_64.sh
ENV PATH /opt/anaconda3/bin:$PATH
COPY requirements.txt /opt/app/requirements.txt
WORKDIR /opt/app
RUN pip3 install --upgrade -r requirements.txt
COPY . /opt/app

WORKDIR /work/streamlit/python
CMD ["streamlit", "run", "streamlit_app.py"]
