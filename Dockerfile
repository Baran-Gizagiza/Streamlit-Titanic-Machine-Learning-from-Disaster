FROM ubuntu:18.04
RUN apt-get update && apt-get install -y \
    sudo \
    wget \
    vim \
    curl \
    unzip \
    git \
    tree \
    python3.7 \
    python3-pip

EXPOSE 8501

WORKDIR /opt
COPY requirements.txt /opt/app/requirements.txt
WORKDIR /opt/app
RUN pip3 install --upgrade -r requirements.txt
COPY . /opt/app
WORKDIR /

CMD streamlit run app.py
