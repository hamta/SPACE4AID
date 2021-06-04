FROM python:3.7

RUN apt-get update && apt-get install -y graphviz graphviz-dev doxygen

COPY ./requirements.txt /home/SPACE4AI/requirements.txt
RUN pip3 install -r /home/SPACE4AI/requirements.txt

COPY . /home/SPACE4AI
WORKDIR /home/SPACE4AI

ENTRYPOINT ["/bin/bash"]
