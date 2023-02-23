FROM python:3.8

RUN apt-get update && apt-get install -y graphviz graphviz-dev doxygen

WORKDIR /home/SPACE4AI
# Install requirements for the SPACE4AI program
COPY ./requirements.txt /home/SPACE4AI/requirements.txt
RUN pip3 install -r /home/SPACE4AI/requirements.txt

# Install requirements for the aMLLibrary
COPY ./aMLLibrary/requirements.txt /home/SPACE4AI/aMLLibrary/requirements.txt
RUN pip3 install -r /home/SPACE4AI/aMLLibrary/requirements.txt

WORKDIR /home/SPACE4AI

ENTRYPOINT ["/bin/bash"]