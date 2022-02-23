FROM python:3.8

RUN apt-get update && apt-get install -y graphviz graphviz-dev doxygen

# Install requirements for the SPACE4AI program
COPY ./requirements.txt /home/SPACE4AI/requirements.txt
RUN pip3 install -r /home/SPACE4AI/requirements.txt

# Install requirements for the a-MLlibrary
COPY ./a-MLlibrary/requirements.txt /home/SPACE4AI/a-MLlibrary/requirements.txt
RUN pip3 install -r /home/SPACE4AI/a-MLlibrary/requirements.txt

WORKDIR /home/SPACE4AI

ENTRYPOINT ["/bin/bash"]
