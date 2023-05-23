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

#ENTRYPOINT ["/bin/bash"]

# define parser url
ENV GITLAB=https://gitlab.polimi.it
ENV PARSER_URL=${GITLAB}/ai-sprint/ai-sprint-parser.git
ENV PROJECT_ID=776

############################################################################
#			build image for development                        #
############################################################################
FROM base as image-dev

# copy the last change from your brach to invalidate the cache if there 
# was a new change
ADD "${GITLAB}/api/v4/projects/${PROJECT_ID}/repository/branches/main" \
	/tmp/devalidateCache

# install parser (latest version)
RUN git clone ${PARSER_URL} ./ai_sprint_parser
RUN pip install --no-cache-dir -r ai_sprint_parser/requirements.txt
ENV PYTHONPATH="${PYTHONPATH}:/home/SPACE4AI-R/ai_sprint_parser"

# entrypoint
CMD bash

############################################################################
#			build image for production                         #
############################################################################
FROM base as image-prod

# define parser tag
ARG PARSER_TAG=23.05.03

# install parser 
RUN git clone	--depth 1 \
		--branch ${PARSER_TAG} \ 
		${PARSER_URL} \
		./ai_sprint_parser
RUN pip install --no-cache-dir -r ai_sprint_parser/requirements.txt
ENV PYTHONPATH="${PYTHONPATH}:/home/SPACE4AI-R/ai_sprint_parser"

# entrypoint
CMD bash

