ARG PYTHON_VERSION
FROM python:${PYTHON_VERSION:-3.12.8-slim}

WORKDIR /app

COPY ./requirements.txt requirements.txt
# Not sure why bentoml is not installed properly when installing
# it through requirements.txt
RUN pip install bentoml 
RUN pip install -r requirements.txt


COPY ./infrastructure/bentoml_server/bentofile.yaml bentofile.yaml
COPY ./scripts/service.py scripts/service.py
COPY ./models models


# Set BentoML CLI as the entrypoint
ENTRYPOINT ["bentoml"]
CMD ["serve"]