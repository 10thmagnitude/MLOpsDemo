FROM conda/miniconda3

COPY requirements.txt /setup/

ENV AZURE_EXTENSION_DIR=/azextensions
	
RUN apt-get update && apt-get install gcc -y && pip install --upgrade -r /setup/requirements.txt && \
    mkdir -p $AZURE_EXTENSION_DIR && az extension add --name azure-cli-ml

CMD ["python"]