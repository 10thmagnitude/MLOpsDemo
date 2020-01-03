FROM conda/miniconda3

COPY requirements.txt /setup/

ENV AZURE_EXTENSION_DIR=/azextensions
	
RUN apt-get update && apt-get install gcc -y && pip install --upgrade -r /setup/requirements.txt && \
    mkdir -p $AZURE_EXTENSION_DIR && az extension add --name azure-cli-ml

# workaround for weird access_denied error
RUN touch /usr/local/lib/python3.7/site-packages/dotnetcore2/bin/deps.lock && chmod ugo+rw /usr/local/lib/python3.7/site-packages/dotnetcore2/bin/deps.lock

CMD ["python"]