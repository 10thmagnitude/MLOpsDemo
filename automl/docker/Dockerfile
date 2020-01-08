FROM conda/miniconda3

COPY requirements.txt /setup/

ENV AZURE_EXTENSION_DIR=/azextensions
	
RUN apt-get update && apt-get install gcc jq curl libicu57 -y && pip install --upgrade -r /setup/requirements.txt && \
    mkdir -p $AZURE_EXTENSION_DIR && az extension add --name azure-cli-ml

# install dotnet core
RUN dotnet_version=3.1.0 \
    && curl -SL --output dotnet.tar.gz https://dotnetcli.azureedge.net/dotnet/Runtime/$dotnet_version/dotnet-runtime-$dotnet_version-linux-x64.tar.gz \
    && dotnet_sha512='99949807c00871d66e8ce7c25c14998e78a0ea60ba8cc42244643ed2e13aa360285df1c8d27729df3efb319f4af9163ea5626c1478a9dd4bed9d2a58e01d6343' \
    && echo "$dotnet_sha512 dotnet.tar.gz" | sha512sum -c - \
    && mkdir -p /usr/share/dotnet \
    && tar -ozxf dotnet.tar.gz -C /usr/share/dotnet \
    && rm dotnet.tar.gz \
    && ln -s /usr/share/dotnet/dotnet /usr/bin/dotnet

# workaround for weird access_denied error
RUN chmod ugo+rw /usr/local/lib/python3.7/site-packages/dotnetcore2/bin

CMD ["python"]