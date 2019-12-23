FROM conda/miniconda3

COPY requirements.txt /setup/
	
RUN apt-get update && apt-get install gcc -y && pip install --upgrade -r /setup/requirements.txt   

CMD ["python"]