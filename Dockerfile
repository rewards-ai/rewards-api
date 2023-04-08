FROM ubuntu:latest 
FROM python:3.10 

# install all the utilities 
RUN apt-get update && \
    apt-get install --no-install-recommends -y curl

# Installing python dependencies
RUN python3 -m pip --no-cache-dir install --upgrade pip && \
    python3 --version && \
    pip3 --version

COPY ./requirements.txt .
RUN pip install -r requirements.txt

# Make the folder 

RUN mkdir rewards-api
COPY ./rewards-api rewards-api
WORKDIR "/rewards-api"
EXPOSE 8900 
ENTRYPOINT [ "uvicorn", "main:app"]
CMD [ "--reload" ]