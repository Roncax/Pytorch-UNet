FROM pytorch/pytorch:latest

WORKDIR /unet
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .

CMD ["python", "train.py"]
