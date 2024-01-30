FROM python:3.9

WORKDIR /usr/src


RUN pip install numpy
RUN pip install Cython

COPY requirements.txt .
RUN pip install --no-cache-dir -r /usr/src/requirements.txt

COPY . .

CMD ["python", "model.py"]