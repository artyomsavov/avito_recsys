# docker build -t avito . && docker run --gpus all --restart on-failure:3 -e SAMPLE=1.0 -e USERS=0.5 -v $PWD/data:/app/data avito
FROM python:3.12-slim
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility SAMPLE=1.0 PYTHONUNBUFFERED=1 UV_CONCURRENT_DOWNLOADS=32
# uv качает пакеты параллельно: pip в одно соединение тянул CuPy + CUDA (~3 GB) часами
RUN pip install --no-cache-dir uv==0.8.0 && uv pip install --system --no-cache polars==1.40.1 numpy==2.4.4 scipy==1.17.1 \
    pandas==3.0.3 pyarrow==24.0.0 catboost==1.2.10 "cupy-cuda12x[ctk]==14.2.0" nbconvert==7.16.6
WORKDIR /app
ENV USERS=1.0
COPY final.ipynb scripts/download_kaggle.py ./
# exec: сигналы (docker stop) получает python, а не sh
CMD python download_kaggle.py && jupyter nbconvert --to python final.ipynb \
 && sed -i "s/^SAMPLE = 0.01/SAMPLE = $SAMPLE/; s/^USERS = 1.0/USERS = $USERS/" final.py && exec python final.py
