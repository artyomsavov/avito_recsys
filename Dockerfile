# docker build -t avito . && docker run --gpus all -e SAMPLE=1.0 -v $PWD/data:/app/data avito
FROM python:3.12-slim
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility SAMPLE=1.0
RUN pip install --no-cache-dir polars==1.40.1 numpy==2.4.4 scipy==1.17.1 pandas==3.0.3 pyarrow==24.0.0 \
    catboost==1.2.10 "cupy-cuda12x[ctk]==14.2.0" nbconvert==7.16.6
WORKDIR /app
COPY final.ipynb scripts/download_kaggle.py ./
CMD python download_kaggle.py && jupyter nbconvert --to script final.ipynb \
 && sed -i "s/^SAMPLE = 0.01/SAMPLE = $SAMPLE/" final.py && python final.py
