# Adventure Story Weaver: a tiny GPT trained from scratch on public domain adventure novels.
#   docker build -t story-weaver .
#   docker run --rm story-weaver generate "The cave mouth was black, and"
#   docker run --rm -it story-weaver play
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

# CPU-only PyTorch keeps the image small; the trained model is tiny and runs fine on CPU.
COPY requirements.txt .
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements.txt

COPY . .

ENTRYPOINT ["python", "-m", "app.cli"]
CMD ["generate", "The cave mouth was black, and from somewhere deep inside it came", "--seed", "1"]
