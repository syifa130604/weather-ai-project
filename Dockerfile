FROM python:3.10

RUN useradd -m -u 1000 user
USER user

ENV PATH="/home/user/.local/bin:${PATH}"

WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .

EXPOSE 7860

# Perintah ini sangat krusial:
# --chdir app : Menyuruh server masuk ke dalam folder "app"
# app:app     : Menjalankan file "app.py" dan mencari variabel bernama "app"
CMD ["gunicorn", "--chdir", "app", "-b", "0.0.0.0:7860", "app:app"]
