FROM python:3.10

RUN useradd -m -u 1000 user

USER user

WORKDIR /home/user/app

COPY --chown=user . /home/user/app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 7860

CMD ["gunicorn", "-b", "0.0.0.0:7860", "app:app"]
