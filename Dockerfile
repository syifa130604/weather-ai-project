FROM python:3.10

RUN useradd -m -u 1000 user
USER user

ENV PATH="/home/user/.local/bin:${PATH}"

WORKDIR /home/user/app

COPY --chown=user requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .

EXPOSE 7860

# GANTI "nama_folder" dengan nama folder kamu yang sebenarnya (misal: src, backend, atau dll)
CMD ["gunicorn", "-b", "0.0.0.0:7860", "nama_folder.app:app"]
