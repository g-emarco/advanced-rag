FROM python:3.11-slim
WORKDIR /app
COPY Pipfile Pipfile.lock ./
RUN python -m pip install --upgrade pip
RUN pip install pipenv && pipenv install torch && pipenv install --dev --system --deploy
COPY . .
EXPOSE 8080
ENTRYPOINT streamlit run app_medical.py --server.address=0.0.0.0 --server.port=8080