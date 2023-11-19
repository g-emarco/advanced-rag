
# Advanced RAG


![Alt Text](https://github.com/g-emarco/advanced-rag/blob/main/static/ai21-adanved-rag.png)


## Tech Stack


**Client:** Streamlit

**Server Side:** LangChain  🦜🔗


**Vectorstore:** PGVector

**Embeddings:** GCP VertexAI  

**LLMS:** PaLM 2, AI21 Contextual Answers


**Runtime:** Cloud Run  

## Environment Variables

To run this project, you will need to add the following environment variables to your .env file



`AI21_API_KEY`


## Run Locally


Clone the project

```bash
  git clone https://github.com/g-emarco/advanced-rag.git
```

Go to the project directory

```bash
  cd advanced-rag
```

Install dependencies

```bash
  pipenv install
```

Start the Streamlit server

```bash
  streamlit run app.py
```

NOTE: When running locally make sure `GOOGLE_APPLICATION_CREDENTIALS` is set to a service account with permissions to use VertexAI


## Deployment to cloud run

CI/CD via Cloud build is availale in ```cloudbuild.yaml```

Please replace $PROJECT_ID with your actual Google Cloud project ID.

To deploy manually:

1. Make sure you enable GCP APIs:

```bash
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable secretmanager.googleapis.com
gcloud services enable artifactregistry.googleapis.com
gcloud services enable vertexai.googleapis.com

```

2. Create a service account `vertex-ai-consumer` with the following roles:




```bash
gcloud iam service-accounts create vertex-ai-consumer \
    --display-name="Vertex AI Consumer"

gcloud projects add-iam-policy-binding PROJECT_ID \
    --member="serviceAccount:vertex-ai-consumer@PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/run.invoker"

gcloud projects add-iam-policy-binding PROJECT_ID \
    --member="serviceAccount:vertex-ai-consumer@PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/serviceusage.serviceUsageConsumer"

gcloud projects add-iam-policy-binding PROJECT_ID \
    --member="serviceAccount:vertex-ai-consumer@PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/ml.admin"

gcloud projects add-iam-policy-binding PROJECT_ID \
    --member="serviceAccount:vertex-ai-consumer@PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/vertexai.admin"

```

3. Create the secrets:

`AI21_API_KEY`

and for each secret grant the SA `vertex-ai-consumer@$PROJECT_ID.iam.gserviceaccount.com` Secret Manager Secret Accessor
role to th secrets

4. Build Image
```bash
docker build . -t us-east1-docker.pkg.dev/$PROJECT_ID/app/github-assitant:latest
```

5. Push to Artifact Registry
```bash
docker push us-east1-docker.pkg.dev/$PROJECT_ID/app/github-assitant:latest
```

6. Deploy to cloud run
```gcloud run deploy $PROJECT_ID \
    --image=us-east1-docker.pkg.dev/PROJECT_ID/app/github-assitant:latest \
    --region=us-east1 \
    --service-account=vertex-ai-consumer@$PROJECT_ID.iam.gserviceaccount.com \
    --allow-unauthenticated \
    --set-secrets="GOOGLE_API_KEY=projects/PROJECT_ID/secrets/AI21_API_KEY/versions/latest 
```



## 🚀 About Me
Eden Marco, Customer Engineer @ Google Cloud, Tel Aviv🇮🇱

[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/eden-marco/) 

[![twitter](https://img.shields.io/badge/twitter-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white)](https://twitter.com/EdenEmarco177)

