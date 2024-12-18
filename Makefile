# Local images - using local computer's architecture.
# i.e. linux/amd64 for Windows / Linux / Apple with Intel chip
#      linux/arm64 for Apple with Apple Silicon (M1 / M2 chip)

docker_build_local:
	docker build --tag=$(GAR_IMAGE):local .

docker_run_local:
	docker run \
		-e PORT=8000 -p 8000:8000 \
		--env-file .env \
		$(GAR_IMAGE):local

docker_run_local_interactively:
	docker run -it \
		-e PORT=8000 -p 8000:8000 \
		--env-file .env \
		$(GAR_IMAGE):local \
		bash

# Cloud images - using architecture compatible with cloud, i.e. linux/amd64

gar_creation:
	gcloud auth configure-docker ${GCP_REGION}-docker.pkg.dev
	gcloud artifacts repositories create ${GAR_REPO} --repository-format=docker \
  --location=${GCP_REGION} --description="Repository for storing ${GAR_REPO} images"

docker_build:
	docker build --platform linux/amd64 -t ${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT}/${GAR_REPO}/${GAR_IMAGE}:prod .

docker_push:
	docker push ${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT}/${GAR_REPO}/${GAR_IMAGE}:prod

docker_run:
	docker run -e PORT=8000 -p 8000:8000 --env-file .env ${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT}/${GAR_REPO}/${GAR_IMAGE}:prod

docker_interactive:
	docker run -it --env-file .env ${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT}/${GAR_REPO}/${GAR_IMAGE}:prod /bin/bash

docker_deploy:
	gcloud run deploy --image ${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT}/${GAR_REPO}/${GAR_IMAGE}:prod --region ${GCP_REGION} --memory 8Gi --cpu 2 --max-instances 2
