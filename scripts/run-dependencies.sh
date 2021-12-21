#!/bin/sh -ex

echo "make prometheus metrics dir"
mkdir -p "$(pwd)/$PROMETHEUS_MULTIPROC_DIR"

echo "running postgres"
docker run --rm --name=postgres -d -p 5432:5432 -e POSTGRES_USER=galileo -e POSTGRES_PASSWORD=dataquality -e POSTGRES_HOST_AUTH_METHOD=password -e POSTGRES_DB=galileo postgres:10.5

echo "running minio"
mkdir -p /tmp/data
docker run --rm --name=minio -d -p 9000:9000 --mount type=bind,source=/tmp/data,target=/data minio/minio:RELEASE.2021-04-22T15-44-28Z server /data

echo "running api"
docker run --rm \
  -p 8088:8088 \
  -e GALILEO_DATABASE_URL_WRITE="postgresql+psycopg2://galileo:dataquality@host.docker.internal:5432/galileo" \
  -e GALILEO_DATABASE_URL_READ="postgresql+psycopg2://galileo:dataquality@host.docker.internal:5432/galileo" \
  -e GALILEO_API_SECRET_KEY="pancakes" \
  -e GALILEO_MINIO_FQDN="host.docker.internal:9000" \
  -e GALILEO_MINIO_REGION="us-east-1" \
  -e GALILEO_MINIO_ENDPOINT_URL="http://host.docker.internal:9000" \
  -e GALILEO_MINIO_K8S_SVC_ADDR="http://minio.galileo:9000" \
  -e GALILEO_MINIO_ACCESS_KEY="minioadmin" \
  -e GALILEO_MINIO_SECRET_KEY="minioadmin" \
  -e GALILEO_CONSOLE_URL="http://host.docker.internal:3000" \
  -e PROMETHEUS_MULTIPROC_DIR="galileo-prometheus-metrics" \
  gcr.io/rungalileo-dev/api:latest

sleep 5
echo "Creating CI user"
curl -X 'POST' \
  'http://localhost:8088/users' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "email": "ci@rungalileo.io",
  "first_name": "CI",
  "last_name": "user",
  "username": "ci_user",
  "auth_method": "email",
  "password": "ci_user_password!123"
}'
