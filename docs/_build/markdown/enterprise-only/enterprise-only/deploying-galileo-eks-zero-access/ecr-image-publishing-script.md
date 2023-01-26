---
description: >-
  Use this script to download the image tarballs from Galileo's cloud storage
  and push them to your private ECR repositories.
---

# ECR Image Publishing Script

{% hint style="warning" %}
You must have the provided `service-account.json` credential file in the same directory that this script is run from.&#x20;
{% endhint %}

{% hint style="info" %}
Running this script requires about 2.5GB of disk space for temporary storage of image tarballs. All intermediate files will be deleted automatically.&#x20;
{% endhint %}

```
#!/bin/bash -e
export AWS_PAGER=""

CONTAINERS_BUCKET=${CONTAINERS_BUCKET:="galileo-containers"}
REPOSITORY_PREFIX=${REPOSITORY_PREFIX:="galileo"}

echo "Publishing images..."
images=(
    gcr.io/rungalileo-prod/postgres:10.5
    gcr.io/rungalileo-prod/minio:RELEASE.2021-04-22T15-44-28Z
    gcr.io/rungalileo-prod/runners:v0.2.21
    gcr.io/rungalileo-prod/api:v0.2.25
    gcr.io/rungalileo-prod/ui:v0.1.62
    gcr.io/rungalileo-prod/grafana:7.5.11
    gcr.io/rungalileo-prod/prometheus:v2.31.1
    gcr.io/rungalileo-prod/alertmanager:v0.23.0
    gcr.io/rungalileo-prod/node-exporter:v1.3.0
    gcr.io/rungalileo-prod/metrics-server:v0.5.1
    gcr.io/rungalileo-prod/kube-state-metrics:v2.2.4
    gcr.io/rungalileo-prod/nginx-ingress-controller:v1.1.1
    gcr.io/rungalileo-prod/kube-webhook-certgen:v1.1.1
    gcr.io/rungalileo-prod/cert-manager-controller:v1.7.1
    gcr.io/rungalileo-prod/cert-manager-webhook:v1.7.1
    gcr.io/rungalileo-prod/cert-manager-cainjector:v1.7.1
    gcr.io/rungalileo-prod/alpine:3.7
)

gcloud auth activate-service-account --key-file=./service-account.json

aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com

gcloud config set storage/check_hashes if_fast_else_skip

for image in ${images[*]}; do
    DIR="$(basename $image | cut -d':' -f1)"
    VERSION="$(basename $image | cut -d':' -f2)"
    FILENAME="$(basename $image).tar"
    echo "Downloading $FILENAME from gs://$CONTAINERS_BUCKET/$DIR/"
    gcloud --verbosity="critical" --quiet alpha storage cp "gs://$CONTAINERS_BUCKET/$DIR/$FILENAME" $FILENAME
    docker load -i $FILENAME

    echo "Tagging image as $AWS_ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPOSITORY_PREFIX/$DIR:$VERSION"
    docker tag $image $AWS_ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPOSITORY_PREFIX/$DIR:$VERSION

    echo "Describing or Creating ECR repository $AWS_ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPOSITORY_PREFIX/$DIR"
    aws ecr describe-repositories --region $REGION --repository-names $REPOSITORY_PREFIX/$DIR || \
    aws ecr create-repository --region $REGION --repository-name $REPOSITORY_PREFIX/$DIR --image-tag-mutability IMMUTABLE
    
    echo "Pushing image to $AWS_ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPOSITORY_PREFIX/$DIR:$VERSION"
    docker push $AWS_ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPOSITORY_PREFIX/$DIR:$VERSION
    
    echo "Removing image tarball $FILENAME"
    rm $FILENAME
done
```
