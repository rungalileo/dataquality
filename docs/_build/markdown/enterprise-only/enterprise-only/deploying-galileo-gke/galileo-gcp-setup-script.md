# Galileo GCP Setup Script



```
#!/bin/sh -e
#
#   Usage
#      CUSTOMER_NAME=customer-name REGION=us-central1 ZONE_ID=a ./bootstrap.sh

if [ -z "$CUSTOMER_NAME" ]; then
    echo "Error: CUSTOMER_NAME is not set"
    exit 0
fi

PROJECT="$CUSTOMER_NAME-galileo"
REGION=${REGION:="us-central1"}
ZONE_ID=${ZONE_ID:="c"}
ZONE="$REGION-$ZONE_ID"
CLUSTER_NAME="galileo"

echo "Bootstrapping cluster with the following parameters:"
echo "PROJECT: ${PROJECT}"
echo "REGION: ${REGION}"
echo "ZONE: ${ZONE}"
echo "CLUSTER_NAME: ${CLUSTER_NAME}"

#
#   Create a project for Galileo.
#
echo "Create a project for Galileo."
gcloud projects create $PROJECT || true

#
#   Enabling services as referenced here https://cloud.google.com/migrate/containers/docs/config-dev-env#enabling_required_services
#
echo "Enabling services as referenced here https://cloud.google.com/migrate/containers/docs/config-dev-env#enabling_required_services"
gcloud services enable --project=$PROJECT servicemanagement.googleapis.com servicecontrol.googleapis.com cloudresourcemanager.googleapis.com compute.googleapis.com container.googleapis.com containerregistry.googleapis.com cloudbuild.googleapis.com

#
#   Grab the project number.
#
echo "Grab the project number."
PROJECT_NUMBER=$(gcloud projects describe $PROJECT --format json | jq -r -c .projectNumber)

#
#   Create service accounts and policy bindings.
#
echo "Create service accounts and policy bindings."
gcloud iam service-accounts create galileoconnect \
--project "$PROJECT"

gcloud iam service-accounts add-iam-policy-binding galileoconnect@$PROJECT.iam.gserviceaccount.com \
--project "$PROJECT" \
--member "group:devs@rungalileo.io" \
--role "roles/iam.serviceAccountUser"

gcloud iam service-accounts add-iam-policy-binding galileoconnect@$PROJECT.iam.gserviceaccount.com \
--project "$PROJECT" \
--member "group:devs@rungalileo.io" \
--role "roles/iam.serviceAccountTokenCreator"

gcloud projects add-iam-policy-binding $PROJECT --member="serviceAccount:galileoconnect@$PROJECT.iam.gserviceaccount.com" --role="roles/container.admin"

gcloud projects add-iam-policy-binding $PROJECT --member="serviceAccount:galileoconnect@$PROJECT.iam.gserviceaccount.com" --role="roles/container.clusterViewer"

#
#   Waiting before provisioning workload identity.
#
echo "Waiting before provisioning workload identity..."
sleep 5

#
#   Create a workload identity pool.
#
echo "Create a workload identity pool."
gcloud iam workload-identity-pools create galileoconnectpool \
--project "$PROJECT" \
--location "global" \
--description "Workload ID Pool for Galileo via GitHub Actions" \
--display-name "GalileoConnectPool"

#
#   Create a workload identity provider .
#
echo "Create a workload identity provider ."
gcloud iam workload-identity-pools providers create-oidc galileoconnectprovider \
--project "$PROJECT" \
--location "global" \
--workload-identity-pool "galileoconnectpool" \
--display-name "GalileoConnectProvider" \
--attribute-mapping="google.subject=assertion.sub,attribute.actor=assertion.actor,attribute.aud=assertion.aud,attribute.repository_owner=assertion.repository_owner,attribute.repository=assertion.repository" \
--issuer-uri="https://token.actions.githubusercontent.com"

#
#   Bind the service account to the workload identity provider.
#
echo "Bind the service account to the workload identity provider."
gcloud iam service-accounts add-iam-policy-binding "galileoconnect@${PROJECT}.iam.gserviceaccount.com" \
--project "$PROJECT" \
--role="roles/iam.workloadIdentityUser" \
--member="principalSet://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/galileoconnectpool/attribute.repository/rungalileo/deploy"

#
#   Create the cluster (with one node pool) and the runners node pool.
#
echo "Create the cluster (with one node pool) and the runners node pool."
gcloud beta container \
--project $PROJECT clusters create $CLUSTER_NAME \
--zone $ZONE \
--no-enable-basic-auth \
--cluster-version "1.21" \
--release-channel "regular" \
--machine-type "e2-standard-4" \
--image-type "COS" \
--disk-type "pd-standard" \
--disk-size "300" \
--node-labels galileo-node-type=galileo-core \
--metadata disable-legacy-endpoints=true \
--scopes "https://www.googleapis.com/auth/devstorage.read_write","https://www.googleapis.com/auth/logging.write","https://www.googleapis.com/auth/monitoring","https://www.googleapis.com/auth/servicecontrol","https://www.googleapis.com/auth/service.management.readonly","https://www.googleapis.com/auth/trace.append" \
--max-pods-per-node "110" \
--num-nodes "4" \
--logging=SYSTEM,WORKLOAD \
--monitoring=SYSTEM \
--enable-ip-alias \
--network "projects/$PROJECT/global/networks/default" \
--subnetwork "projects/$PROJECT/regions/$REGION/subnetworks/default" \
--no-enable-intra-node-visibility \
--default-max-pods-per-node "110" \
--enable-autoscaling \
--min-nodes "4" \
--max-nodes "5" \
--no-enable-master-authorized-networks \
--addons HorizontalPodAutoscaling,HttpLoadBalancing,GcePersistentDiskCsiDriver \
--enable-autoupgrade \
--enable-autorepair \
--max-surge-upgrade 1 \
--max-unavailable-upgrade 0 \
--enable-autoprovisioning \
--min-cpu 0 \
--max-cpu 50 \
--min-memory 0 \
--max-memory 200 \
--autoprovisioning-scopes=https://www.googleapis.com/auth/logging.write,https://www.googleapis.com/auth/monitoring \
--enable-autoprovisioning-autorepair \
--enable-autoprovisioning-autoupgrade \
--autoprovisioning-max-surge-upgrade 1 \
--autoprovisioning-max-unavailable-upgrade 0 \
--enable-shielded-nodes \
--node-locations $ZONE \
--enable-network-policy

gcloud beta container \
--project $PROJECT node-pools create "galileo-runners" \
--cluster $CLUSTER_NAME \
--zone $ZONE \
--machine-type "e2-standard-8" \
--image-type "COS" \
--disk-type "pd-standard" \
--disk-size "100" \
--node-labels galileo-node-type=galileo-runner \
--metadata disable-legacy-endpoints=true \
--scopes "https://www.googleapis.com/auth/devstorage.read_write","https://www.googleapis.com/auth/logging.write","https://www.googleapis.com/auth/monitoring","https://www.googleapis.com/auth/servicecontrol","https://www.googleapis.com/auth/service.management.readonly","https://www.googleapis.com/auth/trace.append" \
--num-nodes "1" \
--enable-autoscaling \
--min-nodes "1" \
--max-nodes "3" \
--enable-autoupgrade \
--enable-autorepair \
--max-surge-upgrade 1 \
--max-unavailable-upgrade 0 \
--max-pods-per-node "110" \
--node-locations $ZONE
```
