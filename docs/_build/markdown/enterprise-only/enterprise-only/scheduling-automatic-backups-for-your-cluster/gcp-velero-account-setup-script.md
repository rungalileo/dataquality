# GCP velero account setup script

```
#!/bin/sh -e
#   Usage
#   ./velero-account-setup-gcp.sh <BUCKET>
#
#
GSA_NAME=velero

ROLE_PERMISSIONS=(
    compute.disks.get
    compute.disks.create
    compute.disks.createSnapshot
    compute.snapshots.get
    compute.snapshots.create
    compute.snapshots.useReadOnly
    compute.snapshots.delete
    compute.zones.get
    storage.objects.create
    storage.objects.delete
    storage.objects.get
    storage.objects.list
)

print_usage() {
  echo -e "\n Usage: \n ./velero-account-setup-gcp.sh <BUCKET>\n"
}

BUCKET="${1}"

if [ -z "$BUCKET" ]; then
  print_usage
  exit 1
fi

gsutil mb gs://$BUCKET

PROJECT_ID=$(gcloud config get-value project)

gcloud iam service-accounts create $GSA_NAME \
    --display-name "Velero service account"

SERVICE_ACCOUNT_EMAIL=$(gcloud iam service-accounts list \
  --filter="displayName:Velero service account" \
  --format 'value(email)')

gcloud iam roles create velero.server \
    --project $PROJECT_ID \
    --title "Velero Server" \
    --permissions "$(IFS=","; echo "${ROLE_PERMISSIONS[*]}")"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member serviceAccount:$SERVICE_ACCOUNT_EMAIL \
    --role projects/$PROJECT_ID/roles/velero.server

gsutil iam ch serviceAccount:$SERVICE_ACCOUNT_EMAIL:objectAdmin gs://${BUCKET}

gcloud iam service-accounts keys create credentials-velero \
    --iam-account $SERVICE_ACCOUNT_EMAIL

```
