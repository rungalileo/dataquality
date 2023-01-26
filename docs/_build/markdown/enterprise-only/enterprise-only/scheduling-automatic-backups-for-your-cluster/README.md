# ⚙ Scheduling Automatic Backups for your Cluster

### Velero

Velero is a convenient backup tool for Kubernetes clusters that compresses and backs up Kubernetes objects to object storage. It also takes snapshots of your cluster’s Persistent Volumes using your cloud provider’s block storage snapshot features, and can then restore your cluster’s objects and Persistent Volumes to a previous state.

{% embed url="https://velero.io/docs/v1.9/" %}

### Installing the Velero CLI

MacOS:

```
brew install velero
```

Linux:

```
INSTALL_PATH='/usr/local/bin'
wget -O velero.tar.gz https://github.com/vmware-tanzu/velero/releases/download/v1.9.2/velero-v1.9.2-linux-amd64.tar.gz
tar -xvf velero.tar.gz && cd velero-v1.9.2-linux-amd64 && mv velero $INSTALL_PATH && chmod +x ${INSTALL_PATH}/velero
```

### Prerequisites

Before setting up the velero components, you will need to prepare your AWS/GCP object storage, secrets and a dedicated user with access to resources required to perform a backup. The instructions below will guide you.

### AWS EKS: Installing Velero

[AWS Setup Script](aws-velero-account-setup-script.md)

Create s3 bucket:

```
aws s3api create-bucket \
    --bucket <BUCKET_NAME> \
    --region <AWS_REGION> \
    --create-bucket-configuration LocationConstraint=<AWS_REGION>
```

1. Create IAM user and attach a IAM policy with necessary permissions:

```
aws iam create-user --user-name velero
```

IAM policy:

```
cat > velero-policy.json <<EOF
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "ec2:DescribeVolumes",
                "ec2:DescribeSnapshots",
                "ec2:CreateTags",
                "ec2:CreateVolume",
                "ec2:CreateSnapshot",
                "ec2:DeleteSnapshot"
            ],
            "Resource": "*"
        },
        {
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:DeleteObject",
                "s3:PutObject",
                "s3:AbortMultipartUpload",
                "s3:ListMultipartUploadParts"
            ],
            "Resource": [
                "arn:aws:s3:::${BUCKET}/*"
            ]
        },
        {
            "Effect": "Allow",
            "Action": [
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::${BUCKET}"
            ]
        }
    ]
}
EOF
```

```
aws iam put-user-policy \
  --user-name velero \
  --policy-name velero \
  --policy-document file://velero-policy.json
```

2\. Create an access key for the user and note the AWS\_SECRET\_ACCESS\_KEY and AWS\_ACCESS\_KEY\_ID.

```
aws iam create-access-key --user-name velero
```

3\. Create a Velero-specific credentials file (credentials-velero)

```
[default]
aws_access_key_id=<AWS_ACCESS_KEY_ID>
aws_secret_access_key=<AWS_SECRET_ACCESS_KEY>
```

All the steps above are included in the [AWS velero account setup script](aws-velero-account-setup-script.md)

4\. Installing velero

The velero install command will perform the setup steps to get the cluster ready for backups.

```
    velero install \
      --provider aws \
      --backup-location-config region=<AWS_REGION> \
      --snapshot-location-config region=<AWS_REGION> \
      --bucket velero-backups \
      --plugins velero/velero-plugin-for-aws:v1.4.0 \
      --secret-file ./credentials-velero
```

### GCP GKE: Installing Velero

[GCP Setup Script](gcp-velero-account-setup-script.md)

1. Create GCS bucket

```
gsutil mb gs://<BUCKET_NAME>/
```

2\. Create Google Service Account (GSA)

```
gcloud iam service-accounts create velero \
    --display-name "Velero service account"
```

3\. Create Custom Role with Permissions for the Velero

```bash
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

PROJECT_ID=$(gcloud config get-value project)

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

gsutil iam ch serviceAccount:$SERVICE_ACCOUNT_EMAIL:objectAdmin gs://<BUCKET_NAME>

gcloud iam service-accounts keys create credentials-velero \
    --iam-account $SERVICE_ACCOUNT_EMAIL
```

All the steps above are included in the [GCP velero account setup script](gcp-velero-account-setup-script.md)

4\. Install velero

```
velero install \
  --provider gcp \
  --bucket velero-backups \
  --plugins velero/velero-plugin-for-gcp:v1.5.0 \
  --secret-file ./credentials-velero
```

#### Backups

Setup daily backups:

```
velero schedule create daily-backups --schedule "0 7 * * *"

# Take initial backup
velero backup create --from-schedule daily-backup

# Get backup list
velero backup get
NAME                          STATUS      ERRORS   WARNINGS   CREATED                          EXPIRES   STORAGE LOCATION   SELECTOR
daily-backup-20221004070030   Completed   0        0          2022-10-04 09:00:30 +0200 CEST   29d       default            <none>
daily-backup-20221003193617   Completed   0        0          2022-10-03 21:36:30 +0200 CEST   29d       default            <none>
```

#### Restore from backup

NOTE: Existing cluster resources will not be overwritten by the restoration process. To restore a PV delete it from the cluster before running the restore command

```
velero restore create --from-backup daily-backup-20221003193617
```

NOTE: All DNS entries have to be updated after restore as velero does not persist the ingress IP/LB names.

