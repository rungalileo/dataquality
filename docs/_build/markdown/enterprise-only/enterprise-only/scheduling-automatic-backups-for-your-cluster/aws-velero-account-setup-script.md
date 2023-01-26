# AWS velero account setup script

```
#!/bin/sh -e
#   Usage
#   ./velero-account-setup-aws.sh <BUCKET> <AWS_REGION>
#
#

print_usage() {
  echo -e "\n Usage: \n ./velero-account-setup-aws.sh <BUCKET> <AWS_REGION>\n"
}

BUCKET="${1}"
AWS_REGION="${2}"

if [ $# -ne 2 ]; then 
  print_usage
  exit 1
fi

aws s3api create-bucket \
    --bucket $BUCKET \
    --region $AWS_REGION \
    --create-bucket-configuration LocationConstraint=$REGION \
    --no-cli-pager

aws iam create-user --user-name velero --no-cli-pager

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

aws iam put-user-policy \
  --user-name velero \
  --policy-name velero \
  --policy-document file://velero-policy.json

resp=`aws iam create-access-key --user-name velero --no-cli-pager`

AWS_ACCESS_KEY_ID=`echo $resp | jq -r .AccessKey.AccessKeyId`
AWS_SECRET_ACCESS_KEY=`echo $resp | jq -r .AccessKey.SecretAccessKey`

cat > credentials-velero <<EOF
[default]
aws_access_key_id=$AWS_ACCESS_KEY_ID
aws_secret_access_key=$AWS_SECRET_ACCESS_KEY
EOF

echo "Credenials file created - credentials-velero"


```
