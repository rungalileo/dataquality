# 🔰 Deploying Galileo - EKS

Get your Kubernetes cluster up and running while setting up IAM and Trust Policies and the 4 Galileo DNS endpoints. The Galileo applications run on managed Kubernetes environments like EKS and GKE, but this document will specifically cover the configuration and deployment of an EKS environment.

{% hint style="info" %}
**⏱ Total time for deployment:** 30-45 minutes
{% endhint %}

{% hint style="info" %}
**This deployment requires the use of AWS CLI commands.  If you only have cloud console access, follow the optional instructions below to get** [**eksctl**](https://eksctl.io/introduction/#installation) **working with AWS CloudShell.**
{% endhint %}

### Step 0: (Optional) Deploying via AWS CloudShell

To use [`eksctl`](https://eksctl.io/introduction/#installation) via CloudShell in the AWS console, open a CloudShell session and do the following:

```bash
# Create directory
mkdir -p $HOME/.local/bin
cd $HOME/.local/bin

# eksctl
curl --silent --location "https://github.com/weaveworks/eksctl/releases/latest/download/eksctl_$(uname -s)_amd64.tar.gz" | tar xz -C /tmp
sudo mv /tmp/eksctl $HOME/.local/bin
```

The rest of the installation deployment can now be run from the CloudShell session. You can use `vim` to create/edit the required yaml and json files within the shell session.

### Recommended Cluster Configuration

Galileo recommends the following Kubernetes deployment configuration:

|                      Configuration                     |              Recommended Value             |
| :----------------------------------------------------: | :----------------------------------------: |
|        **Nodes in the cluster’s core nodegroup**       | <p>4 (min) <br>5 (max) <br>4 (desired)</p> |
|                  **CPU per core node**                 |                    4 CPU                   |
|                  **RAM per core node**                 |                 16 GiB RAM                 |
| **Number of nodes in the cluster’s runners nodegroup** |         1 (min) 5 (max) 1 (desired)        |
|                 **CPU per runner node**                |                    8 CPU                   |
|                 **RAM per runner node**                |                 32 GiB RAM                 |
|            **Minimum volume size per node**            |                   200 GiB                  |
|           **Required Kubernetes API version**          |                    1.21                    |
|                    **Storage class**                   |                     gp2                    |

Here's an [example EKS cluster configuration](eks-cluster-config-example.md).

### Step 1: Creating Roles and Policies for the Cluster

* **Galileo IAM Policy:** This policy is attached to the Galileo IAM Role. Add the following to a file called `galileo-policy.json`

```
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "eks:AccessKubernetesApi",
                "eks:DescribeCluster"
            ],
            "Resource": "arn:aws:eks:CLUSTER_REGION:ACCOUNT_ID:cluster/CLUSTER_NAME"
        }
    ]
}
```

* **Galileo IAM Trust Policy:** This trust policy enables an external Galileo user to assume your Galileo IAM Role to deploy changes to your cluster securely. Add the following to a file called `galileo-trust-policy.json`

```
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "AWS": [
                    "arn:aws:iam::273352303610:role/GalileoConnect"
                ],
                "Service": "ec2.amazonaws.com"
            },
            "Action": "sts:AssumeRole"
        }
    ]
}
```

* **Galileo IAM Role with Policy:** Role should only include the Galileo IAM Policy mentioned in this table. Create a file called `create-galileo-role-and-policies.sh`, make it executable with `chmod +x create-galileo-role-and-policies.sh` and run it. Make sure to run in the same directory as the json files created in the above steps.

```
#!/bin/sh -ex

aws iam create-policy --policy-name Galileo --policy-document file://galileo-policy.json
aws iam create-role --role-name Galileo --assume-role-policy-document file://galileo-trust-policy.json
aws iam attach-role-policy --role-name Galileo --policy-arn $(aws iam list-policies | jq -r '.Policies[] | select (.PolicyName == "Galileo") | .Arn')

```

### Step 2: Deploying the EKS Cluster

With the role and policies created, the cluster itself can be deployed in a single command using [eksctl](https://eksctl.io/introduction/#installation). Using the cluster template [here](eks-cluster-config-example.md), create a `galileo-cluster.yaml` file and edit the contents to replace `CUSTOMER_NAME` with your company name like `galileo`. Also check and update all `availabilityZones` as appropriate.

With the yaml file saved, run the following command to deploy the cluster:

```
eksctl create cluster -f galileo-cluster.yaml
```

### Step 3: EKS IAM Identity Mapping

This ensures that only users who have access to this role can deploy changes to the cluster. Account owners can also make changes. This is easy to do with [eksctl](https://eksctl.io/usage/iam-identity-mappings/) with the following command:

```
eksctl create iamidentitymapping
--cluster customer-cluster
--region your-region-id
--arn "arn:aws:iam::CUSTOMER-ACCOUNT-ID:role/Galileo"
--username galileo
--group system:masters
```

{% hint style="info" %}
**NOTE for the user:** For connected clusters, Galileo will apply changes from github actions. So github.com should be allow-listed for your cluster’s ingress rules if you have any specific network requirements.
{% endhint %}

### **Step 4: Required Configuration Values**

Customer specific cluster values (e.g. domain name, slack channel for notifications etc) will be placed in a base64 encoded string, stored as a secret in GitHub that Galileo’s deployment automation will read in and use when templating a cluster’s resource files.\


|                                 Mandatory Field                                |                                                                                                                                              Description                                                                                                                                             |
| :----------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|                               **AWS Account ID**                               |                                                                                                           The Customer's AWS Account ID that the customer will use for provisioning Galileo                                                                                                          |
|                            **Galileo IAM Role Name**                           |                                                                                                     The AWS IAM Role name the customer has created for the galileo deployment account to assume.                                                                                                     |
|                              **EKS Cluster Name**                              |                                                                                                                    The EKS cluster name that Galileo will deploy the platform to.                                                                                                                    |
|                                 **Domain Name**                                |                                                                                                                    The customer wishes to deploy the cluster under e.g. google.com                                                                                                                   |
|                               **Root subdomain**                               |                                                                                                                                e.g. "galileo" as in galileo.google.com                                                                                                                               |
|                     **Trusted SSL Certificates (Optional)**                    | <p></p><p>By default, Galileo provisions Let’s Encrypt certificates. But if you wish to use your own trusted SSL certificates, you should submit a base64 encoded string of</p><ol><li>the full certificate chain, and</li><li>another, separate base64 encoded string of the signing key.</li></ol> |
| **AWS Access Key ID and Secret Access Key for Internal S3 Uploads (Optional)** |                                                                 If you would like to export data into an s3 bucket of your choice. Please let us know the access key and secret key of the account that can make those upload calls.                                                                 |

{% hint style="info" %}
**NOTE for the user:** Let Galileo know if you’d like to use LetsEncrypt or your own certificate before deployment.
{% endhint %}

### Step 5: Access to Deployment Logs

As a customer, you have full access to the deployment logs in Google Cloud Storage. You (customer) are able to view all configuration there. A customer email address must be provided to have access to this log.

### **Step 6: Customer DNS Configuration**

Galileo has 4 main URLs (shown below). In order to make the URLs accessible across the company, you have to set the following DNS addresses in your DNS provider after the platform is deployed.&#x20;

{% hint style="info" %}
**⏱ Time taken :** 5-10 minutes (post the ingress endpoint / load balancer provisioning)
{% endhint %}

| Service |                     URL                     |
| :-----: | :-----------------------------------------: |
|   API   |   **api.galileo**.company.\[com\|ai\|io…]   |
|   Data  |   **data.galileo**.company.\[com\|ai\|io…]  |
|    UI   | **console.galileo**.company.\[com\|ai\|io…] |
| Grafana | **grafana.galileo**.company.\[com\|ai\|io…] |

Each URL must be entered as a CNAME record into your DNS management system as the ELB address. You can find this address by listing the kubernetes ingresses that the platform has provisioned.



### Step 7: Post-deployment health-checks
