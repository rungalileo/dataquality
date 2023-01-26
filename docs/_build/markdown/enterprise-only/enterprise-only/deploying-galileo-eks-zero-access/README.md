# 🦕 Deploying Galileo - EKS (Zero Access)

Create a private Kubernetes Cluster with EKS in your AWS Account, upload containers to your container registry, and deploy Galileo.

{% hint style="info" %}
**⏱ Total time for deployment:** 45-60 minutes
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

Here's an [example EKS cluster configuration](eks-cluster-config-example-zero-access.md).

### Step 1: Deploying the EKS Cluster

The cluster itself can be deployed in a single command using [eksctl](https://eksctl.io/introduction/#installation). Using the cluster template [here](eks-cluster-config-example-zero-access.md), create a `galileo-cluster.yaml` file and edit the contents to replace CLUSTER`_NAME` with a name for your cluster like `galileo`. Also check and update all `availabilityZones` as appropriate.

With the yaml file saved, run the following command to deploy the cluster:

```
eksctl create cluster -f galileo-cluster.yaml
```

### **Step 2: Required Configuration Values**

Customer specific cluster values (e.g. domain name, slack channel for notifications etc) will be placed in a base64 encoded string, stored as a secret in GitHub that Galileo’s deployment automation will read in and use when templating a cluster’s resource files.\


**Mandatory fields the Galileo team requires:**

|              Mandatory Field             |                                                                                                Description                                                                                                |
| :--------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| **Container Registry Repository\*\*\***  |               The Registry Repository where you have uploaded Galileo containers. For example, in ECR, it would look something like 000000000000.dkr.ecr.MY-REGION-1.amazonaws.com/REPO-NAME              |
|           **EKS Cluster Name**           |                                                                       The EKS cluster name that Galileo will deploy the platform to.                                                                      |
|          **EKS Cluster Region**          |                                                                     The AWS Region the cluster is deployed to, for example, us-east-2                                                                     |
|              **Domain Name**             |                                                                      The customer wishes to deploy the cluster under e.g. google.com                                                                      |
|            **Root subdomain**            |                                                                              e.g. "**galileo**" as in **galileo**.google.com                                                                              |
|       **Trusted SSL Certificates**       | <p></p><p>These certificate should support the provided domain name. You should submit 2 base64 encoded strings;</p><ol><li>one for the full certificate chain</li><li>one for the signing key.</li></ol> |

### Step 3: Download Containers Images and Publish to ECR&#x20;

The required container images will be accessible via Galileo's cloud storage, so you can use the provided script to download these images and publish them to your private ECR repositories.&#x20;

{% hint style="warning" %}
These steps are to be performed on a machine with internet access
{% endhint %}

**A service account credential file will be provided so you can access the required data.**&#x20;

1. Make sure that you have the latest version of the AWS CLI and Docker installed. For more information, see [Getting Started with Amazon ECR ](http://docs.aws.amazon.com/AmazonECR/latest/userguide/getting-started-cli.html).
2. Install the `gcloud` CLI tool appropriate for your system, see [Install the gcloud CLI](https://cloud.google.com/sdk/docs/install).
3. Run the provided [ECR Image Publishing Script](ecr-image-publishing-script.md) to download the image tarballs and publish them to ECR.
   1. Make sure the `service-account.json` credential file is in the same directory
   2. Both `AWS_ACCOUNT_ID` and `REGION` are required to run the script

### Step 4: Deploy the Galileo Applications

VPN access is required to connect to the Kubernetes API when interacting with a private cluster. If you do not have appropriate VPN access with private DNS resolution, you can use a bastion machine with public ssh access as a bridge to the private cluster. The bastion will only act as a simple shell environment, so a machine type of `t3.micro` or equivalent will suffice.

{% hint style="warning" %}
Except where specifically noted, these steps are to be performed on a machine with internet access
{% endhint %}

1. Download version 1.21 of `kubectl` as explained [here](https://docs.aws.amazon.com/eks/latest/userguide/install-kubectl.html), and `scp` that file to the working directory of the bastion.
2. Generate the cluster config file by running `aws eks update-kubeconfig --name $CLUSTER_NAME --region $REGION`&#x20;
3. If using a bastion machine, prepare the required environment with the following:
   1. Either `scp` or copy and paste the contents of `~/.kube/config` from your local machine to the same directory on the bastion
   2. `scp` the provided `deployment-manifest.yaml` file to the working directory of the bastion
4. With your VPN connected, or if using a bastion, ssh'ing into the bastion's shell:
   1. Run `kubectl cluster-info` to verify your cluster config is set appropriately. If the cluster information is returned, you can proceed with the deployment.
   2. Run `kubectl apply -f deployment-manifest.yaml` to deploy the Galileo applications. Re-run this command if there are errors related to custom resources not being defined as there are sometimes race conditions when applying large templates.

### **Step 4: Customer DNS Configuration**

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

Each URL must be entered as a CNAME record into your DNS management system as the ELB address. You can find this address by running `kubectl -n galileo get svc/ingress-nginx-controller` and looking at the value for `EXTERNAL-IP`.
