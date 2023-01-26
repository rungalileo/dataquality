# 🥽 Deploying Galileo – GKE

Get your Kubernetes cluster up and running while setting up IAM and the 4 Galileo DNS endpoints. The Galileo applications run on managed Kubernetes environments like EKS and GKE, but this document will specifically cover the configuration and deployment of a GKE environment.

{% hint style="info" %}
**⏱ Total time for deployment:** 30-45 minutes
{% endhint %}

{% hint style="info" %}
**This deployment requires the use of Google Cloud's CLI, `gcloud`.  Please follow** [**these instructions**](https://cloud.google.com/sdk/docs/install) **to install and set up gcloud for your GCP account.**
{% endhint %}

### Recommended Cluster Configuration

Galileo recommends the following Kubernetes deployment configuration. These details are captured in the bootstrap script Galileo provides.

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
|                    **Storage class**                   |                  standard                  |

### Step 0: Deploying the GKE Cluster

Run [this script](https://app.gitbook.com/o/-MO05cVyQ2tmzGFt9tky/s/4jVWiQpRqmi04OnqZmn2/\~/changes/PlDLDrPRWPBeWPFwsDkA/enterprise-only/enterprise-only/deploying-galileo-gke/galileo-gcp-setup-script) as instructed. If you have any questions, please reach out to a Galilean in the slack channel Galileo shares with you and your team.

### **Step 1: Required Configuration Values**

Customer specific cluster values (e.g. domain name, slack channel for notifications etc) will be placed in a base64 encoded string, stored as a secret in GitHub that Galileo’s deployment automation will read in and use when templating a cluster’s resource files.\


**Mandatory fields the Galileo team requires:**

|                  Mandatory Field                 |                                                                                                                                              Description                                                                                                                                             |
| :----------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|                **GCP Account ID**                |                                                                                                           The Customer's GCP Account ID that the customer will use for provisioning Galileo                                                                                                          |
|           **Customer GCP Project Name**          |                                                                                                                The Name of the GCP project the customer is using to provision Galileo.                                                                                                               |
| **Customer Service Account Address for Galileo** |                                                                                                  The Service account address the customer has created for the galileo deployment account to assume.                                                                                                  |
|               **GKE Cluster Name**               |                                                                                                                    The GKE cluster name that Galileo will deploy the platform to.                                                                                                                    |
|                  **Domain Name**                 |                                                                                                                    The customer wishes to deploy the cluster under e.g. google.com                                                                                                                   |
|              **GKE Cluster Region**              |                                                                                                                                      The region of the cluster.                                                                                                                                      |
|                **Root subdomain**                |                                                                                                                                e.g. "galileo" as in galileo.google.com                                                                                                                               |
|      **Trusted SSL Certificates (Optional)**     | <p></p><p>By default, Galileo provisions Let’s Encrypt certificates. But if you wish to use your own trusted SSL certificates, you should submit a base64 encoded string of</p><ol><li>the full certificate chain, and</li><li>another, separate base64 encoded string of the signing key.</li></ol> |

### Step 2: Access to Deployment Logs

As a customer, you have full access to the deployment logs in Google Cloud Storage. You (customer) are able to view all configuration there. A customer email address must be provided to have access to this log.

### **Step 3: Customer DNS Configuration**

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
