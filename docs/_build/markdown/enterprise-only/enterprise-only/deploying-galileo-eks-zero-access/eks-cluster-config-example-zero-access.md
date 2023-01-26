# EKS Cluster Config Example (Zero Access)

```yaml
---
apiVersion: eksctl.io/v1alpha5
kind: ClusterConfig

metadata:
  name: CLUSTER_NAME
  region: us-east-2
  version: "1.21"
  tags:
    env: CLUSTER_NAME

vpc:
  id: VPC_ID
  subnets:
    private:
      us-east-2a:
          id: SUBNET_1_ID
      us-east-2b:
          id: SUBNET_2_ID

cloudWatch:
  clusterLogging:
    enableTypes: ["*"]

privateCluster:
  enabled: true

addons:
- name: vpc-cni
  version: 1.11.0

managedNodeGroups:
  - name: galileo-core
    privateNetworking: true
    availabilityZones: ["us-east-2a", "us-east-2b"]
    labels: { galileo-node-type: galileo-core }
    tags:
      {
        "k8s.io/cluster-autoscaler/CLUSTER_NAME": "owned",
        "k8s.io/cluster-autoscaler/enabled": "true",
      }
    amiFamily: AmazonLinux2
    instanceType: m5a.xlarge
    minSize: 4
    maxSize: 5
    desiredCapacity: 4
    volumeSize: 200 # GiB
    volumeType: gp2
    iam:
      attachPolicyARNs:
        - arn:aws:iam::aws:policy/AmazonEKSWorkerNodePolicy
        - arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy
        - arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly
        - arn:aws:iam::aws:policy/AmazonS3FullAccess
        - arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore
      withAddonPolicies:
        autoScaler: true
        cloudWatch: true
    updateConfig:
      maxUnavailable: 2
  - name: galileo-runner
    privateNetworking: true
    availabilityZones: ["us-east-2a", "us-east-2b"]
    labels: { galileo-node-type: galileo-runner }
    tags:
      {
        "k8s.io/cluster-autoscaler/CLUSTER_NAME": "owned",
        "k8s.io/cluster-autoscaler/enabled": "true",
      }
    amiFamily: AmazonLinux2
    instanceType: m5a.2xlarge
    minSize: 1
    maxSize: 5
    desiredCapacity: 1
    volumeSize: 200 # GiB
    volumeType: gp2
    iam:
      attachPolicyARNs:
        - arn:aws:iam::aws:policy/AmazonEKSWorkerNodePolicy
        - arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy
        - arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly
        - arn:aws:iam::aws:policy/AmazonS3FullAccess
        - arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore
      withAddonPolicies:
        autoScaler: true
        cloudWatch: true
    updateConfig:
      maxUnavailable: 2

```
