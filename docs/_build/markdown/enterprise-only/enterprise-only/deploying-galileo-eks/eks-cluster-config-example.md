# EKS Cluster Config Example

```yaml
---
apiVersion: eksctl.io/v1alpha5
kind: ClusterConfig

metadata:
  name: CLUSTER_NAME
  region: us-east-2
  version: "1.23"
  tags:
    env: CLUSTER_NAME

availabilityZones: ["us-east-2a", "us-east-2b"]

cloudWatch:
  clusterLogging:
    enableTypes: ["*"]

addons:
  - name: vpc-cni
    version: 1.11.0
  - name: aws-ebs-csi-driver
    version: 1.11.4
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
    instanceType: m5a.large
    minSize: 4
    maxSize: 5
    desiredCapacity: 5
    volumeSize: 200
    volumeType: gp2
    volumeEncrypted: true
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
        ebs: true
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
    instanceType: m5a.xlarge # m5a.large
    minSize: 1
    maxSize: 5
    desiredCapacity: 1
    volumeSize: 200 # GiB
    volumeType: gp2
    volumeEncrypted: true
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
        ebs: true
    updateConfig:
      maxUnavailable: 2

```
