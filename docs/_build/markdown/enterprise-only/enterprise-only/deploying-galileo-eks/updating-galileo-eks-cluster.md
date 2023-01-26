---
description: Galileo EKS cluster update from 1.21 -> 1.23
---

# Updating Galileo EKS cluster

### Prerequisites:

The AWS EBS CSI plugin has to be installed. This can be added to the `addons` sections in the eksctl config file.

```
addons:
  - name: aws-ebs-csi-driver
    version: 1.11.4
```

The Amazon EBS CSI plugin requires IAM permissions to make calls to AWS APIs on your behalf, additional EBS policy has to be attached to the existing Galileo node groups This can be added in the ekscstl config file:

```
withAddonPolicies:
- ebs: true
```

Apply changes to node-groups:

```
eksctl update nodegroup -f cluster-config.yaml
```

### Upgrade to 1.23

Because Amazon EKS runs a highly available control plane, you can update only one minor version at a time. Current cluster version is 1.21 and you want to update to 1.23. You must first update your cluster to 1.22 and then update your 1.22 cluster to 1.23.

#### Upgrade controle plane to 1.22

```
eksctl upgrade cluster --name CLUSTER_NAME --version 1.22 --approve
```

#### Upgrade node groups to 1.22

```
eksctl upgrade nodegroup --name=galileo-runner --cluster=CLUSTER_NAME --kubernetes-version=1.22

eksctl upgrade nodegroup --name=galileo-core --cluster=CLUSTER_NAME --kubernetes-version=1.22
```

#### Upgrade controle plane to 1.23

```
eksctl upgrade cluster --name CLUSTER_NAME --version 1.23 --approve
```

#### Upgrade node groups to 1.23

```
eksctl upgrade nodegroup --name=galileo-core --cluster=CLUSTER_NAME --kubernetes-version=1.23

eksctl upgrade nodegroup --name=galileo-runner --cluster=CLUSTER_NAME --kubernetes-version=1.23
```

#### Post upgrade checks

Check if all pods are in ready state:

```
kubectl get pods --all-namespaces -o go-template='{{ range  $item := .items }}{{ range .status.conditions }}{{ if (or (and (eq .type "PodScheduled") (eq .status "False")) (and (eq .type "Ready") (eq .status "False"))) }}{{ $item.metadata.name}} {{ end }}{{ end }}{{ end }}'
```

Check for pending persistance volumes:

```
kubectl get pvc --all-namespaces | grep -i pending
```
