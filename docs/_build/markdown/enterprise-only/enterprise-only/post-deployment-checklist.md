---
description: >-
  The following guide will walk you through steps you can take to make sure your
  Galileo cluster is properly deployed and running well.
---

# 👩⚕ Post Deployment Checklist

_This guide applies to all cloud providers._

### 1. Confirm that all DNS records have been created.

Galileo will not set DNS records for your cluster and as such you need to set those appropriately for your company. Each record should have a TTL of 60 seconds or less.

If you are letting Galileo provision Let's Encrypt certificates for you automatically with cert-manager, it's important to make sure that all of cert-manager's http solvers have told Let's Encrypt to provision a certificate with all of the domains specified for the cluster (i.e. `api|console|data|grafana.my-cluster.my-domain.com` )&#x20;

```
kubectl get ingress -n galileo | grep -i http-solver
```

When you run the above command, if you see no output, then the solvers should have finished. You can check this by visiting any of the domains for your cluster.

### 2. Check the API's health-check.

```
curl -I -X GET https://api.<CLUSTER_SUBDOMAIN>.<CLUSTER_DOMAIN>/healthcheck
```

If the response is a 200, then this is a good sign that almost everything is up and running as expected.

### 3. Check for unready pods.

```
kubectl get pods --all-namespaces -o go-template='{{ range  $item := .items }}{{ range .status.conditions }}{{ if (or (and (eq .type "PodScheduled") (eq .status "False")) (and (eq .type "Ready") (eq .status "False"))) }}{{ $item.metadata.name}} {{ end }}{{ end }}{{ end }}'
```

If any pods are in an unready state, especially in the namespace where the Galileo platform was deployed, please notify the appropriate representative from Galileo and they will help to solve the issue.

### 4. Check for pending persistent volume claims.

```
kubectl get pvc --all-namespaces | grep -i pending
```

If any persistent volume claims are in a pending state, especially in the namespace where the Galileo platform was deployed, please notify the appropriate representative from Galileo and they will help to solve the issue.
