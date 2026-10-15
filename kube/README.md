# Kubernetes setup

R2E-Gym uses Kubernetes pods as isolated CPU environments for repository tasks and executable verification. The GPU training machines run Ray, veRL, and vLLM; every GPU machine must be able to reach the Kubernetes API.

This repository connects to an existing Kubernetes cluster. It does not create the cluster for you.

## Requirements

- A Linux Kubernetes cluster with containerd or another compatible container runtime.
- A kubeconfig with permission to create, inspect, execute commands in, and delete pods in the `default` namespace.
- Outbound network access from task pods when required by a repository.
- Enough pod capacity, memory, CPU, disk, and image-cache space for the requested rollout concurrency.
- Docker Hub credentials if the task images require authenticated pulls or you need higher pull limits.

The paper used two CPU nodes with 128 CPU cores and approximately 500 GB of memory per node. Each optimization step could request 512 environments from 64 tasks with eight rollouts each. These values describe the paper setup and are not minimum requirements.

## Configure cluster access

Obtain a kubeconfig from your cluster administrator or cloud provider. Store it securely and do not commit it to the repository.

```bash
export KUBECONFIG=/path/to/your/kubeconfig
chmod 600 "$KUBECONFIG"
```

Cloud-managed clusters commonly provide it with one of these commands:

```bash
az aks get-credentials --resource-group "my-resource-group" --name "my-cluster"
aws eks update-kubeconfig --region "us-east-1" --name "my-cluster"
gcloud container clusters get-credentials "my-cluster" --region "us-east1"
```

Install `kubectl` if needed and validate the credentials:

```bash
bash kube/setup_kube.sh
```

The script only installs `kubectl` and checks the current context, API server, and nodes. It does not change kernel settings or provision cluster resources.

## Create the image-pull secret

The current R2E-Gym Kubernetes runtime expects a Docker registry secret named `dockerhub-pro` in the `default` namespace. Create a Docker Hub access token and supply it through your shell environment:

```bash
export DOCKERHUB_USERNAME="<your-dockerhub-username>"
export DOCKERHUB_TOKEN="<your-dockerhub-access-token>"

kubectl create secret docker-registry dockerhub-pro \
  --namespace default \
  --docker-server=https://index.docker.io/v1/ \
  --docker-username="$DOCKERHUB_USERNAME" \
  --docker-password="$DOCKERHUB_TOKEN"
```

Do not put the token itself in a YAML file or shell script. If your cluster already has access to every image, create or configure the expected secret before launching, or adjust the runtime's pull-secret setting for your deployment.

## Validate access from a GPU machine

Run these checks on every GPU machine that will launch environments:

```bash
kubectl config current-context
kubectl cluster-info
kubectl get nodes
python kube/test_kube.py "$KUBECONFIG"
python kube/test_kube_connect.py
```

The Python checks verify kubeconfig loading, API access, and one real R2E-Gym pod lifecycle. You can inspect the required permissions before a full run:

```bash
kubectl auth can-i create pods -n default
kubectl auth can-i get pods -n default
kubectl auth can-i create pods/exec -n default
kubectl auth can-i delete pods -n default
```

## Kubernetes API overrides

By default, R2E-Gym uses the API endpoint from the kubeconfig or in-cluster configuration. You can explicitly provide one endpoint or a comma-separated pool:

```bash
export R2EGYM_K8S_API_SERVER="https://your-api-server"
export R2EGYM_K8S_API_SERVERS="https://api-1,https://api-2"
```

Use the overrides only when the endpoint in your kubeconfig is not reachable from the GPU machines.

## Cache task images

Pre-pulling the R2E-Gym images on every CPU node avoids large rollout startup delays. After installing the GPU environment and configuring Kubernetes:

```bash
source .venv/bin/activate
python rllm/environments/swe/cache_images_k8.py
```

The helper reads the `R2E-Gym/R2E-Gym-Subset` image list and creates temporary DaemonSets. It expects the `dockerhub-pro` secret. Monitor the run with:

```bash
kubectl get daemonsets,pods -n default
kubectl get events -n default --sort-by=.lastTimestamp
```

## Capacity and cleanup

The runtime currently requests `80m` CPU and `300Mi` memory per environment pod and allows a larger CPU limit. These small requests help scheduling, but they do not represent worst-case repository usage. Measure representative workloads before increasing concurrency.

For large runs:

- Increase the maximum pods per node to match the required concurrency.
- Leave headroom for asynchronously prefetched environments.
- Configure containerd image garbage collection and disk eviction so frequently used images remain cached.
- Monitor failed and completed pods and clean them up promptly.
- Use a unique `RUN_NAME`; training attaches it as the `pod-tag` label.

Inspect pods for one run:

```bash
export RUN_NAME="<your-run-name>"
kubectl get pods -n default -l "pod-tag=$RUN_NAME"
```

Delete pods for that run after confirming the label:

```bash
kubectl delete pods -n default -l "pod-tag=$RUN_NAME"
```

Avoid broad namespace-wide deletion commands on a shared cluster.

## Tokens used outside Kubernetes

Training also requires a Hugging Face token for model access and a Weights & Biases API key for logging. Create them in the respective services and export them only on the GPU machines:

```bash
export HF_TOKEN="<your-hugging-face-token>"
export WANDB_API_KEY="<your-wandb-api-key>"
```

The launch scripts stop with an error when either variable is missing. Never replace these with literal placeholders such as `huggingface_TOKEN` or `WANDB_TOKEN` inside the scripts.
