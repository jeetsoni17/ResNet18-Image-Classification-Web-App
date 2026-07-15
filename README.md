# 🚀 ResNet Image Classification on Kubernetes with Custom Queue-Based Autoscaling

A cloud-native image classification system built using **PyTorch, Docker, Kubernetes, Redis, and Minikube**, designed to evaluate different autoscaling strategies for AI inference workloads.

The project compares Kubernetes Horizontal Pod Autoscaler (HPA) with a custom queue-length-based autoscaler under varying workloads to determine which approach provides better latency, CPU utilization, and overall system performance.

---

## 📌 Features

- 🖼️ Image Classification using pretrained ResNet-18
- ☸️ Kubernetes deployment with Docker containers
- 📈 Horizontal Pod Autoscaler (70% & 90% CPU targets)
- ⚡ Custom Queue-Length Based Autoscaler
- 📦 Redis message queue for asynchronous processing
- 📊 Prometheus monitoring
- 🧪 Automated load testing
- 📉 CPU, Memory & Latency analysis
- 📈 Performance graph generation

---

## 🏗️ System Architecture

The application follows a distributed microservice architecture.

```
                Client
                   │
                   ▼
             Dispatcher API
                   │
                   ▼
             Redis Queue
                   │
        ┌──────────┴──────────┐
        ▼                     ▼
 Custom Autoscaler      Worker Pods
        │                     │
        └──────────► Kubernetes ◄──────────┐
                                           │
                                  ResNet-18 Model
```

### Components

| Component | Purpose |
|-----------|----------|
| Dispatcher | Receives image requests and pushes them into Redis |
| Redis | Stores incoming prediction jobs |
| Worker Pods | Perform image classification using ResNet-18 |
| Custom Autoscaler | Scales workers based on Redis queue length |
| Kubernetes | Orchestrates deployments and scaling |
| Prometheus | Collects monitoring metrics |
| Load Tester | Simulates user traffic |

---

# 🛠 Tech Stack

### Backend
- Python
- FastAPI

### Machine Learning
- PyTorch
- ResNet-18

### Cloud & DevOps
- Docker
- Kubernetes
- Minikube
- kubectl
- Helm

### Monitoring
- Prometheus
- Kubernetes Metrics Server

### Messaging
- Redis

### Testing
- Python Load Tester

---

# 📂 Project Structure

```
ResNet_Project
│
├── autoscaler/
│   ├── autoscaler.py
│   ├── Dockerfile
│   └── requirements.txt
│
├── dispatcher/
│   ├── dispatcher.py
│   ├── Dockerfile
│   └── requirements.txt
│
├── worker/
│   ├── worker.py
│   ├── Dockerfile
│   └── requirements.txt
│
├── loadtester/
│   ├── tester.py
│   └── sample-images/
│
├── k8s/
│   ├── deployment.yaml
│   ├── worker.yaml
│   ├── dispatcher.yaml
│   ├── redis.yaml
│   ├── hpa70.yaml
│   ├── hpa90.yaml
│   ├── autoscaler.yaml
│   └── service.yaml
│
├── graph/
│   ├── cpu.py
│   ├── P99_Latency.py
│   └── statistics.py
│
└── README.md
```

---

# ⚙️ Prerequisites

Install the following before running the project:

- Docker Desktop
- Kubernetes
- Minikube
- kubectl
- Helm
- Python 3.x
- Git

Verify installation:

```bash
docker --version
minikube version
kubectl version --client
python --version
helm version
```

---

# 🚀 Running the Project

## 1. Start Minikube

```bash
minikube start --driver=docker
```

Enable Metrics Server

```bash
minikube addons enable metrics-server
```

---

## 2. Deploy Kubernetes Resources

```bash
kubectl apply -f k8s/
```

Verify deployment

```bash
kubectl get pods
kubectl get deployments
kubectl get svc
```

---

## 3. Install Monitoring Stack

```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts

helm repo update

helm install monitoring prometheus-community/kube-prometheus-stack
```

---

## 4. Expose Application

```bash
minikube tunnel
```

Then obtain the application URL

```bash
minikube service resnet-service --url
```

---

## 5. Run Load Test

```bash
cd loadtester

python tester.py
```

---

# 📊 Autoscaling Strategies

## HPA (70%)

```bash
kubectl apply -f k8s/hpa70.yaml
```

Monitor

```bash
kubectl get hpa -w
```

---

## HPA (90%)

```bash
kubectl apply -f k8s/hpa90.yaml
```

---

## Custom Autoscaler

Unlike Kubernetes HPA, the custom autoscaler scales worker pods using the Redis queue length.

Scaling formula:

```
Desired Replicas =
Ceiling(Current Queue Length / Messages Per Worker)
```

Configuration

| Parameter | Value |
|-----------|------|
| Poll Interval | 15 seconds |
| Messages per Worker | 10 |
| Minimum Replicas | 1 |
| Maximum Replicas | 10 |

---

# 📈 Monitoring

Monitor cluster resources

```bash
kubectl top pods

kubectl top nodes

kubectl get hpa -w
```

View application logs

```bash
kubectl logs -f deployment/resnet-deployment
```

---

# 📊 Performance Metrics

The project compares:

- CPU Utilization
- Memory Usage
- Average Latency
- P95 Latency
- P99 Latency
- Worker Replicas
- Throughput

Graphs generated:

- CPU Usage
- Average Latency
- P99 Latency
- Overall Statistics

---

# 📉 Experimental Results

| Metric | HPA 70 | HPA 90 | Custom Autoscaler |
|---------|---------|----------|------------------|
| CPU Stability | Medium | Low | ✅ High |
| Average Latency | Good | Poor | Very Good |
| P95 Latency | Medium | Poor | ✅ Best |
| P99 Latency | High | Very High | ✅ Lowest |
| Burst Handling | Medium | Poor | ✅ Excellent |
| Resource Efficiency | Medium | Low | ✅ High |

### Key Findings

- Queue-length-based scaling reacted faster to workload changes than CPU-based scaling.
- Custom Autoscaler maintained lower P99 latency under heavy traffic.
- HPA 90% delayed scaling, resulting in CPU spikes and increased response times.
- HPA 70% improved responsiveness but exhibited frequent scaling oscillations.
- Custom Autoscaler provided the most stable and efficient resource utilization.

---

# 📈 Generate Performance Graphs

```bash
python cpu.py

python P99_Latency.py

python statistics.py
```

---

# 📚 Future Improvements

- GPU-based inference
- Multi-node Kubernetes cluster
- Grafana dashboards
- CI/CD using GitHub Actions
- Kubernetes Cluster Autoscaler integration
- Model versioning and A/B testing

---

# 📝 License

This project is developed for academic and research purposes.
