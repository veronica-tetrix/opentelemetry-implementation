# OpenTelemetry Distributed Tracing Demo

Simple demo showing distributed tracing between two FastAPI services using OpenTelemetry.

## What This Does

- User Service calls Product Service
- OpenTelemetry traces the request across both services
- View traces in Jaeger UI

## Setup & Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start Infrastructure (Jaeger + OpenTelemetry Collector)
```bash
docker-compose up -d
```


### 3. Start Services

**Terminal 1 - Product Service:**
```bash
cd services/product-service  
python main.py
```
`http://localhost:8000`

**Terminal 2 - User Service:**
```bash
cd services/user-service
python main.py  
```
`http://localhost:8001`

### 4. Test Individual Services
```bash
# Product service
curl http://localhost:8000/health

# User service  
curl http://localhost:8001/health
```

### 5. Test Cross-Service Call (Creates Distributed Trace)
```bash
curl http://localhost:8001/users/1/recommendations
```

### 6. View Traces in Jaeger
1. Open http://localhost:16686
2. Select "user-service" from the Service dropdown
3. Click "Find Traces"
4. Click on a trace to see the distributed request flow:
   - `user-service: get_user_recommendations` 
   - `product-service: recommend_products`

## Files Explained
- `services/user-service/main.py` - User service (calls Product service)
- `services/product-service/main.py` - Product service (returns recommendations)
- `docker-compose.yml` - Runs Jaeger, OpenTelemetry Collector, OpenSearch
- `otel-collector-config.yaml` - Routes traces from services to Jaeger