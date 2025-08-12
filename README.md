# OpenTelemetry Distributed Tracing Demo

Demo showing distributed tracing between three FastAPI services using OpenTelemetry.

## What This Does

- User Service calls Product Service
- OpenTelemetry traces the request across all three services
- View traces in Jaeger UI / OpenSearch Dashboard

## Setup & Run
```bash
pip install -r requirements.txt
docker-compose up -d # start infrastructure (Jaeger + OpenTelemetry Collector)
cd services/user-service && python main.py
cd services/product-service && python main.py
cd services/inventory-service && python main.py
```
`http://localhost:8000`<br>
`http://localhost:8001`<br>
`http://localhost:8002`

```bash
curl http://localhost:8000/health
curl http://localhost:8001/health
curl http://localhost:8002/health
```

### 5. Test Cross-Service Calls (Creates Distributed Traces)

#### Services Calls
```bash
curl http://localhost:8001/users/1/recommendations
curl http://localhost:8001/users/1/shopping-analysis
```

### 6. View Traces

In Jaeger UI (for better trace visualization):
http://localhost:16686
   - `user-service: get_user_recommendations` 
   - `product-service: recommend_products`
   - `product-service: calculate_user_score`
   - `product-service: filter_products_by_category`
   - `product-service: apply_pricing_logic`
   - `product-service: enrich_with_inventory`
   - `inventory-service: get_inventory`
   - `inventory-service: check_warehouse_capacity`
   - `inventory-service: calculate_shipping_time`

In OpenSearch Dashboard:
http://localhost:5601

## Files Explained
- `services/user-service/main.py` - User service (calls Product service)
- `services/product-service/main.py` - Product service (returns recommendations, calls Inventory service)
- `services/inventory-service/main.py` - Inventory service (manages product availability and shipping)
- `docker-compose.yml` - Runs Jaeger, OpenTelemetry Collector, OpenSearch, and OpenSearch Dashboards
- `otel-collector-config.yaml` - Routes traces from services to Jaeger