# OpenTelemetry Distributed Tracing Demo

Demo showing distributed tracing between three FastAPI services using OpenTelemetry.

## What This Does

- User Service calls Product Service
- OpenTelemetry traces the request across all three services
- View traces in Jaeger UI / OpenSearch Dashboard

## Setup & Run
```bash
pip install -r requirements.txt
open -a Docker
docker-compose up -d
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

#### Services Calls
```bash
curl http://localhost:8001/users/1/recommendations
curl http://localhost:8001/users/1/shopping-analysis
```

#### Custom Metadata Injection

Generate traces run:
```bash
# different users with session tracking
curl -H "x-client-id: client-001" -H "x-session-id: session-abc" "http://localhost:8000/products/recommend?user_id=1001&category=electronics"
curl -H "x-client-id: client-002" -H "x-session-id: session-xyz" "http://localhost:8000/products/recommend?user_id=2002"
```

#### View Traces
#### Custom Metadata Injection

Generate traces run:
```bash
# different users with session tracking
curl -H "x-client-id: client-001" -H "x-session-id: session-abc" "http://localhost:8001/products/recommend?user_id=1001&category=electronics"
curl -H "x-client-id: client-002" -H "x-session-id: session-xyz" "http://localhost:8001/products/recommend?user_id=2002"
```

#### View Traces

**Jaeger UI** (for better trace visualization):
http://localhost:16686

**Note**: If Jaeger fails to start initially due to OpenSearch connection issues, restart it:
```bash
docker restart jaeger
```
   - `user-service: get_user_recommendations` 
   - `product-service: recommend_products`
   - `product-service: calculate_user_score`
   - `product-service: filter_products_by_category`
   - `product-service: apply_pricing_logic`
   - `product-service: enrich_with_inventory`
   - `inventory-service: get_inventory`
   - `inventory-service: check_warehouse_capacity`
   - `inventory-service: calculate_shipping_time`
   - Service: `product-service` 
   - Click any trace -> expand spans -> see custom metadata:
   - `business.user_id`, `business.session_id`
   - `performance.execution_time_ms`, `system.memory_usage_mb`
   - `custom.operation_type`, feature flags
   - Service: `product-service` 
   - Click any trace -> expand spans -> see custom metadata:
   - `business.user_id`, `business.session_id`
   - `performance.execution_time_ms`, `system.memory_usage_mb`
   - `custom.operation_type`, feature flags

In OpenSearch Dashboard:
http://localhost:5601
- Create index pattern: `jaeger-jaeger-span-*`
- Filter: `process.serviceName: "product-service" AND operationName: "recommend_products"`
Or we can go to Dev Tools and run the following queries:
```bash
# find spans with specific user:
GET /jaeger-jaeger-span-*/_search
  {
    "query": {
      "bool": {
        "must": [
          {"term": {"operationName": "recommend_products"}},
          {"nested": {
            "path": "tags",
            "query": {
              "bool": {
                "must": [
                  {"term": {"tags.key": "business.user_id"}},
                  {"term": {"tags.value": "1234"}}
                ]
              }
            }
          }}
        ]
      }
    }
  }

  # get performance metrics:
  GET /jaeger-jaeger-span-*/_search
  {
    "query": {
      "nested": {
        "path": "tags",
        "query": {
          "term": {"tags.key": "performance.execution_time_ms"}
        }
      }
    },
    "_source": ["duration", "tags"],
    "size": 5
  }
```

In Prometheus metrics:
http://localhost:8889/metrics

#### Files Explained
- `services/user-service/main.py` - User service (calls Product service)
- `services/product-service/main.py` - Product service (returns recommendations, calls Inventory service)
- `services/inventory-service/main.py` - Inventory service (manages product availability and shipping)
- `docker-compose.yml` - Runs Jaeger, OpenTelemetry Collector, OpenSearch, and OpenSearch Dashboards
- `otel-collector-config.yaml` - Routes traces from services to Jaeger