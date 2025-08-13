import os
import httpx
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from opentelemetry import trace, metrics
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.propagate import extract, inject
from opentelemetry.trace import Status, StatusCode
import time
import psutil
import threading
from typing import Dict, Any, Optional

SERVICE_NAME = os.getenv("SERVICE_NAME", "product-service")
OTEL_ENDPOINT = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
INVENTORY_SERVICE_URL = os.getenv("INVENTORY_SERVICE_URL", "http://localhost:8002")

resource = Resource.create({
    "service.name": SERVICE_NAME,
    "service.version": "1.0.0",
    "deployment.environment": os.getenv("ENVIRONMENT", "development")
})

# Configure tracing
trace.set_tracer_provider(TracerProvider(resource=resource))
tracer = trace.get_tracer(__name__)

otlp_exporter = OTLPSpanExporter(endpoint=OTEL_ENDPOINT, insecure=True)
span_processor = BatchSpanProcessor(otlp_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)

# Configure metrics
metric_reader = PeriodicExportingMetricReader(
    OTLPMetricExporter(endpoint=OTEL_ENDPOINT, insecure=True),
    export_interval_millis=5000,
)
metrics.set_meter_provider(MeterProvider(resource=resource, metric_readers=[metric_reader]))
meter = metrics.get_meter(__name__)

app = FastAPI(title="Product Service")
FastAPIInstrumentor.instrument_app(app)
HTTPXClientInstrumentor().instrument()

class Product(BaseModel):
    id: int
    name: str
    category: str
    price: float

products_db = {
    1: Product(id=1, name="Laptop Pro", category="electronics", price=1299.99),
    2: Product(id=2, name="Wireless Headphones", category="electronics", price=199.99),
    3: Product(id=3, name="Programming Book", category="books", price=49.99),
}

class RealCostTracker:
    """Track real OpenTelemetry operation costs and system metrics"""
    
    def __init__(self):
        # OpenTelemetry metrics
        self.operation_counter = meter.create_counter(
            "otel_operations_total",
            description="Total number of OpenTelemetry operations",
            unit="1"
        )
        
        self.span_duration_histogram = meter.create_histogram(
            "otel_span_duration_ms",
            description="Duration of OpenTelemetry spans in milliseconds",
            unit="ms"
        )
        
        self.http_requests_counter = meter.create_counter(
            "http_requests_total",
            description="Total number of HTTP requests made",
            unit="1"
        )
        
        self.memory_usage_gauge = meter.create_gauge(
            "memory_usage_mb",
            description="Current memory usage in MB",
            unit="MB"
        )
        
        self.cpu_usage_gauge = meter.create_gauge(
            "cpu_usage_percent",
            description="Current CPU usage percentage",
            unit="percent"
        )
        
        # Start system metrics collection
        self._start_system_metrics_collection()
    
    def _start_system_metrics_collection(self):
        """Start background thread to collect system metrics"""
        def collect_system_metrics():
            while True:
                try:
                    # Memory usage
                    memory = psutil.virtual_memory()
                    self.memory_usage_gauge.set(memory.used / (1024 * 1024))  # Convert to MB
                    
                    # CPU usage
                    cpu_percent = psutil.cpu_percent(interval=1)
                    self.cpu_usage_gauge.set(cpu_percent)
                    
                except Exception:
                    pass
                time.sleep(5)
        
        thread = threading.Thread(target=collect_system_metrics, daemon=True)
        thread.start()
    
    def record_operation(self, operation_type: str, duration_ms: float = None, **attributes):
        """Record an OpenTelemetry operation with real metrics"""
        labels = {"operation_type": operation_type, "service": SERVICE_NAME}
        labels.update(attributes)
        
        # Count the operation
        self.operation_counter.add(1, labels)
        
        # Record duration if provided
        if duration_ms is not None:
            self.span_duration_histogram.record(duration_ms, labels)
    
    def record_http_request(self, method: str, endpoint: str, status_code: int, duration_ms: float):
        """Record HTTP request metrics"""
        labels = {
            "method": method,
            "endpoint": endpoint,
            "status_code": str(status_code),
            "service": SERVICE_NAME
        }
        
        self.http_requests_counter.add(1, labels)
        
        # Also record as operation
        self.record_operation("http_request", duration_ms, **labels)

cost_tracker = RealCostTracker()

class MetadataInjector:
    """Custom metadata injection API for OpenTelemetry spans with real metrics"""
    
    def __init__(self, cost_tracker: RealCostTracker):
        self.cost_tracker = cost_tracker
    
    def inject_custom_metadata(self, span, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Inject custom metadata into a span and record real metrics"""
        start_time = time.time()
        
        try:
            attribute_count = 0
            
            # Inject standard metadata with proper span attributes for OpenSearch visibility
            for key, value in metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    span.set_attribute(f"custom.{key}", value)
                    attribute_count += 1
                elif isinstance(value, list):
                    # Handle list values by converting to string
                    span.set_attribute(f"custom.{key}", str(value))
                    attribute_count += 1
                elif isinstance(value, dict):
                    # Handle nested dict by flattening
                    for nested_key, nested_value in value.items():
                        if isinstance(nested_value, (str, int, float, bool)):
                            span.set_attribute(f"custom.{key}.{nested_key}", nested_value)
                            attribute_count += 1
            
            # Add real performance and operation metadata
            processing_time_ms = (time.time() - start_time) * 1000
            
            # Add span attributes that will be visible in OpenSearch Dashboard
            span.set_attribute("otel.operation.type", "metadata_injection")
            span.set_attribute("otel.processing_time_ms", processing_time_ms)
            span.set_attribute("otel.attributes_added", attribute_count)
            span.set_attribute("otel.service_name", SERVICE_NAME)
            span.set_attribute("otel.timestamp", int(time.time()))
            
            # Get current system metrics
            try:
                memory = psutil.virtual_memory()
                cpu_percent = psutil.cpu_percent(interval=0.1)
                span.set_attribute("system.memory_usage_mb", memory.used / (1024 * 1024))
                span.set_attribute("system.memory_percent", memory.percent)
                span.set_attribute("system.cpu_percent", cpu_percent)
            except Exception:
                pass
            
            # Record operation metrics
            self.cost_tracker.record_operation(
                "metadata_injection",
                processing_time_ms,
                attributes_added=attribute_count
            )
            
            return {
                "processing_time_ms": processing_time_ms,
                "attributes_added": attribute_count,
                "operation_type": "metadata_injection"
            }
            
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.set_attribute("error.message", str(e))
            span.set_attribute("otel.operation.status", "error")
            
            # Record error metrics
            self.cost_tracker.record_operation(
                "metadata_injection_error",
                (time.time() - start_time) * 1000,
                error_type=type(e).__name__
            )
            
            return {
                "processing_time_ms": (time.time() - start_time) * 1000,
                "error": str(e),
                "operation_type": "metadata_injection_error"
            }
    
    def inject_business_context(self, span, user_id: Optional[int] = None, 
                              session_id: Optional[str] = None,
                              feature_flags: Optional[Dict[str, bool]] = None) -> Dict[str, Any]:
        """Inject business context metadata"""
        metadata = {
            "context_type": "business",
            "timestamp": int(time.time())
        }
        
        if user_id:
            metadata["user_id"] = user_id
            span.set_attribute("business.user_id", user_id)
        if session_id:
            metadata["session_id"] = session_id
            span.set_attribute("business.session_id", session_id)
        if feature_flags:
            metadata["feature_flags"] = feature_flags
            # Add individual feature flags as span attributes
            for flag, enabled in feature_flags.items():
                span.set_attribute(f"feature_flag.{flag}", enabled)
        
        span.set_attribute("business.context_injected", True)
        
        return self.inject_custom_metadata(span, metadata)
    
    def inject_performance_metadata(self, span, execution_time_ms: float,
                                   memory_usage_mb: Optional[float] = None,
                                   cpu_usage_percent: Optional[float] = None) -> Dict[str, Any]:
        """Inject performance-related metadata"""
        metadata = {
            "context_type": "performance",
            "execution_time_ms": execution_time_ms,
        }
        
        # Add performance attributes directly to span for OpenSearch visibility
        span.set_attribute("performance.execution_time_ms", execution_time_ms)
        
        if memory_usage_mb:
            metadata["memory_usage_mb"] = memory_usage_mb
            span.set_attribute("performance.memory_usage_mb", memory_usage_mb)
        if cpu_usage_percent:
            metadata["cpu_usage_percent"] = cpu_usage_percent
            span.set_attribute("performance.cpu_usage_percent", cpu_usage_percent)
        
        span.set_attribute("performance.tracking_enabled", True)
        
        return self.inject_custom_metadata(span, metadata)

metadata_injector = MetadataInjector(cost_tracker)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": SERVICE_NAME}

@app.get("/products")
async def get_products():
    return list(products_db.values())

def calculate_user_score(user_preferences: dict = None):
    """Calculate a user score for personalized recommendations"""
    start_time = time.time()
    with tracer.start_as_current_span("calculate_user_score") as span:
        
        span.set_attribute("user.has_preferences", bool(user_preferences))
        span.set_attribute("operation.type", "user_scoring")
        span.set_attribute("algorithm.name", "preference_based")
        span.set_attribute("algorithm.version", "1.0")
        
        # Inject custom metadata for user scoring
        metadata_injector.inject_custom_metadata(span, {
            "operation_type": "user_scoring",
            "algorithm": "preference_based",
            "version": "1.0"
        })
        
        # Simulate some computation
        base_score = 100
        if user_preferences:
            preference_bonus = len(user_preferences.get("categories", [])) * 10
            span.set_attribute("user.preference_bonus", preference_bonus)
            span.set_attribute("user.categories_count", len(user_preferences.get("categories", [])))
            
            # Inject business context
            metadata_injector.inject_business_context(span, 
                feature_flags={"personalization_enabled": True})
            
            execution_time = (time.time() - start_time) * 1000
            metadata_injector.inject_performance_metadata(span, execution_time)
            
            # Record operation metrics
            cost_tracker.record_operation("user_scoring", execution_time, 
                                        has_preferences=True, 
                                        preference_bonus=preference_bonus)
            
            return base_score + preference_bonus
        
        execution_time = (time.time() - start_time) * 1000
        metadata_injector.inject_performance_metadata(span, execution_time)
        
        # Record operation metrics
        cost_tracker.record_operation("user_scoring", execution_time, 
                                    has_preferences=False)
        
        return base_score

def filter_products_by_category(products: list, category: str = None):
    """Filter products by category"""
    with tracer.start_as_current_span("filter_products_by_category") as span:
        span.set_attribute("filter.category", category or "all")
        span.set_attribute("products.count_before", len(products))
        
        if category:
            filtered = [p for p in products if p.category == category]
        else:
            filtered = products
        
        span.set_attribute("products.count_after", len(filtered))
        return filtered

def apply_pricing_logic(products: list, user_score: int):
    """Apply pricing logic based on user score"""
    with tracer.start_as_current_span("apply_pricing_logic") as span:
        span.set_attribute("user.score", user_score)
        span.set_attribute("products.count", len(products))
        
        # Simulate pricing adjustments
        for product in products:
            if user_score > 120:
                product.price = product.price * 0.9  # 10% discount for high score users
        
        return products

async def enrich_with_inventory(products: list):
    """Enrich products with inventory information"""
    start_time = time.time()
    with tracer.start_as_current_span("enrich_with_inventory") as span:
        
        span.set_attribute("products.count", len(products))
        span.set_attribute("operation.type", "data_enrichment")
        span.set_attribute("enrichment.source", "inventory_service")
        span.set_attribute("enrichment.batch_size", len(products))
        
        # Inject custom metadata for enrichment operation
        metadata_injector.inject_custom_metadata(span, {
            "operation_type": "data_enrichment",
            "enrichment_source": "inventory_service",
            "batch_size": len(products)
        })
        
        enriched_products = []
        headers = {}
        inject(headers)
        http_requests_made = 0
        
        async with httpx.AsyncClient() as client:
            for product in products:
                try:
                    request_start = time.time()
                    
                    response = await client.get(
                        f"{INVENTORY_SERVICE_URL}/inventory/{product.id}",
                        headers=headers
                    )
                    
                    request_duration = (time.time() - request_start) * 1000
                    http_requests_made += 1
                    
                    # Record HTTP request metrics
                    cost_tracker.record_http_request(
                        "GET", 
                        f"/inventory/{product.id}", 
                        response.status_code, 
                        request_duration
                    )
                    
                    if response.status_code == 200:
                        inventory_data = response.json()
                        product_dict = product.model_dump()
                        product_dict.update({
                            "available_quantity": inventory_data["available_quantity"],
                            "warehouse_location": inventory_data["warehouse_location"],
                            "estimated_shipping_days": inventory_data["estimated_shipping_days"]
                        })
                        enriched_products.append(product_dict)
                    else:
                        enriched_products.append(product.model_dump())
                except Exception as e:
                    enriched_products.append(product.model_dump())
                    span.set_attribute(f"error.product_{product.id}", str(e))
        
        successful_calls = len([p for p in enriched_products if "available_quantity" in p])
        span.set_attribute("enrichment.successful_calls", successful_calls)
        span.set_attribute("enrichment.failed_calls", len(products) - successful_calls)
        span.set_attribute("enrichment.success_rate", successful_calls / len(products) if products else 0)
        span.set_attribute("http.requests_made", http_requests_made)
        
        # Inject performance metadata
        execution_time = (time.time() - start_time) * 1000
        metadata_injector.inject_performance_metadata(span, execution_time)
        
        # Record operation metrics
        cost_tracker.record_operation("data_enrichment", execution_time,
                                    products_processed=len(products),
                                    successful_enrichments=successful_calls,
                                    http_requests=http_requests_made)
        
        return enriched_products

@app.get("/products/recommend")
async def recommend_products(request: Request, category: str = None, user_id: int = None):
    start_time = time.time()
    with tracer.start_as_current_span("recommend_products") as span:
        
        parent_context = extract(dict(request.headers))
        
        span.set_attribute("request.category", category or "all")
        span.set_attribute("request.user_id", user_id or 0)
        span.set_attribute("operation.type", "product_recommendation")
        span.set_attribute("endpoint.name", "/products/recommend")
        
        # Inject custom metadata for recommendation operation
        session_id = request.headers.get("x-session-id", "unknown")
        metadata_injector.inject_business_context(span, 
            user_id=user_id, 
            session_id=session_id,
            feature_flags={
                "recommendation_engine": True,
                "inventory_enrichment": True,
                "real_metrics_tracking": True
            })
        
        # Get all products
        all_products = list(products_db.values())
        span.set_attribute("products.total_available", len(all_products))
        
        # Calculate user score
        user_preferences = {"categories": ["electronics"]} if user_id else None
        user_score = calculate_user_score(user_preferences)
        span.set_attribute("recommendation.user_score", user_score)
        
        # Filter products by category
        filtered_products = filter_products_by_category(all_products, category)
        span.set_attribute("products.after_filter", len(filtered_products))
        
        # Apply pricing logic
        priced_products = apply_pricing_logic(filtered_products, user_score)
        
        # Enrich with inventory information
        final_products = await enrich_with_inventory(priced_products)
        span.set_attribute("products.final_count", len(final_products))
        
        # Add final performance metadata
        execution_time = (time.time() - start_time) * 1000
        span.set_attribute("performance.total_duration_ms", execution_time)
        metadata_injector.inject_performance_metadata(span, execution_time)
        
        # Record operation metrics
        cost_tracker.record_operation("product_recommendation", execution_time,
                                    user_id=user_id or 0,
                                    category=category or "all",
                                    products_returned=len(final_products),
                                    user_score=user_score)
        
        return {
            "products": final_products,
            "user_score": user_score,
            "total_count": len(final_products),
            "metadata": {
                "operation_duration_ms": execution_time,
                "session_id": session_id,
                "timestamp": int(time.time()),
                "service": SERVICE_NAME
            }
        }

@app.get("/telemetry/metrics-info")
async def get_metrics_info():
    """Get information about OpenTelemetry metrics being tracked"""
    start_time = time.time()
    with tracer.start_as_current_span("get_metrics_info") as span:
        
        span.set_attribute("operation.type", "metrics_info")
        span.set_attribute("endpoint.name", "/telemetry/metrics-info")
        
        # Inject metadata for metrics info operation
        metadata_injector.inject_custom_metadata(span, {
            "operation_type": "metrics_info",
            "purpose": "telemetry_monitoring"
        })
        
        # Get current system metrics
        try:
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            system_metrics = {
                "memory_usage_mb": memory.used / (1024 * 1024),
                "memory_percent": memory.percent,
                "cpu_percent": cpu_percent,
                "total_memory_mb": memory.total / (1024 * 1024)
            }
            
            # Add system metrics as span attributes
            for key, value in system_metrics.items():
                span.set_attribute(f"system.{key}", value)
                
        except Exception as e:
            system_metrics = {"error": str(e)}
            span.set_attribute("system.metrics_error", str(e))
        
        execution_time = (time.time() - start_time) * 1000
        metadata_injector.inject_performance_metadata(span, execution_time)
        
        # Record operation metrics
        cost_tracker.record_operation("metrics_info", execution_time)
        
        return {
            "service_info": {
                "service_name": SERVICE_NAME,
                "version": "1.0.0",
                "environment": os.getenv("ENVIRONMENT", "development"),
                "otel_endpoint": OTEL_ENDPOINT
            },
            "metrics_configuration": {
                "metrics_enabled": True,
                "traces_enabled": True,
                "export_interval_ms": 5000,
                "collectors": ["OpenSearch", "Prometheus"]
            },
            "tracked_metrics": [
                "otel_operations_total - Counter of OpenTelemetry operations",
                "otel_span_duration_ms - Histogram of span durations",
                "http_requests_total - Counter of HTTP requests",
                "memory_usage_mb - Gauge of memory usage",
                "cpu_usage_percent - Gauge of CPU usage"
            ],
            "system_metrics": system_metrics,
            "metadata": {
                "tracking_enabled": True,
                "real_metrics": True,
                "dashboard_visible": True,
                "timestamp": int(time.time()),
                "operation_duration_ms": execution_time
            }
        }

@app.get("/telemetry/dashboard-queries")
async def get_dashboard_queries():
    """Get sample queries for OpenSearch Dashboard to visualize the metrics"""
    start_time = time.time()
    with tracer.start_as_current_span("get_dashboard_queries") as span:
        
        span.set_attribute("operation.type", "dashboard_queries")
        span.set_attribute("endpoint.name", "/telemetry/dashboard-queries")
        
        metadata_injector.inject_custom_metadata(span, {
            "operation_type": "dashboard_queries",
            "purpose": "dashboard_configuration"
        })
        
        execution_time = (time.time() - start_time) * 1000
        metadata_injector.inject_performance_metadata(span, execution_time)
        
        cost_tracker.record_operation("dashboard_queries", execution_time)
        
        return {
            "opensearch_dashboard_queries": {
                "traces_with_custom_metadata": {
                    "index": "otel-traces",
                    "query": {
                        "bool": {
                            "must": [
                                {"exists": {"field": "custom.operation_type"}},
                                {"term": {"resource.attributes.service.name": SERVICE_NAME}}
                            ]
                        }
                    },
                    "description": "Find all traces with custom metadata from this service"
                },
                "performance_metrics": {
                    "index": "otel-traces", 
                    "query": {
                        "bool": {
                            "must": [
                                {"exists": {"field": "performance.execution_time_ms"}},
                                {"range": {"performance.execution_time_ms": {"gte": 0}}}
                            ]
                        }
                    },
                    "description": "Find traces with performance metadata"
                },
                "business_context_traces": {
                    "index": "otel-traces",
                    "query": {
                        "bool": {
                            "must": [
                                {"exists": {"field": "business.user_id"}},
                                {"term": {"business.context_injected": True}}
                            ]
                        }
                    },
                    "description": "Find traces with business context (user sessions)"
                },
                "http_request_metrics": {
                    "index": "otel-metrics",
                    "query": {
                        "bool": {
                            "must": [
                                {"term": {"metric_name": "http_requests_total"}},
                                {"term": {"resource.attributes.service.name": SERVICE_NAME}}
                            ]
                        }
                    },
                    "description": "Get HTTP request metrics from this service"
                }
            },
            "dashboard_visualizations": {
                "request_duration_histogram": "Use otel_span_duration_ms metric to create latency histograms",
                "operation_count_timeseries": "Use otel_operations_total to track operation counts over time", 
                "system_metrics_gauges": "Use memory_usage_mb and cpu_usage_percent for system monitoring",
                "error_rate_tracking": "Filter traces by span status to calculate error rates"
            },
            "sample_filters": {
                "by_user": "business.user_id: [USER_ID]",
                "by_operation": "custom.operation_type: [OPERATION_TYPE]", 
                "by_duration": "performance.execution_time_ms: [MIN TO MAX]",
                "by_status": "span.status: ERROR or OK"
            }
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)