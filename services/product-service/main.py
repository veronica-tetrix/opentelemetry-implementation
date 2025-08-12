import os
import httpx
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.propagate import extract, inject

SERVICE_NAME = os.getenv("SERVICE_NAME", "product-service")
OTEL_ENDPOINT = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
INVENTORY_SERVICE_URL = os.getenv("INVENTORY_SERVICE_URL", "http://localhost:8002")

resource = Resource.create({"service.name": SERVICE_NAME})
trace.set_tracer_provider(TracerProvider(resource=resource))
tracer = trace.get_tracer(__name__)

otlp_exporter = OTLPSpanExporter(endpoint=OTEL_ENDPOINT, insecure=True)
span_processor = BatchSpanProcessor(otlp_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)

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

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": SERVICE_NAME}

@app.get("/products")
async def get_products():
    return list(products_db.values())

def calculate_user_score(user_preferences: dict = None):
    """Calculate a user score for personalized recommendations"""
    with tracer.start_as_current_span("calculate_user_score") as span:
        span.set_attribute("user.has_preferences", bool(user_preferences))
        # Simulate some computation
        base_score = 100
        if user_preferences:
            preference_bonus = len(user_preferences.get("categories", [])) * 10
            span.set_attribute("user.preference_bonus", preference_bonus)
            return base_score + preference_bonus
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
    with tracer.start_as_current_span("enrich_with_inventory") as span:
        span.set_attribute("products.count", len(products))
        
        enriched_products = []
        headers = {}
        inject(headers)
        
        async with httpx.AsyncClient() as client:
            for product in products:
                try:
                    response = await client.get(
                        f"{INVENTORY_SERVICE_URL}/inventory/{product.id}",
                        headers=headers
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
                except Exception:
                    enriched_products.append(product.model_dump())
        
        span.set_attribute("enrichment.successful_calls", len([p for p in enriched_products if "available_quantity" in p]))
        return enriched_products

@app.get("/products/recommend")
async def recommend_products(request: Request, category: str = None, user_id: int = None):
    with tracer.start_as_current_span("recommend_products") as span:
        parent_context = extract(dict(request.headers)) # receives that trace context, so both services are part of the same trace
        
        span.set_attribute("request.category", category or "all")
        span.set_attribute("request.user_id", user_id or 0)
        
        # Get all products
        all_products = list(products_db.values())
        
        # Calculate user score
        user_preferences = {"categories": ["electronics"]} if user_id else None
        user_score = calculate_user_score(user_preferences)
        
        # Filter products by category
        filtered_products = filter_products_by_category(all_products, category)
        
        # Apply pricing logic
        priced_products = apply_pricing_logic(filtered_products, user_score)
        
        # Enrich with inventory information
        final_products = await enrich_with_inventory(priced_products)
        
        return {
            "products": final_products,
            "user_score": user_score,
            "total_count": len(final_products)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)