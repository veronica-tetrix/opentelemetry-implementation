import os
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# import the OpenTelemetry libraries
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.propagate import inject

SERVICE_NAME = os.getenv("SERVICE_NAME", "user-service")
OTEL_ENDPOINT = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
PRODUCT_SERVICE_URL = os.getenv("PRODUCT_SERVICE_URL", "http://localhost:8000")
INVENTORY_SERVICE_URL = os.getenv("INVENTORY_SERVICE_URL", "http://localhost:8002")

# create a tracer and configure it to send traces to our collector using the OTLP protocol
resource = Resource.create({"service.name": SERVICE_NAME})
trace.set_tracer_provider(TracerProvider(resource=resource))
tracer = trace.get_tracer(__name__)

otlp_exporter = OTLPSpanExporter(endpoint=OTEL_ENDPOINT, insecure=True)
span_processor = BatchSpanProcessor(otlp_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)

# create a FastAPI application and instrument it with OpenTelemetry
app = FastAPI(title="User Service")
FastAPIInstrumentor.instrument_app(app)
HTTPXClientInstrumentor().instrument()

class User(BaseModel):
    id: int
    name: str
    email: str

users_db = {
    1: User(id=1, name="John Doe", email="john@example.com"),
    2: User(id=2, name="Jane Smith", email="jane@example.com"),
}

@app.get("/health")
async def health_check():
    # return {"status": "healthy", "service": SERVICE_NAME}
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{PRODUCT_SERVICE_URL}/health")
            product_service_healthy = response.status_code == 200
    except:
        product_service_healthy = False

    database_healthy = len(users_db) > 0
    overall_healthy = product_service_healthy and database_healthy
    
    return {
        "status": "healthy" if overall_healthy else "unhealthy",
        "service": SERVICE_NAME,
        "dependencies": {
            "product_service": "healthy" if product_service_healthy else "unhealthy",
            "database": "healthy" if database_healthy else "unhealthy"
        }
    }

@app.get("/users")
async def get_users():
    return list(users_db.values())

@app.get("/users/{user_id}")
async def get_user(user_id: int):
    if user_id not in users_db:
        raise HTTPException(status_code=404, detail="User not found")
    return users_db[user_id]

# when someone requests recommendations, we create a manual span and call the Product Service
@app.get("/users/{user_id}/recommendations")
async def get_user_recommendations(user_id: int):
    with tracer.start_as_current_span("get_user_recommendations") as span:
        if user_id not in users_db:
            raise HTTPException(status_code=404, detail="User not found")
        
        user = users_db[user_id]
        
        headers = {}
        inject(headers) # passes the trace context to the next service
        
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{PRODUCT_SERVICE_URL}/products/recommend",
                headers=headers
            )
            products = response.json()
        
        return {"user": user, "products": products}

async def check_user_preferences(user_id: int):
    """Check user's purchase history and preferences"""
    with tracer.start_as_current_span("check_user_preferences") as span:
        span.set_attribute("user.id", user_id)
        # Simulate preference analysis
        preferences = {"preferred_categories": ["electronics"], "budget_range": "high"}
        span.set_attribute("preferences.categories", len(preferences["preferred_categories"]))
        return preferences

async def validate_inventory_availability(product_ids: list):
    """Validate inventory availability for multiple products"""
    with tracer.start_as_current_span("validate_inventory_availability") as span:
        span.set_attribute("products.count", len(product_ids))
        
        headers = {}
        inject(headers)
        
        availability_results = []
        async with httpx.AsyncClient() as client:
            for product_id in product_ids:
                try:
                    response = await client.get(
                        f"{INVENTORY_SERVICE_URL}/inventory/{product_id}",
                        headers=headers
                    )
                    if response.status_code == 200:
                        inventory_data = response.json()
                        availability_results.append({
                            "product_id": product_id,
                            "available": inventory_data["available_quantity"] > 0,
                            "quantity": inventory_data["available_quantity"]
                        })
                    else:
                        availability_results.append({
                            "product_id": product_id,
                            "available": False,
                            "quantity": 0
                        })
                except Exception:
                    availability_results.append({
                        "product_id": product_id,
                        "available": False,
                        "quantity": 0
                    })
        
        available_count = len([r for r in availability_results if r["available"]])
        span.set_attribute("inventory.available_products", available_count)
        span.set_attribute("inventory.total_checked", len(availability_results))
        
        return availability_results

@app.get("/users/{user_id}/shopping-analysis")
async def get_shopping_analysis(user_id: int):
    """Complex endpoint that calls multiple services and functions"""
    with tracer.start_as_current_span("get_shopping_analysis") as span:
        if user_id not in users_db:
            raise HTTPException(status_code=404, detail="User not found")
        
        user = users_db[user_id]
        span.set_attribute("user.id", user_id)
        span.set_attribute("user.name", user.name)
        
        # Step 1: Check user preferences
        preferences = await check_user_preferences(user_id)
        
        # Step 2: Get product recommendations
        headers = {}
        inject(headers)
        
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{PRODUCT_SERVICE_URL}/products/recommend?user_id={user_id}",
                headers=headers
            )
            recommendations = response.json()
        
        # Step 3: Validate inventory for recommended products
        product_ids = [p["id"] for p in recommendations["products"]]
        inventory_check = await validate_inventory_availability(product_ids)
        
        # Step 4: Calculate shopping score
        with tracer.start_as_current_span("calculate_shopping_score") as span:
            available_products = [r for r in inventory_check if r["available"]]
            shopping_score = len(available_products) * 10 + recommendations["user_score"]
            span.set_attribute("shopping.score", shopping_score)
            span.set_attribute("shopping.available_products", len(available_products))
        
        return {
            "user": user,
            "preferences": preferences,
            "recommendations": recommendations,
            "inventory_status": inventory_check,
            "shopping_score": shopping_score,
            "analysis_summary": {
                "total_recommendations": len(recommendations["products"]),
                "available_products": len(available_products),
                "user_score": recommendations["user_score"],
                "final_shopping_score": shopping_score
            }
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, log_config=None)