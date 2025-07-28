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
    return {"status": "healthy", "service": SERVICE_NAME}

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, log_config=None)