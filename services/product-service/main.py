import os
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.propagate import extract

SERVICE_NAME = os.getenv("SERVICE_NAME", "product-service")
OTEL_ENDPOINT = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")

resource = Resource.create({"service.name": SERVICE_NAME})
trace.set_tracer_provider(TracerProvider(resource=resource))
tracer = trace.get_tracer(__name__)

otlp_exporter = OTLPSpanExporter(endpoint=OTEL_ENDPOINT, insecure=True)
span_processor = BatchSpanProcessor(otlp_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)

app = FastAPI(title="Product Service")
FastAPIInstrumentor.instrument_app(app)

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

@app.get("/products/recommend")
async def recommend_products(request: Request):
    with tracer.start_as_current_span("recommend_products") as span:
        parent_context = extract(dict(request.headers))
        
        # Simple recommendation for demostration, return all products
        products = list(products_db.values())
        return products

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
