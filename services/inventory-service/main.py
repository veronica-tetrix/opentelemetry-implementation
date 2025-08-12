import os
import random
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import Dict, List
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.propagate import extract

SERVICE_NAME = os.getenv("SERVICE_NAME", "inventory-service")
OTEL_ENDPOINT = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")

resource = Resource.create({"service.name": SERVICE_NAME})
trace.set_tracer_provider(TracerProvider(resource=resource))
tracer = trace.get_tracer(__name__)

otlp_exporter = OTLPSpanExporter(endpoint=OTEL_ENDPOINT, insecure=True)
span_processor = BatchSpanProcessor(otlp_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)

app = FastAPI(title="Inventory Service")
FastAPIInstrumentor.instrument_app(app)

class InventoryItem(BaseModel):
    product_id: int
    quantity: int
    reserved: int
    warehouse_location: str

inventory_db = {
    1: InventoryItem(product_id=1, quantity=50, reserved=5, warehouse_location="A-1"),
    2: InventoryItem(product_id=2, quantity=120, reserved=10, warehouse_location="B-2"),
    3: InventoryItem(product_id=3, quantity=200, reserved=0, warehouse_location="C-1"),
}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": SERVICE_NAME}

def check_warehouse_capacity(warehouse_location: str):
    """Check warehouse capacity for a given location"""
    with tracer.start_as_current_span("check_warehouse_capacity") as span:
        span.set_attribute("warehouse.location", warehouse_location)
        # simulate warehouse capacity check
        capacity = random.randint(80, 100)
        span.set_attribute("warehouse.capacity_percent", capacity)
        return capacity

def calculate_shipping_time(warehouse_location: str):
    """Calculate estimated shipping time based on warehouse location"""
    with tracer.start_as_current_span("calculate_shipping_time") as span:
        span.set_attribute("warehouse.location", warehouse_location)
        # simulate shipping time calculation
        shipping_days = {"A": 1, "B": 2, "C": 3}.get(warehouse_location[0], 5)
        span.set_attribute("shipping.estimated_days", shipping_days)
        return shipping_days

@app.get("/inventory/{product_id}")
async def get_inventory(product_id: int, request: Request):
    with tracer.start_as_current_span("get_inventory") as span:
        parent_context = extract(dict(request.headers))
        span.set_attribute("product.id", product_id)
        
        if product_id not in inventory_db:
            span.set_attribute("inventory.found", False)
            raise HTTPException(status_code=404, detail="Product not found in inventory")
        
        inventory = inventory_db[product_id]
        span.set_attribute("inventory.found", True)
        span.set_attribute("inventory.quantity", inventory.quantity)
        span.set_attribute("inventory.reserved", inventory.reserved)
        
        # check warehouse capacity
        capacity = check_warehouse_capacity(inventory.warehouse_location)
        
        # calculate shipping time
        shipping_days = calculate_shipping_time(inventory.warehouse_location)
        
        return {
            "product_id": inventory.product_id,
            "available_quantity": inventory.quantity - inventory.reserved,
            "total_quantity": inventory.quantity,
            "warehouse_location": inventory.warehouse_location,
            "warehouse_capacity_percent": capacity,
            "estimated_shipping_days": shipping_days
        }

@app.post("/inventory/{product_id}/reserve")
async def reserve_inventory(product_id: int, quantity: int, request: Request):
    with tracer.start_as_current_span("reserve_inventory") as span:
        parent_context = extract(dict(request.headers))
        span.set_attribute("product.id", product_id)
        span.set_attribute("reservation.quantity", quantity)
        
        if product_id not in inventory_db:
            span.set_attribute("reservation.success", False)
            raise HTTPException(status_code=404, detail="Product not found in inventory")
        
        inventory = inventory_db[product_id]
        available = inventory.quantity - inventory.reserved
        
        if available < quantity:
            span.set_attribute("reservation.success", False)
            span.set_attribute("inventory.available", available)
            raise HTTPException(status_code=400, detail=f"Insufficient inventory. Available: {available}")
        
        # reserve the inventory
        inventory.reserved += quantity
        span.set_attribute("reservation.success", True)
        span.set_attribute("inventory.new_reserved", inventory.reserved)

        
        return {
            "success": True,
            "reserved_quantity": quantity,
            "remaining_available": inventory.quantity - inventory.reserved
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)