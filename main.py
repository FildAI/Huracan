from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from routes.companies import router as companies_router

# Create FastAPI app
app = FastAPI(
    title="Huracan - Company Search API",
    description="A modern vector search engine for company data using Qdrant and multi-embedding fusion",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(companies_router)

# Health check endpoint
@app.get("/")
async def root():
    """Root endpoint with basic API information."""
    return {
        "message": "Huracan Company Search API",
        "version": "0.1.0",
        "docs": "/docs"
    }

@app.get("/health")
async def health():
    """Global health check endpoint."""
    return {"status": "healthy", "service": "huracan-api"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True) 