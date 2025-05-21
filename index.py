from fastapi import FastAPI, HTTPException, Body, UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
import torch
from sklearn.metrics.pairwise import cosine_similarity
import io
import json
import os
import uuid
import re
import string
from qdrant_client import QdrantClient
from qdrant_client.http import models
import pathlib

app = FastAPI(title="Semantic Search API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Global variables
bi_encoder = None
cross_encoder = None
qdrant_client = None
collection_name = "semantic_search"

# Configuration
VECTOR_SIZE = 384  

# Local Qdrant configuration
# Path for local Qdrant storage
QDRANT_PATH = os.getenv("QDRANT_PATH", os.path.join(os.path.expanduser("~"), "qdrant_data"))

# Models
class Query(BaseModel):
    text: str
    use_cross_encoder: bool = True  # Option to use cross-encoder for re-ranking
    candidate_count: int = 100  # Number of candidates to consider for cross-encoder re-ranking

class DatasetInfo(BaseModel):
    col1_name: str
    col2_name: str
    
class SearchResult(BaseModel):
    query: str
    result: str
    similarity_score: float
    top_matches: List[Dict[str, Any]]
    method_used: str  # Indicate which method was used (bi-encoder only or with cross-encoder)

# Dependency for Qdrant client
def get_qdrant_client():
    global qdrant_client
    if qdrant_client is None:
        try:
            # Create directory for Qdrant storage if it doesn't exist
            pathlib.Path(QDRANT_PATH).mkdir(parents=True, exist_ok=True)
            
            # Initialize local Qdrant client
            qdrant_client = QdrantClient(path=QDRANT_PATH)
            print(f"Connected to local Qdrant at {QDRANT_PATH}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to connect to local Qdrant: {str(e)}")
    return qdrant_client

def clean_text(text):
    """
    Comprehensive text cleaning function to handle various edge cases.
    """
    if pd.isna(text) or text is None:
        return ""
    
    # Convert to string if not already
    text = str(text)
    
    # Strip whitespace
    text = text.strip()
    
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    
    # Remove control characters
    text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)
    
    # Replace problematic characters
    text = text.replace('\t', ' ').replace('\n', ' ').replace('\r', ' ')
    
    return text

@app.on_event("startup")
async def load_model():
    global bi_encoder, cross_encoder, qdrant_client
    bi_encoder = SentenceTransformer('all-MiniLM-L6-v2')
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    try:
        pathlib.Path(QDRANT_PATH).mkdir(parents=True, exist_ok=True)
        
        qdrant_client = QdrantClient(path=QDRANT_PATH)
        
        collections = qdrant_client.get_collections().collections
        collection_exists = any(collection.name == collection_name for collection in collections)
        
        if not collection_exists:
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=VECTOR_SIZE,
                    distance=models.Distance.COSINE
                )
            )
            print(f"Created local Qdrant collection: {collection_name}")
        else:
            print(f"Using existing Qdrant collection: {collection_name}")
            
        print(f"Connected to local Qdrant at {QDRANT_PATH}")
    except Exception as e:
        print(f"Error connecting to local Qdrant: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to initialize Qdrant: {str(e)}")
    
    print("Models loaded successfully")

@app.get("/")
async def root():
    return {"message": "Semantic Search API is running with local Qdrant storage"}

@app.get("/status")
async def status(qdrant: QdrantClient = Depends(get_qdrant_client)):
    global bi_encoder, cross_encoder
    
    qdrant_status = "connected"
    qdrant_collection_count = 0
    qdrant_storage_path = QDRANT_PATH
    
    try:
        collections = qdrant.get_collections().collections
        collection_exists = any(collection.name == collection_name for collection in collections)
        if collection_exists:
            collection_info = qdrant.get_collection(collection_name=collection_name)
            qdrant_collection_count = collection_info.points_count
    except Exception as e:
        qdrant_status = f"error: {str(e)}"
    
    return {
        "bi_encoder_loaded": bi_encoder is not None,
        "cross_encoder_loaded": cross_encoder is not None,
        "qdrant_status": qdrant_status,
        "qdrant_collection_count": qdrant_collection_count,
        "qdrant_storage_path": qdrant_storage_path
    }

def preprocess_dataframe(df, col1_name, col2_name):
    """
    Preprocess the dataframe to ensure data quality and handle edge cases.
    """
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Ensure the required columns exist
    if col1_name not in df.columns or col2_name not in df.columns:
        raise ValueError(f"Required columns not found in dataset: {col1_name}, {col2_name}")
    
    # Apply cleaning to both columns
    df[col1_name] = df[col1_name].apply(clean_text)
    df[col2_name] = df[col2_name].apply(clean_text)
    
    # Remove rows where the first column is empty after cleaning
    df = df[df[col1_name] != ""]
    
    # Handle duplicates
    df_deduplicated = df.drop_duplicates(subset=[col1_name])
    
    # Log any changes made
    dropped_empty = len(df) - len(df[df[col1_name] != ""])
    dropped_duplicates = len(df) - len(df_deduplicated)
    
    preprocessing_stats = {
        "original_rows": len(df),
        "empty_rows_removed": dropped_empty,
        "duplicate_rows_removed": dropped_duplicates,
        "final_rows": len(df_deduplicated)
    }
    
    return df_deduplicated, preprocessing_stats

@app.post("/upload")
async def upload_data(
    file: UploadFile = File(...), 
    col1_name: str = None, 
    col2_name: str = None,
    qdrant: QdrantClient = Depends(get_qdrant_client)
):
    global bi_encoder
    
    # Handle different file formats
    try:
        if file.filename.endswith('.csv'):
            content = await file.read()
            df = pd.read_csv(io.BytesIO(content))
        elif file.filename.endswith(('.xlsx', '.xls')):
            content = await file.read()
            df = pd.read_excel(io.BytesIO(content))
        elif file.filename.endswith('.json'):
            content = await file.read()
            df = pd.DataFrame(json.loads(content.decode('utf-8')))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Please upload CSV, Excel, or JSON.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading file: {str(e)}")
    
    # Auto-detect columns if not provided
    if col1_name is None:
        col1_name = df.columns[0]
    if col2_name is None:
        col2_name = df.columns[1] if len(df.columns) > 1 else df.columns[0]
    
    if col1_name not in df.columns or col2_name not in df.columns:
        raise HTTPException(status_code=400, detail=f"Specified columns not found in dataset: {col1_name}, {col2_name}")
    
    try:
        # Preprocess the dataframe
        df_clean, preprocessing_stats = preprocess_dataframe(df, col1_name, col2_name)
        
        # Recreate the Qdrant collection
        try:
            collections = qdrant.get_collections().collections
            collection_exists = any(collection.name == collection_name for collection in collections)
            if collection_exists:
                qdrant.delete_collection(collection_name=collection_name)
        except Exception as e:
            print(f"Error deleting collection: {str(e)}")
        
        qdrant.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=VECTOR_SIZE,
                distance=models.Distance.COSINE
            )
        )
        
        # Process in batches to avoid memory issues
        batch_size = 100
        total_points = 0
        skipped_points = 0
        
        for i in range(0, len(df_clean), batch_size):
            batch = df_clean.iloc[i:i+batch_size]
            batch_col1 = batch[col1_name].tolist()
            batch_col2 = batch[col2_name].tolist()
            
            # Skip empty batches
            if not batch_col1:
                continue
                
            try:
                batch_embeddings = bi_encoder.encode(batch_col1)
                
                points = []
                for j in range(len(batch_col1)):
                    if batch_col1[j]:  # Additional check to ensure non-empty text
                        points.append(models.PointStruct(
                            id=str(uuid.uuid4()),
                            vector=batch_embeddings[j].tolist(),
                            payload={
                                "text": batch_col1[j],
                                "result": batch_col2[j] if j < len(batch_col2) and batch_col2[j] else ""
                            }
                        ))
                    else:
                        skipped_points += 1
                
                if points:
                    qdrant.upload_points(
                        collection_name=collection_name,
                        points=points
                    )
                    total_points += len(points)
            except Exception as e:
                print(f"Error processing batch {i//batch_size}: {str(e)}")
                # Continue with next batch instead of failing completely
                skipped_points += len(batch_col1)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing and storing data: {str(e)}")
    
    return {
        "status": "success", 
        "message": f"Uploaded dataset with {total_points} records",
        "columns": {
            "col1": col1_name,
            "col2": col2_name
        },
        "preprocessing_stats": preprocessing_stats,
        "storage_details": {
            "qdrant_total_points": total_points,
            "skipped_points": skipped_points,
            "qdrant_path": QDRANT_PATH
        }
    }

@app.post("/search", response_model=SearchResult)
async def search(
    query: Query, 
    top_k: int = 5,
    qdrant: QdrantClient = Depends(get_qdrant_client)
):
    global bi_encoder, cross_encoder
    
    if bi_encoder is None:
        raise HTTPException(status_code=500, detail="Bi-encoder model not loaded")
    
    # Clean and normalize the query text
    cleaned_query = clean_text(query.text)
    if not cleaned_query:
        raise HTTPException(status_code=400, detail="Query text is empty after cleaning")
    
    # Encode the query
    try:
        query_embedding = bi_encoder.encode(cleaned_query, convert_to_tensor=True)
        query_embedding_np = query_embedding.cpu().numpy()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error encoding query: {str(e)}")
    
    try:
        # Check if collection exists
        collections = qdrant.get_collections().collections
        collection_exists = any(collection.name == collection_name for collection in collections)
        if not collection_exists:
            raise HTTPException(status_code=400, detail="No Qdrant collection found. Please upload a dataset first.")
        
        collection_info = qdrant.get_collection(collection_name=collection_name)
        if collection_info.points_count == 0:
            raise HTTPException(status_code=400, detail="Qdrant collection is empty. Please upload a dataset first.")
        
        return qdrant_search(query, cleaned_query, query_embedding_np, top_k, qdrant)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching in Qdrant: {str(e)}")

def qdrant_search(query, cleaned_query, query_embedding_np, top_k, qdrant):
    global cross_encoder
    
    # Perform the search
    try:
        search_result = qdrant.search(
            collection_name=collection_name,
            query_vector=query_embedding_np.tolist(),
            limit=query.candidate_count if query.use_cross_encoder else top_k
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Qdrant search error: {str(e)}")
    
    method_used = "bi-encoder"
    
    if query.use_cross_encoder and cross_encoder is not None and search_result:
        method_used = "cross-encoder"
        
        candidates = []
        candidate_texts = []
        
        for hit in search_result:
            candidates.append({
                "text": hit.payload.get("text", ""),
                "result": hit.payload.get("result", ""),
                "similarity": hit.score
            })
            candidate_texts.append(hit.payload.get("text", ""))
        
        if candidates:
            try:
                sentence_pairs = [[cleaned_query, text] for text in candidate_texts]
                cross_scores = cross_encoder.predict(sentence_pairs)
                
                for i, score in enumerate(cross_scores):
                    candidates[i]["similarity"] = float(score)
                
                candidates.sort(key=lambda x: x["similarity"], reverse=True)
                
                top_candidates = candidates[:top_k]
                top_scores = [c["similarity"] for c in top_candidates]
                
                # Normalize scores if outside 0-1 range
                if top_scores and (np.max(top_scores) > 1.0 or np.min(top_scores) < 0.0):
                    min_score = np.min(top_scores)
                    max_score = np.max(top_scores)
                    if max_score > min_score:  
                        for i in range(len(top_candidates)):
                            top_candidates[i]["similarity"] = float((top_candidates[i]["similarity"] - min_score) / (max_score - min_score))
                
                top_matches = top_candidates
            except Exception as e:
                # Fall back to bi-encoder results if cross-encoder fails
                method_used = "bi-encoder (cross-encoder failed)"
                top_matches = [
                    {
                        "text": hit.payload.get("text", ""),
                        "result": hit.payload.get("result", ""),
                        "similarity": hit.score
                    }
                    for hit in search_result[:top_k]
                ]
        else:
            top_matches = []
    else:
        top_matches = [
            {
                "text": hit.payload.get("text", ""),
                "result": hit.payload.get("result", ""),
                "similarity": hit.score
            }
            for hit in search_result[:top_k]
        ]
    
    if top_matches:
        return {
            "query": cleaned_query,
            "result": top_matches[0]["result"],
            "similarity_score": top_matches[0]["similarity"],
            "top_matches": top_matches,
            "method_used": method_used
        }
    else:
        return {
            "query": cleaned_query,
            "result": "",
            "similarity_score": 0.0,
            "top_matches": [],
            "method_used": method_used
        }

@app.get("/dataset-info")
async def get_dataset_info(qdrant: QdrantClient = Depends(get_qdrant_client)):
    try:
        # Check if collection exists
        collections = qdrant.get_collections().collections
        collection_exists = any(collection.name == collection_name for collection in collections)
        if not collection_exists:
            raise HTTPException(status_code=400, detail="No Qdrant collection found")
        
        # Get sample points
        scroll_result = qdrant.scroll(
            collection_name=collection_name,
            limit=5,
            with_payload=True,
            with_vectors=False
        )
        
        points = scroll_result[0]
        
        samples_col1 = [point.payload.get("text", "") for point in points]
        samples_col2 = [point.payload.get("result", "") for point in points]
        
        collection_info = qdrant.get_collection(collection_name=collection_name)
        
        return {
            "storage": "qdrant",
            "storage_path": QDRANT_PATH,
            "num_records": collection_info.points_count,
            "sample_col1": samples_col1,
            "sample_col2": samples_col2
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching data from Qdrant: {str(e)}")

# Mount static files for frontend
try:
    os.makedirs("static", exist_ok=True)
    app.mount("/static", StaticFiles(directory="static"), name="static")
except Exception as e:
    print(f"Warning: Could not mount static files: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8699)