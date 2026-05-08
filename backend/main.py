from fastapi import FastAPI, UploadFile, File, Form
from PIL import Image
import open_clip
import torch
import numpy as np
import faiss
import pickle
import io
import os
from rembg import remove

app = FastAPI()

# =========================
# LOAD MODEL
# =========================

model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32',
    pretrained='openai'
)

# =========================
# LOAD FAISS INDEX
# =========================

INDEX_FILE = "faiss.index"
EMBEDDINGS_FILE = "embeddings.pkl"

if os.path.exists(INDEX_FILE):
    index = faiss.read_index(INDEX_FILE)
else:
    index = faiss.IndexFlatL2(512)

# =========================
# LOAD PRODUCTS
# =========================

if os.path.exists(EMBEDDINGS_FILE):
    with open(EMBEDDINGS_FILE, "rb") as f:
        products = pickle.load(f)
else:
    products = []

# =========================
# IMAGE EMBEDDING FUNCTION
# =========================

def image_embedding(image):

    image = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        embedding = model.encode_image(image)

    embedding /= embedding.norm(dim=-1, keepdim=True)

    return embedding[0].cpu().numpy()

# =========================
# HOME ROUTE
# =========================

@app.get("/")
def home():
    return {
        "status": "running",
        "message": "AI Product Search API Running"
    }

# =========================
# ADD PRODUCT
# =========================

@app.post("/add")
async def add_product(
    id: int = Form(...),
    image: UploadFile = File(...)
):

    contents = await image.read()

    # Remove background
    output = remove(contents)

    image_data = Image.open(
        io.BytesIO(output)
    ).convert("RGB")

    # Generate embedding
    emb = image_embedding(image_data)

    # Add to FAISS
    index.add(
        np.array([emb]).astype("float32")
    )

    # Save product
    products.append({
        "id": id,
        "embedding": emb.tolist()
    })

    # Save index
    faiss.write_index(index, INDEX_FILE)

    # Save embeddings
    with open(EMBEDDINGS_FILE, "wb") as f:
        pickle.dump(products, f)

    return {
        "status": "success",
        "message": "Product added"
    }

# =========================
# SEARCH PRODUCT
# =========================

@app.post("/search")
async def search_product(
    image: UploadFile = File(...)
):

    if len(products) == 0:
        return {
            "results": []
        }

    contents = await image.read()

    # Remove background
    output = remove(contents)

    image_data = Image.open(
        io.BytesIO(output)
    ).convert("RGB")

    # Generate embedding
    emb = image_embedding(image_data)

    # Search FAISS
    D, I = index.search(
        np.array([emb]).astype("float32"),
        10
    )

    results = []

    for distance, idx in zip(D[0], I[0]):

        if idx == -1:
            continue

        similarity = round(
            max(0, (1 - float(distance))) * 100,
            2
        )

        p = products[idx]

        results.append({
            "id": p["id"],
            "score": similarity
        })

    return {
        "results": results
    }
