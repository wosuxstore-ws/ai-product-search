from fastapi import FastAPI, UploadFile, File
from PIL import Image
import open_clip
import torch
import numpy as np
import faiss
import io
from rembg import remove

app = FastAPI()

model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32',
    pretrained='laion2b_s34b_b79k'
)

index = faiss.IndexFlatL2(512)

products = []

def image_embedding(image):

    image = preprocess(image).unsqueeze(0)

    with torch.no_grad():

        embedding = model.encode_image(image)

    embedding /= embedding.norm(dim=-1, keepdim=True)

    return embedding[0].cpu().numpy()

@app.get("/")
def home():

    return {"status":"AI Running"}

@app.post("/add")
async def add_product(
    id:int,
    image:UploadFile = File(...)
):

    contents = await image.read()

    output = remove(contents)

    image_data = Image.open(
        io.BytesIO(output)
    ).convert('RGB')

    emb = image_embedding(image_data)

    index.add(
        np.array([emb]).astype('float32')
    )

    products.append({
        'id':id,
        'embedding':emb
    })

    return {
        'status':'success'
    }

@app.post("/search")
async def search_product(
    image:UploadFile = File(...)
):

    contents = await image.read()

    output = remove(contents)

    image_data = Image.open(
        io.BytesIO(output)
    ).convert('RGB')

    emb = image_embedding(image_data)

    D, I = index.search(
        np.array([emb]).astype('float32'),
        5
    )

    results = []

    for distance, idx in zip(D[0], I[0]):

        if idx == -1:
            continue

        similarity = round(
            (1 - float(distance)) * 100,
            2
        )

        p = products[idx]

        results.append({

            'id':p['id'],

            'score':similarity

        })

    return {
        'results':results
    }
