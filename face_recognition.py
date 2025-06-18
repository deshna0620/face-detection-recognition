from keras_facenet import FaceNet
import numpy as np
import cv2
from sklearn.metrics.pairwise import cosine_similarity

embedder = FaceNet()

def preprocess(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def get_embedding(img_path):
    img = preprocess(img_path)
    embedding = embedder.embeddings([img])[0]
    return embedding

def compare_faces(path1, path2):
    emb1 = get_embedding(path1)
    emb2 = get_embedding(path2)
    similarity = cosine_similarity([emb1], [emb2])[0][0]
    print(f"Similarity between {path1} and {path2}: {similarity * 100:.2f}%")
    if similarity > 0.98:
        print("Faces Match (above 98% confidence)")
    else:
        print("Faces Do NOT Match (below 98% confidence)")

compare_faces("data/face1.png", "data/face1_masked.png")
compare_faces("data/face2.png", "data/face2_capped.png")