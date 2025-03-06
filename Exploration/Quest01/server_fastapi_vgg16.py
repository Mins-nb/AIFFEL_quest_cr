import uvicorn
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image
import numpy as np
from PIL import Image
import io
import os

# FastAPI 애플리케이션 초기화
app = FastAPI()

# CORS 설정 (필요시 다른 도메인에서 요청 허용)
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 모델 로드
MODEL_PATH = "flower_model.h5"  # 모델 파일 경로
model = tf.keras.models.load_model(MODEL_PATH)

# 클래스 레이블 (5개의 꽃 분류)
class_names = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]

# 이미지 전처리 함수
def prepare_image(image: UploadFile):
    image_data = image.file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    image = image.resize((160, 160))  # 모델의 입력 크기에 맞게 조정
    image_array = np.array(image) / 255.0  # 정규화
    image_array = np.expand_dims(image_array, axis=0)  # 배치 차원 추가
    return image_array

# 예측 함수
def predict_image(image: np.ndarray):
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction, axis=1)
    confidence = np.max(prediction)
    return class_names[predicted_class[0]], confidence

# 기본 루트 URL
@app.get("/")
async def root():
    return {"message": "Flower Classification API"}

# 이미지 업로드 및 예측 엔드포인트
@app.post("/upload/")
async def upload_and_predict(file: UploadFile = File(...)):
    try:
        # 이미지 전처리
        image_array = prepare_image(file)

        # 예측
        predicted_class, confidence = predict_image(image_array)

        return {
            "predicted_class": predicted_class,
            "confidence": round(confidence, 4)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 서버 실행
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
