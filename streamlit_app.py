#분류 결과 + 이미지 + 텍스트와 함께 분류 결과에 따라 다른 출력 보여주기
#파일 이름 streamlit_app.py



import streamlit as st
from fastai.vision.all import *
import gdown

# Google Drive에서 모델 다운로드
file_id = '1eDKYoNJ01VhBEc8u0V-NzdvhtO07lSOW?usp=sharing'  # Colab에서 생성한 모델의 파일 ID 입력
url = f'https://drive.google.com/uc?id={file_id}'
output = 'cat_dog_model.pkl'
gdown.download(url, output, quiet=False)

# 모델 로드
learner = load_learner(output)

# 이미지 업로드
uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    img = PILImage.create(uploaded_file)
    prediction, _, probs = learner.predict(img)
    st.image(img, caption="업로드된 이미지", use_column_width=True)
    st.write(f"예측 결과: {prediction}")
    st.write(f"확률: {probs}")
