# Disease Classification of Chest X-ray using CLIP and Few-shot Learning
> MIMIC-CXR 데이터셋과 CLIP을 이용한 구조화된 임상적 소견 예측

## Dataset
- [MIMIC-CXR Database](https://physionet.org/content/mimic-cxr/2.0.0/)
- [MIMIC-CXR-JPG](https://physionet.org/content/mimic-cxr-jpg/2.0.0/)
- [Chest ImaGenome Dataset](https://physionet.org/content/chest-imagenome/1.0.0/) 

## Method
### FlexR (Few-shot classification with Language Embeddings for chest X-ray Reporting)
1단계. [CLIP](https://github.com/openai/CLIP) (Contrastive Language-Image Pretraining)

| encoder       | model       |
|---------------|-------------|
| image encoder | ResNet50    |
| text encoder  | Transformer |

2단계. Language embeddings of clinical findings 

3단계. Fine-tuning the classifier 

## Demo
https://cxr-fsc.streamlit.app/

## Teammate - R&R
Deep Daiv. M3

|<img src="https://www.notion.so/image/https%3A%2F%2Fs3-us-west-2.amazonaws.com%2Fpublic.notion-static.com%2F0c6814c2-cd3b-4772-b745-4a26bec7921b%2Fnotion-avatar-1690384507896.png?width=240&userId=d115cd65-c0ba-4dd5-90d1-29d55b5020b1&cache=v2" width="100">|<img src="https://www.notion.so/image/https%3A%2F%2Fs3-us-west-2.amazonaws.com%2Fpublic.notion-static.com%2Fb7a57a73-1ed8-440d-b7a2-8b1271dba682%2FKakaoTalk_20230120_001023309.jpg?width=240&userId=d115cd65-c0ba-4dd5-90d1-29d55b5020b1&cache=v2" width="100">|<img src="https://www.notion.so/image/https%3A%2F%2Fs3-us-west-2.amazonaws.com%2Fpublic.notion-static.com%2F6798f19c-99ac-46b4-996b-9ee4dbb5efe8%2Fnotion_avartar.png?width=240&userId=d115cd65-c0ba-4dd5-90d1-29d55b5020b1&cache=v2" width="100">|
|:---:|:---:|:---:|
|[정수미](https://github.com/learntosurf)|[김예린](https://github.com/yesyell)|[홍재령](https://github.com/Jar199)|
|1단계|2단계|3단계|
