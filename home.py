import streamlit as st
from PIL import Image
from st_pages import Page, Section, show_pages, add_page_title
from annotated_text import annotated_text

st.set_page_config(layout="wide", page_title="M3 demo", page_icon=":green_apple:")

st.markdown("<h1 style='text-align: center; color: black;'>Few-shot Classification for Chest X-ray diagnosis</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: gray;'>MIMIC-CXR 데이터셋과 CLIP을 이용한 구조화된 임상적 소견 예측</h4>", unsafe_allow_html=True)

st.subheader('Motivation')
st.image(Image.open('main.png'), width=700)
# st.caption('MIMIC 데이터셋과 CLIP을 이용한 구조화된 임상적 소견 예측')
motivation = '''
이 프로젝트는 소량의 X-선 이미지로부터 흉부 질병을 예측 및 분류할 수 있는  모델을 구현하는 것을 목표로 합니다. 예측 및 분류의 정확성을 높이기 위해  임상 소견이 포함된 방사선 리포트, 즉 X-선 이미지와 관련된 텍스트 정보를 함께 사용하였습니다. 
**MIMIC-CXR** 데이터셋을 사용하고, **CLIP** (Contrastive Language-Image Pre-training)에서 사전학습한 인코더를 사용하여 질병 별로 적은 양의 데이터로도 효과적인 분류가 가능하도록 학습(few-shot learning) 하도록 구현하였습니다. 
'''
st.markdown(motivation)

st.subheader('Method')
method = '''
FlexR (Few-shot classification with Language Embeddings for chest X-ray reporting)은 대량의 구조화되지 않은 방사선 데이터를 사용하여 몇가지 고품질의 주석만으로도 구조화되고 세밀한 임상적 소견을 예측하는 방법을 제안합니다. 
흉부  X-선과 연관된 방사선 리포트를 이용하여 언어-이미지 모델을 학습하고 텍스트 프롬프트를 구조화한 후, 분류기를 의료 이미지로부터 구조화된 임상적 소견을 예측하도록 최적화하는 것이라고 할 수 있습니다. 
'''
st.markdown(method)

st.subheader('Experiments')
st.image(Image.open('method.jpg'))
experiments = '''
1️⃣ CLIP (Contrastive Language-Image Pretraining) : 
의료 이미지와 자연 이미지 간의 도메인 차이로 CLIP 모델을 재학습합니다. 이미지 인코더로 ResNet50, 텍스트 인코더로 Transformer를 사용합니다. 

2️⃣ Language embeddings of clinical findings : 
구조화된 리포트 템플릿의 모든 가능한 옵션을 개별 문장으로 추출하고 CLIP의 인코더로 인코딩하여 각 임상 소견에 대한 텍스트 임베딩을 생성합니다. 

3️⃣ Fine-tuning the classifier : 
앞선 단계에서 준비한 모델과 데이터를 사용하여 분류기를 파인 튜닝 합니다. 
'''
st.markdown(experiments)

st.image(Image.open('text.png'), width=700)

col1, col2 = st.columns(2)

with col1:
    pathology = st.selectbox(
      'Pathology',
      ('pneumothorax', 'pneumonia', 'fluid overload/heart failure', 
      'consolidation', 'pleural effusion', 'atelectasis', 
      'pulmonary edema/hazy opacity', 'lung opacity', 'enlarged cardiac silhouette'),
      # index=None,
      # placeholder="Select pathology"
    )

with col2: 
    location = st.selectbox(
      'Location',
      ('left apical zone', 'left lower lung zone', 'left lung', 
      'left mid lung zone', 'left upper lung zone', 
      'right apical zone', 'right lower lung zone', 'right lung',
      'right mid lung zone', 'right upper lung zone', 
      'left hilar structures', 'right hilar structures', 
      'left costophrenic angle', 'right costophrenic angle',
      'mediastinum', 'upper mediastinum',
      'cardiac silhouette', 'trachea')
    )

# st.write('Template : ', pathology, 'in the', location)
annotated_text(
    (pathology, "pathology", "#afa"),
    " in the ",
    (location, "location", "#8ef")
)

st.subheader('Contribution')
contribution = '''
본 프로젝트를 통해 얻은 주요 결과 중 하나는 방사선 리포트를 활용함으로써 소량의 데이터로도 질병 분류의 성능을 향상시킬 수 있다는 것입니다. 
흉부 X-선 이미지와 표준화된 리포트의 텍스트 정보를 연결함으로써  적은 양의 데이터로도 정확하고 신뢰할 수 있는 질병 분류를 달성할 수 있습니다. 
'''
st.markdown(contribution)

show_pages([
        Page("home.py", "Home", "🏠"),
        Page("test.py", "Test", icon="🫁"),
])