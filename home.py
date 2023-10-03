import streamlit as st
from PIL import Image

st.title('Medical Image Project - by M3 :apple:')
# emoji: https://streamlit-emoji-shortcodes-streamlit-app-gwckff.streamlit.app/
# st.title('스마일 :sunglasses:')

st.header('Few-shot Classification of Chest X-ray Disease')
st.subheader('MIMIC 데이터셋과 CLIP을 이용한 구조화된 임상적 소견 예측')
st.image(Image.open('/Users/kim-yelin/Documents/DeepDaiv/demo/주제.png'))

# st.bar_chart(
#   data={'time': [0, 1, 2, 3, 4, 5, 6], 'stock_value': [100, 200, 150, 300, 450, 500, 600]},
#   x='time',
#   y='stock_value'
# )

# st.file_uploader('Browse files')

st.image(Image.open('/Users/kim-yelin/Documents/DeepDaiv/demo/텍스트.png'))

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

st.write('Template : ', pathology, 'in the', location)
# st.caption('캡션을 한 번 넣어 봤습니다')
# st.text('일반적인 텍스트를 입력해 보았습니다.')

# 컬러코드: blue, green, orange, red, violet
# st.markdown("텍스트의 색상을 :green[초록색]으로, 그리고 **:blue[파란색]** 볼트체로 설정할 수 있습니다.")

# LaTex 수식 지원
# st.latex(r'\mathcal{L}_{\mathrm{LSES}} = \mathrm{log} \left( 1+\sum^C_{i=1}e^{-y_i\gamma s_i} \right)')

from st_pages import Page, Section, show_pages, add_page_title

# add_page_title() 
show_pages([
        Page("home.py", "Home", "🏠"),
        Page("test.py", "Test", icon="🫁"),
])