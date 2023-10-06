import streamlit as st
from PIL import Image
import pandas as pd

st.set_page_config(layout="wide", page_title="M3 demo", page_icon=":green_apple:")
st.header('What diseases can be predicted from Chest X-ray images?')

MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
image_num = 1

df = pd.read_csv('result.csv')

col1, col2 = st.columns(2)
my_upload = st.sidebar.file_uploader("Upload an image :gear:", type=["png", "jpg", "jpeg"])
# if my_upload is not None:
#     image_num = my_upload.name[:-4]

def fix_image(upload):
    image = Image.open(upload)
    col1.write("X-ray Image :camera:")
    col1.image(image)

    col2.write("Prediction :mag:")
    col2.info(df['1-shot'][df['i']==int(image_num)].values, icon="1️⃣")
    col2.info(df['10-shot'][df['i']==int(image_num)].values, icon="🔟")
    # col2.markdown('---')
    col2.write("Label")
    col2.warning(df['label'][df['i']==int(image_num)].values, icon="👀")
    # col2.success('This is correct.', icon="⭕️")
    # col2.error('This is wrong.', icon="❌") 
    # st.balloons()

if my_upload is not None:
    image_num = my_upload.name[:-4]
    if my_upload.size > MAX_FILE_SIZE:
        st.error("The uploaded file is too large. Please upload an image smaller than 5MB.")
    else:
        fix_image(upload=my_upload)
else:
    fix_image("test/1.jpg")
