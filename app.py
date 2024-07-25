import numpy as np
import joblib
import streamlit as st
st.set_page_config(layout='wide')
from PIL import Image
import cv2
from streamlit_drawable_canvas import st_canvas

knn=joblib.load('knn_model.pkl')

st.title("Handwritten Digit Recognition")
st.markdown(
    """
<style>
.ok-font{
    font-size:20px
}
""",unsafe_allow_html=True
)
st.markdown('<p class="ok-font">This model is trained on the MNIST Dataset using K-Nearest Neighbours to accurately recognize handwritten digits</p',unsafe_allow_html=True)


st.text("")
st.text("")
col1,col2,col3,col4=st.columns([.2,.3,.3,.2],gap='medium')


def preprocess_image(img):
    img=img.resize((28,28))

    img=img.convert('L')

    img=Image.fromarray(255-np.array(img))

    img=np.array(img)

    img=img.flatten().reshape(1,-1)
    return img


with col2:
    canvas_result=st_canvas(
    fill_color='white',
    stroke_width=10,
    stroke_color='black',
    background_color='white',
    height=350,
    width=350,
    drawing_mode='freedraw',
    key='canvas'
    )   
    predicted=st.button('Predict')


st.markdown(
    """
<style>
.big-font{
    font-size:80px
}
""",unsafe_allow_html=True
)

with col3:
    st.subheader(':1234: Predicted digit: ')
    if predicted:
        if canvas_result.image_data is not None:
            img=Image.fromarray(canvas_result.image_data.astype('uint8'),'RGBA')
            img=img.convert('L')
            img=preprocess_image(img)
            prediction=knn.predict(img)
            
            st.markdown(f'<p class="big-font">{prediction[0]}</p>',unsafe_allow_html=True)
        else:
            st.write('Please draw a digit first')









footer="""<style>
a:link , a:visited{
color: white;
background-color: transparent;
text-decoration: underline;
}

a:hover,  a:active {
color: lavender;
background-color: transparent;
text-decoration: underline;
}

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: #702963;
color: white;
text-align: center;
}
</style>
<div class="footer">
<p> </p>
<p>Developed with ❤ by <a style='display: block; text-align: center;' href="https://github.com/arushi-midha" target="_blank">Arushi Midha</a></p>
</div>
"""
st.markdown(footer,unsafe_allow_html=True)