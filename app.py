import cv2
import numpy as np
import streamlit as st
from sklearn.mixture import GaussianMixture
import imutils
import matplotlib.pyplot as plt

st.sidebar.title("Segmentação de Imagem")
uploaded_file = st.sidebar.file_uploader("Escolha uma imagem", type="jpg")
image = st.sidebar.empty()


if uploaded_file is not None:
	classes = st.sidebar.slider("Numero de classes",1,20,2)
	blur = st.sidebar.slider("Filtro Gaussiano",1,15,1)
	tamanho = st.sidebar.slider("Redimensionamento",1,15,5)
	file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
	opencv_image = cv2.imdecode(file_bytes, 1)
	mini = imutils.resize(opencv_image, width=200)
	image.image(mini, channels = 'BGR')
	opencv_image = imutils.resize(opencv_image, width=100*tamanho)

	#st.image(opencv_image, channels='BGR')

	# Aplicando filtro gaussiano para redução de ruídos
	opencv_image_blurred = cv2.blur(opencv_image,(blur,blur))

	# Processamento da imagem para matrix vertical
	img_temp = cv2.cvtColor(opencv_image_blurred,cv2.COLOR_BGR2GRAY).reshape(-1,1)


	# Treinamento de Modelo de Mistura Gaussiana determinando a quantidade de grupos
	gm = GaussianMixture(n_components = classes, random_state=1).fit(img_temp)

	resultado = gm.predict(img_temp)

	maskgm = resultado.reshape(opencv_image.shape[:2]) * 127.5
	maskgm = cv2.normalize(maskgm,maskgm, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
	fig, ax = plt.subplots(1,figsize=(10,10))
	ax.imshow(maskgm, cmap="Spectral")
	st.pyplot(fig)
