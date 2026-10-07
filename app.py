import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Cargar el modelo entrenado
modelo_rf = joblib.load('modelo_rf.pkl')

# Configuración de la página
st.set_page_config(page_title="Evaluación Crediticia", page_icon="🛡️", layout="centered")

st.title("🛡️ Plataforma de Evaluación Crediticia")
st.markdown("### Vista del Oficial de Crédito")
st.write("Ingrese los datos del solicitante para evaluar el riesgo de default.")

# Formularios de entrada
col1, col2 = st.columns(2)
with col1:
    monto = st.number_input("Monto del Crédito Solicitado ($)", min_value=0.0, value=100000.0)
    ingreso = st.number_input("Ingreso Total Declarado ($)", min_value=0.0, value=50000.0)
    edad = st.number_input("Edad del Solicitante (Años)", min_value=18.0, value=30.0)
with col2:
    anios_emp = st.number_input("Antigüedad Laboral (Años)", min_value=0.0, value=5.0)
    score2 = st.slider("Score Externo 2 (Buró)", 0.0, 1.0, 0.5)
    score3 = st.slider("Score Externo 3 (Buró)", 0.0, 1.0, 0.5)

# Botón de predicción
if st.button("Evaluar Riesgo Crediticio", type="primary"):
    # Armamos el dataframe con los datos ingresados
    datos_cliente = pd.DataFrame([[ingreso, monto, edad, anios_emp, score2, score3]], 
                                 columns=['ingreso_total', 'monto_credito', 'edad_anios', 'anios_empleado', 'score_externo_2', 'score_externo_3'])
    
    # Calculamos la probabilidad
    probabilidad = modelo_rf.predict_proba(datos_cliente)[0][1] * 100
    
    st.divider()
    st.markdown("### 📊 DICTAMEN DEL SISTEMA")
    
    # Lógica del semáforo
    if probabilidad < 30:
        st.success(f"**APROBADO** | Riesgo Default: {probabilidad:.1f}%")
        st.write("Cliente con bajo riesgo de impago. Proceder con el alta.")
    elif probabilidad < 60:
        st.warning(f"**REVISIÓN MANUAL** | Riesgo Default: {probabilidad:.1f}%")
        st.write("Riesgo moderado. Derivar a supervisor para análisis adicional.")
    else:
        st.error(f"**RECHAZADO** | Riesgo Default: {probabilidad:.1f}%")
        st.write("Alto riesgo crediticio detectado. Denegar solicitud.")