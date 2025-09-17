import time
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from dataset_utils import detect_io, preprocess_for_perceptron

# ---------------------------
# Interfaz Streamlit
# ---------------------------

st.title("🔵 Perceptrón Simple")



# Cargar dataset
st.subheader("📂 Cargar Dataset")
uploaded_file = st.file_uploader("Sube tu dataset (CSV, Excel o Json)", type=["csv", "xlsx", "json"])

if uploaded_file is not None:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(".json"):
        df = pd.read_json(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.write("Vista previa del dataset:")
    st.dataframe(df.head())

    #Mostrar valores entrada,salida y patrones.
    inputs, outputs, n_patterns = detect_io(df)

    st.write("Entradas:", len(inputs))
    st.write("Salidas:", len(outputs))
    st.write("Patrones:", n_patterns)


    X, Y = preprocess_for_perceptron(df, inputs, outputs)

    st.sidebar.header("⚙️ Configuración")
    rng = np.random.RandomState(42)
    W = rng.uniform(-1, 1, size=(X.shape[1],))
    U = rng.uniform(-1, 1)
    eta = st.sidebar.slider("Tasa de Aprendizaje (η)", 0.01, 1.0, 0.1)
    max_iter = st.sidebar.slider("Número máximo de iteraciones", 1, 10, 100)
    error_max = st.sidebar.slider("Error máximo permitido (ϵ)", 0.0, 1.0, 0.01)


    def escalon(s): 
        return 1 if s >= 0 else 0

    # ---------- Contenedor para gráfico ----------
    plot_area = st.empty()

    # ---------- Entrenamiento ----------
    errores = []

    start_training = st.button("🚀 Iniciar Entrenamiento")
    if start_training:
        for epoch in range(max_iter):
            error_total = 0
            for i in range(n_patterns):
                x = X[i]
                yd = Y[i]
                s = np.dot(x, W) - U
                y = escalon(s)
                e = yd - y
                error_total += e**2
                W = W + eta * e * x
                U = U - eta * e

            # Calcular error RMS
            erms = np.sqrt(error_total / n_patterns)
            errores.append(erms)

            # Redibujar el gráfico en el mismo lugar
            fig, ax = plt.subplots()
            ax.plot(errores, marker="o", color="blue")
            ax.set_xlabel("Iteración")
            ax.set_ylabel("Error RMS")
            ax.set_title("Evolución del error")
            plot_area.pyplot(fig)

            # Mostrar iteración y error
            st.write(f"Iteración {epoch+1}, Error RMS = {erms:.4f}")

            time.sleep(0.2)  # pausa para efecto "tiempo real"

            if erms <= error_max:
                st.success("✅ Condición de parada alcanzada")
                break

        st.write("Pesos finales:", W)
        st.write("Umbral final:", U)

        # Simulación
        st.subheader("🔎 Simulación con los patrones")
        for x in X:
           s = np.dot(x, W) - U
           y = escalon(s)
           st.write(f"{x} -> {y}")

        M = np.array([[5,69,8],[3,74,2],[6,75,3],[1,76.5,0]])
        for m in M:
            s = np.dot(m, W) - U
            y = escalon(s)
            st.write(f"{m} -> {y}")



