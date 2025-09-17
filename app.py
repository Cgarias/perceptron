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
        st.session_state.entrenado = False
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
        st.session_state.entrenado = True

        # Simulación
        st.subheader("🔎 Simulación con los patrones")
        for x in X:
           s = np.dot(x, W) - U
           y = escalon(s)
           st.write(f"{x} -> {y}")
           
# Crear un contenedor para almacenar pruebas
# ---------- Pruebas manuales (solo después de entrenar) ----------
if st.session_state.get("entrenado", False):

    st.subheader("📝 Probar con valores manuales")

    if "pruebas" not in st.session_state:
        st.session_state.pruebas = []

    with st.form("form_prueba"):
        a = st.number_input("Ingrese el primer valor (ej: 5)", value=0.0, key="a")
        b = st.number_input("Ingrese el segundo valor (ej: 69)", value=0.0, key="b")
        c = st.number_input("Ingrese el tercer valor (ej: 8)", value=0.0, key="c")

        submitted = st.form_submit_button("Agregar prueba")

        if submitted:
            m = np.array([a, b, c])
            s = np.dot(m, W) - U
            y = escalon(s)
            st.session_state.pruebas.append({"x1": a, "x2": b, "x3": c, "salida": y})

    if st.session_state.pruebas:
        st.write("### 📊 Resultados de las pruebas")
        df_pruebas = pd.DataFrame(st.session_state.pruebas)
        st.dataframe(df_pruebas, use_container_width=True)

        if st.button("🗑️ Limpiar pruebas"):
            st.session_state.pruebas = []

else:
    st.info("⚠️ Primero debes entrenar el perceptrón para poder hacer pruebas manuales.")




