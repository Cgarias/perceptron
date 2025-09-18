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
    # Lectura del archivo según extensión
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(".json"):
        df = pd.read_json(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.write("Vista previa del dataset:")
    st.dataframe(df.head())

    # Detectar entradas, salidas y patrones
    inputs, outputs, n_patterns = detect_io(df)
    st.write("Entradas:", len(inputs))
    st.write("Salidas:", len(outputs))
    st.write("Patrones:", n_patterns)

    # Preprocesar dataset
    X, Y = preprocess_for_perceptron(df, inputs, outputs)

    # Sidebar configuración
    st.sidebar.header("⚙️ Configuración")
    rng = np.random.RandomState(42)

    # Inicializar pesos/umbral
    if "W" not in st.session_state or st.session_state.get("W") is None or st.session_state.W.shape[0] != X.shape[1]:
        st.session_state.W = rng.uniform(-1, 1, size=(X.shape[1],))
    if "U" not in st.session_state or st.session_state.get("U") is None:
        st.session_state.U = rng.uniform(-1, 1)

    W = st.session_state.W
    U = st.session_state.U

    # Sliders corregidos
    eta = st.sidebar.slider("Tasa de Aprendizaje (η)", 0.01, 1.0, 0.1)
    max_iter = st.sidebar.slider("Número máximo de iteraciones", 1, 1000, 100)  # corregido
    error_max = st.sidebar.slider("Error máximo permitido (ϵ)", 0.0, 1.0, 0.01)

    # Función de activación
    def escalon(s):
        return 1 if s >= 0 else 0

    # Contenedor para gráfico
    plot_area = st.empty()

    # Entrenamiento
    errores = []
    start_training = st.button("🚀 Iniciar Entrenamiento")

    if start_training:
        st.session_state.entrenado = False
        for epoch in range(max_iter):
            error_total = 0
            for i in range(n_patterns):
                x = X[i]
                yd = int(np.array(Y).ravel()[i])
                s = np.dot(x, W) - U
                y = escalon(s)
                e = yd - y
                error_total += e**2
                W = W + eta * e * x
                U = U - eta * e

            # Calcular error RMS
            erms = np.sqrt(error_total / n_patterns)
            errores.append(erms)

            # Redibujar gráfico
            fig, ax = plt.subplots()
            ax.plot(errores, marker="o", color="blue")
            ax.set_xlabel("Iteración")
            ax.set_ylabel("Error RMS")
            ax.set_title("Evolución del error")
            plot_area.pyplot(fig)

            # Mostrar progreso
            st.write(f"Iteración {epoch+1}, Error RMS = {erms:.4f}")
            time.sleep(0.2)

            if erms <= error_max:
                st.success("✅ Condición de parada alcanzada")
                break

        st.write("Pesos finales:", W)
        st.write("Umbral final:", U)

        st.session_state.W = W
        st.session_state.U = U
        st.session_state.entrenado = True

        # Simulación con patrones del dataset
        st.subheader("🔎 Simulación con los patrones")
        for x in X:
            s = np.dot(x, W) - U
            y = escalon(s)
            st.write(f"{x} -> {y}")

# ---------------------------
# Pruebas manuales
# ---------------------------
if st.session_state.get("entrenado", False):

    st.subheader("📝 Probar con valores manuales")

    if "pruebas" not in st.session_state:
        st.session_state.pruebas = []

    with st.form("form_prueba"):
        entradas_usuario = []
        for i in range(st.session_state.W.shape[0]):
            valor = st.number_input(f"Ingrese el valor de x{i+1}", value=0.0, key=f"entrada_{i}")
            entradas_usuario.append(valor)

        submitted = st.form_submit_button("Agregar prueba")

        if submitted:
            m = np.array(entradas_usuario, dtype=float)
            W_trained = st.session_state.W
            U_trained = st.session_state.U

            if m.shape[0] != W_trained.shape[0]:
                st.error(
                    f"❌ Número de valores ingresados incorrecto.\n"
                    f"El modelo espera {W_trained.shape[0]} entradas."
                )
            else:
                s = np.dot(m, W_trained) - U_trained
                y = escalon(s)
                resultado = {f"x{i+1}": entradas_usuario[i] for i in range(len(entradas_usuario))}
                resultado["salida"] = int(y)
                st.session_state.pruebas.append(resultado)

    if st.session_state.pruebas:
        st.write("### 📊 Resultados de las pruebas")
        df_pruebas = pd.DataFrame(st.session_state.pruebas)
        st.dataframe(df_pruebas, use_container_width=True)

        if st.button("🗑️ Limpiar pruebas"):
            st.session_state.pruebas = []

else:
    st.info("⚠️ Primero debes entrenar el perceptrón para poder hacer pruebas manuales.")
