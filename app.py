import streamlit as st
from streamlit_folium import st_folium
import folium
import pandas as pd
import numpy as np
from geopy.distance import geodesic
import statsmodels.formula.api as smf
import unicodedata
import random
import joblib
import os

# ====== FUNÇÕES AUXILIARES ======

def remover_acentos(colunas):
    return [unicodedata.normalize('NFKD', col).encode('ascii', errors='ignore').decode('utf-8').strip() for col in colunas]

def carregar_dados(file_path):
    df = pd.read_excel(file_path, sheet_name="Amostra")
    df.columns = remover_acentos(df.columns)

    categorias = {3: 'Março', 4: 'Abril', 5: 'Maio', 6: 'Junho'}
    df['Meses'] = df['Mes'].map(categorias)

    categorias = {1: 'Dallas', 2: 'Fortworth', 3: 'Outros'}
    df['Regiao'] = df['Localizacao'].map(categorias)

    # ⚠️ USAR LATITUDE E LONGITUDE reais da planilha
    df['DistanciaCentro'] = df.apply(
        lambda row: geodesic((row['Latitude'], row['Longitude']), (32.7767, -96.7970)).km,
        axis=1
    )

    return df

def treinar_modelo(df):
    modelo = smf.ols('Preco ~ Area + Quartos + Idade + DistanciaCentro', data=df).fit()
    return modelo

# ====== INÍCIO DO APP ======

st.set_page_config(page_title="Estimador de Imóveis", layout="centered")
st.title("🏡 Estimador de Preço de Imóveis")
st.markdown("Selecione a localização no mapa e informe os dados do imóvel para obter o preço estimado.")

# ====== ETAPA 1: Carregar dados e treinar ou carregar modelo ======

file_path = 'EASTON.xlsx'
modelo_path = 'modelo_imovel.pkl'

df = carregar_dados(file_path)

# 👉 Comente a linha abaixo se NÃO quiser treinar o modelo novamente
forcar_treinamento = True

if forcar_treinamento or not os.path.exists(modelo_path):
    modelo = treinar_modelo(df)
    joblib.dump(modelo, modelo_path)
    st.info("🔄 Modelo treinado e salvo novamente.")
else:
    modelo = joblib.load(modelo_path)
    st.success("✅ Modelo carregado de modelo_imovel.pkl")

centro_dallas = (32.7767, -96.7970)

# ====== ETAPA 2: Mapa com clique e marcador persistente ======

if 'pin_coords' not in st.session_state:
    st.session_state.pin_coords = None

m = folium.Map(location=centro_dallas, zoom_start=11)
folium.Marker(location=centro_dallas, tooltip="Centro de Dallas", icon=folium.Icon(color="blue")).add_to(m)

if st.session_state.pin_coords:
    folium.Marker(
        location=(st.session_state.pin_coords['lat'], st.session_state.pin_coords['lng']),
        tooltip="Novo imóvel",
        icon=folium.Icon(color="green", icon="home")
    ).add_to(m)

st.markdown("### 🗺️ Clique no mapa para escolher a localização do imóvel:")
map_data = st_folium(m, width=700, height=500)

if map_data and map_data['last_clicked']:
    st.session_state.pin_coords = map_data['last_clicked']

# ====== ETAPA 3: Formulário de dados e predição ======

if st.session_state.pin_coords:
    lat = st.session_state.pin_coords['lat']
    lon = st.session_state.pin_coords['lng']

    st.success(f"📍 Local selecionado: ({lat:.5f}, {lon:.5f})")

    with st.form("dados_imovel"):
        area = st.number_input("Área do imóvel (m²):", min_value=10, step=1, format="%d")
        quartos = st.number_input("Quantidade de quartos:", min_value=1, step=1)
        idade = st.number_input("Idade do imóvel (anos):", min_value=0, step=1)
        mes = st.selectbox("Mês da compra:", ['Março', 'Abril', 'Maio', 'Junho'])
        enviar = st.form_submit_button("📈 Estimar preço")

    if enviar:
        distancia = geodesic((lat, lon), centro_dallas).km
        entrada = pd.DataFrame({
            'Area': [area],
            'Quartos': [quartos],
            'Idade': [idade],
            'DistanciaCentro': [distancia]
        })

        preco = modelo.predict(entrada)[0]
        st.subheader("💰 Preço estimado:")
        st.success(f"${preco:,.2f}")
