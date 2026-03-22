# 🛰️ Orbital Insight Analytics

> **Plataforma de análise de dados de satélite** para detecção de desmatamento,
> previsão de produtividade agrícola e monitoramento de crescimento urbano —
> usando Ciência de Dados e Machine Learning.

---

## 📋 Funcionalidades

| Módulo | Descrição | Modelo ML |
|--------|-----------|-----------|
| 🌳 **Desmatamento** | Detecta perda de vegetação via NDVI temporal | Random Forest (classificação) |
| 🌾 **Agricultura** | Prevê produtividade da safra (ton/ha) | Gradient Boosting (regressão) |
| 🏙️ **Crescimento Urbano** | Mapeia expansão de áreas urbanas | Random Forest (classificação) |

---

## 🏗️ Estrutura do Projeto

```
Orbital-Insight-Analitycs/
│
├── backend/                  # API REST (FastAPI)
│   ├── main.py               # Aplicação principal + CORS
│   └── api/
│       ├── deforestation.py  # Endpoints de desmatamento
│       ├── agriculture.py    # Endpoints de agricultura
│       └── urban.py          # Endpoints de crescimento urbano
│
├── frontend/
│   └── app.py                # Dashboard Streamlit
│
├── models/                   # Modelos de Machine Learning
│   ├── deforestation_model.py  # Classificador (Random Forest)
│   ├── agriculture_model.py    # Regressor (Gradient Boosting)
│   └── urban_model.py          # Classificador urbano (Random Forest)
│
├── data/
│   ├── simulate_data.py          # Scripts de simulação de dados
│   ├── deforestation_dataset.csv # Dataset simulado de desmatamento
│   ├── agriculture_dataset.csv   # Dataset simulado de agricultura
│   ├── urban_growth_dataset.csv  # Dataset simulado de crescimento urbano
│   └── satellite_images/         # Imagens NDVI simuladas (.npy)
│
├── notebooks/
│   └── exploratory_analysis.ipynb  # Análise exploratória completa
│
├── utils/
│   ├── ndvi.py             # Cálculo e simulação de NDVI
│   ├── visualization.py    # Gráficos matplotlib/plotly
│   └── data_processing.py  # Normalização, extração de features
│
└── requirements.txt         # Dependências Python
```

---

## ⚡ Como Executar

### 1. Clone e instale as dependências

```bash
git clone https://github.com/germanoafb/Orbital-Insight-Analitycs.git
cd Orbital-Insight-Analitycs

pip install -r requirements.txt
```

### 2. Gere os dados simulados

```bash
python data/simulate_data.py
```

### 3. Inicie o Backend (FastAPI)

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Acesse a documentação interativa: http://localhost:8000/docs

### 4. Inicie o Frontend (Streamlit)

Em outro terminal:

```bash
streamlit run frontend/app.py
```

Acesse o dashboard: http://localhost:8501

### 5. Execute o Notebook de Análise

```bash
jupyter notebook notebooks/exploratory_analysis.ipynb
```

---

## 📊 Endpoints da API

### Desmatamento (`/api/deforestation`)

| Método | Rota | Descrição |
|--------|------|-----------|
| GET | `/api/deforestation/train` | Treina o classificador |
| POST | `/api/deforestation/predict` | Prediz desmatamento por features |
| POST | `/api/deforestation/simulate` | Simula cenário before/after |
| GET | `/api/deforestation/demo` | Executa análise demo |

### Agricultura (`/api/agriculture`)

| Método | Rota | Descrição |
|--------|------|-----------|
| GET | `/api/agriculture/train` | Treina o regressor |
| POST | `/api/agriculture/predict` | Prediz produtividade |
| POST | `/api/agriculture/trend` | Tendência multi-anual |
| GET | `/api/agriculture/demo` | Executa predição demo |

### Crescimento Urbano (`/api/urban`)

| Método | Rota | Descrição |
|--------|------|-----------|
| GET | `/api/urban/train` | Treina o classificador urbano |
| POST | `/api/urban/simulate` | Simula expansão urbana |
| POST | `/api/urban/stats` | Calcula estatísticas de crescimento |
| GET | `/api/urban/demo` | Executa análise demo |

---

## 🧠 Modelos de Machine Learning

### Detecção de Desmatamento (Classificação)
- **Algoritmo:** Random Forest (200 estimadores)
- **Features:** NDVI médio, desvio padrão, percentis, frações de cobertura, variação temporal
- **Métrica:** Accuracy ≥ 0.99

### Produtividade Agrícola (Regressão)
- **Algoritmo:** Gradient Boosting (200 estimadores)
- **Features:** NDVI, umidade do solo, temperatura, precipitação, tipo de cultura, dias para colheita
- **Métricas:** RMSE ~1.0 t/ha, R² ~0.97

### Crescimento Urbano (Classificação)
- **Algoritmo:** Random Forest (200 estimadores)
- **Features:** NDVI, bandas Red/NIR, índice de área construída, brilho, umidade
- **Métrica:** Accuracy ≥ 0.99

---

## 🌍 Tecnologias Utilizadas

- **Backend:** Python 3.10+, FastAPI, Uvicorn
- **Frontend:** Streamlit
- **Machine Learning:** Scikit-learn (RandomForest, GradientBoosting)
- **Processamento de dados:** NumPy, Pandas
- **Visualização:** Matplotlib, Plotly
- **Serialização de modelos:** Joblib

---

## 📸 Dashboard

O dashboard Streamlit inclui 5 páginas:

1. **🏠 Home** — Visão geral da plataforma e exemplo de mapa NDVI
2. **🌳 Desmatamento** — Simulação e análise de perda de vegetação
3. **🌾 Agricultura** — Previsão de produtividade e tendências multi-anuais
4. **🏙️ Crescimento Urbano** — Classificação e comparação temporal de áreas urbanas
5. **📊 Métricas dos Modelos** — Accuracy, RMSE, R², importância de features

---

## 🔬 NDVI (Índice de Vegetação por Diferença Normalizada)

```
NDVI = (NIR - Red) / (NIR + Red)
```

| Faixa NDVI | Interpretação |
|------------|---------------|
| < 0 | Água / sem dados |
| 0.0 – 0.2 | Solo exposto / área urbana |
| 0.2 – 0.4 | Vegetação esparsa |
| 0.4 – 0.6 | Vegetação moderada |
| ≥ 0.6 | Vegetação densa / floresta |

---

## 📄 Licença

MIT License — veja o arquivo [LICENSE](LICENSE) para detalhes.
