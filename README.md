# AchillesMoE-XAU 🤖💰

Bot profesional de trading para XAUUSD (oro) con arquitectura MoE (Mixture of Experts):
- **Burócrata**: experto para mercados tranquilos  
- **Seldon**: experto para crisis (2008, COVID, Ucrania, 2025…)  
- Entrenamiento en Google Colab o local (con GPU support)  
- Deploy en Vertex AI (custom container)  
- Ejecución real-time desde MetaTrader 5  

## Estructura del proyecto
```
AchillesMoE-XAU/
├── colab/                  ← Notebook completo listo para ejecutar
│   └── AchillesMoE_V4.ipynb
├── docker/
│   ├── Dockerfile
│   └── predictor.py        ← Custom prediction routine para Vertex AI
├── mt5/
│   └── mt5_bot.py          ← Script que corre en tu PC y llama al endpoint
├── data/                   ← (vacía, tú subes tus CSVs de crisis aquí)
├── models/                 ← (se genera automáticamente al entrenar)
├── .github/
│   └── workflows/
│       └── test.yml        ← Auto-tests con GitHub Actions (verifica sintaxis al push)
├── .gitignore              ← Ignora archivos sensibles/grandes
├── requirements.txt        ← Dependencias para instalar con pip
├── .env.example            ← Template para secrets como ENDPOINT (no subas .env real)
└── README.md
```

## Instalación rápida

### 1. Clonar el repo
```bash
git clone https://github.com/trece37/AchillesMoE-XAU.git
cd AchillesMoE-XAU
```

### 2. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 3. (Opcional) Entrenar en Google Colab
1. Abre `colab/AchillesMoE_V4.ipynb` en Colab  
2. Ejecuta todas las celdas  
3. Descarga los modelos generados (`burocrata.pth`, `seldon.pth`) y guárdalos en `models/`

### 4. Deploy en Vertex AI
```bash
cd docker/
# Construir imagen
docker build -t gcr.io/[TU_PROJECT_ID]/achillesmoe:latest .

# Subir a GCP
docker push gcr.io/[TU_PROJECT_ID]/achillesmoe:latest

# Crear endpoint en Vertex AI con esa imagen
```

### 5. Ejecutar desde MT5
1. Copia tu `.env` con la URL del endpoint de Vertex AI  
2. Corre `mt5/mt5_bot.py` en tu PC con MetaTrader 5 abierto  
3. El bot envía datos de mercado cada tick al endpoint y recibe predicciones

---

## Características principales

- ✅ **Arquitectura MoE**: dos expertos LSTM entrenados en distintos regímenes de mercado  
- ✅ **Footprint data**: incluye volumen/tick y footprint si está disponible  
- ✅ **Docker + Vertex AI**: deploy productivo con auto-scaling  
- ✅ **GitHub Actions**: CI/CD automático para tests de sintaxis  
- ✅ **Python 3.10+** con TensorFlow 2.x  
- ✅ **Licencia MIT**: úsalo y modifícalo libremente  

---

## Roadmap & To-Do

- [ ] Añadir scripts de backtesting histórico  
- [ ] Implementar RL (Reinforcement Learning) como experto adicional  
- [ ] Métricas de performance en tiempo real (Sharpe, DD, etc.)  
- [ ] Interfaz web con Streamlit para monitoreo  

---

## Contribuciones

¡Pull requests bienvenidos! Si encuentras bugs o quieres añadir features, abre un issue o envía un PR.

---

## Licencia

[MIT License](LICENSE) — usa, copia, modifica y distribuye como quieras.

---

## Disclaimer

**Este bot es sólo para fines educativos y de investigación.**  
Trading automatizado conlleva riesgos. No hay garantías de rentabilidad.  
Usa bajo tu propia responsabilidad.
