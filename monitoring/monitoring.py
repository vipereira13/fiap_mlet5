import pandas as pd
import os
from datetime import datetime

LOG_FILE = "logs/drift_monitoring.csv"

def registrar_inferencia(probabilidade, contratado_predito, contratado_real=None):
    os.makedirs("logs", exist_ok=True)

    nova_linha = {
        "data": datetime.now().strftime("%Y-%m-%d"),
        "hora": datetime.now().strftime("%H:%M:%S"),
        "probabilidade": round(float(probabilidade), 4),
        "contratado_predito": int(contratado_predito),
        "contratado_real": contratado_real if contratado_real is not None else -1  # -1 = não informado
    }

    if os.path.exists(LOG_FILE):
        df = pd.read_csv(LOG_FILE)
        df = pd.concat([df, pd.DataFrame([nova_linha])], ignore_index=True)
    else:
        df = pd.DataFrame([nova_linha])

    df.to_csv(LOG_FILE, index=False)