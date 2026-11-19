import os, json, time, joblib, requests
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="Predicción de rendimiento académico")

model = joblib.load("model_v1.joblib")
with open("metrics.json") as f:
    TRAIN = json.load(f)

SUPA_URL = os.getenv("SUPABASE_URL")
SUPA_KEY = os.getenv("SUPABASE_KEY")
HEADERS = {"apikey": SUPA_KEY or "", "Authorization": f"Bearer {SUPA_KEY}",
           "Content-Type": "application/json"}
total_predictions = 0


class Estudiante(BaseModel):
    nombre: str
    asistencia: float = Field(ge=0, le=100)
    horas_estudio: float = Field(ge=0, le=40)
    parcial1: float = Field(ge=0, le=100)
    parcial2: float = Field(ge=0, le=100)
    tareas: float = Field(ge=0, le=100)


@app.get("/health")
def health():
    return {"status": "ok", "model_version": TRAIN["model_version"]}


@app.get("/metrics")
def metrics():
    return {**TRAIN, "total_predictions": total_predictions}


@app.post("/estudiantes")
def registrar(e: Estudiante):
    global total_predictions
    t0 = time.time()
    x = np.array([[e.asistencia, e.horas_estudio, e.parcial1,
                   e.parcial2, e.tareas]])
    p = float(model.predict_proba(x)[0][1])
    latency = round((time.time() - t0) * 1000, 2)
    total_predictions += 1

    prediccion = "Aprobará" if p >= 0.5 else "En riesgo"
    riesgo = "Bajo" if p >= 0.75 else ("Medio" if p >= 0.5 else "Alto")

    row = {**e.model_dump(), "prob_aprobar": round(p, 4), "prediccion": prediccion,
           "model_version": TRAIN["model_version"], "latency_ms": latency}
    if SUPA_URL and SUPA_KEY:
        try:
            requests.post(f"{SUPA_URL}/rest/v1/estudiantes",
                          headers=HEADERS, json=row, timeout=5)
        except Exception:
            pass

    return {"nombre": e.nombre, "prediccion": prediccion,
            "prob_aprobar": round(p, 4), "riesgo": riesgo,
            "model_version": TRAIN["model_version"], "latency_ms": latency}


@app.get("/estudiantes")
def listar():
    if not (SUPA_URL and SUPA_KEY):
        return []
    try:
        r = requests.get(
            f"{SUPA_URL}/rest/v1/estudiantes?select=id,nombre,prediccion,prob_aprobar"
            "&order=id.desc&limit=5", headers=HEADERS, timeout=5)
        return r.json()
    except Exception:
        return []