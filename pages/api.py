from fastapi import FastAPI
from forecast_engine import get_forecast

app = FastAPI()

@app.get("/forecast")
def forecast(mandate_id: str):
    result = get_forecast(mandate_id)
    return result
