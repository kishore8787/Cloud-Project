from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from datetime import timedelta
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

df = None
model = None
forecast_df = None
mae = None


def train_model(data: pd.DataFrame):
    # Ensure correct columns exist
    if "date" not in data.columns or "cost" not in data.columns:
        raise ValueError("CSV must contain 'date' and 'cost' columns.")

    # Parse and clean data
    data["date"] = pd.to_datetime(data["date"], format="%m/%d/%Y", errors="coerce")
    data = data.dropna(subset=["date", "cost"])
    data["cost"] = pd.to_numeric(data["cost"], errors="coerce").fillna(0)

    # Aggregate cost by date
    data = data.groupby("date", as_index=False)["cost"].sum()
    data = data.sort_values("date")

    # Prepare features
    data["day_ordinal"] = data["date"].map(pd.Timestamp.toordinal)
    X = data[["day_ordinal"]]
    y = data["cost"]

    # Train model
    model = LinearRegression().fit(X, y)

    # Forecast for next 7 days
    last_date = data["date"].max()
    future_dates = [last_date + timedelta(days=i) for i in range(1, 8)]
    future_ord = np.array([d.toordinal() for d in future_dates]).reshape(-1, 1)
    future_pred = model.predict(future_ord)

    forecast_df = pd.DataFrame({
        "date": [d.strftime("%Y-%m-%d") for d in future_dates],
        "predicted_cost": np.round(future_pred, 2)
    })

    mae = mean_absolute_error(y, model.predict(X))
    return data, model, forecast_df, mae


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    global df, model, forecast_df, mae
    try:
        contents = await file.read()
        df_upload = pd.read_csv(io.StringIO(contents.decode("utf-8")))
        df, model, forecast_df, mae = train_model(df_upload)
        return {"message": "File processed successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/predict")
async def get_predictions():
    if forecast_df is None:
        raise HTTPException(status_code=400, detail="No data available")
    return {"mae": mae, "forecast": forecast_df.to_dict("records")}


@app.get("/plot")
async def get_plot():
    if df is None or forecast_df is None:
        raise HTTPException(status_code=400, detail="No data to plot")

    plt.figure(figsize=(10, 4))
    plt.plot(df["date"], df["cost"], label="Actual Cost", marker="o")
    plt.plot(pd.to_datetime(forecast_df["date"]), forecast_df["predicted_cost"],
             "r--", label="Predicted (7 days)")
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Cost ($)")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close()

    return FileResponse(buf, media_type="image/png")


@app.get("/budget")
async def check_budget(budget: float):
    if forecast_df is None:
        raise HTTPException(status_code=400, detail="No prediction data available")

    pred_total = forecast_df["predicted_cost"].sum()
    alert_level = "low"
    message = f"Predicted spend ${pred_total:.2f} is within your budget (${budget:.2f})."

    if pred_total > budget * 1.10:
        alert_level = "high"
        message = f"Predicted spend ${pred_total:.2f} exceeds budget by >10%."
    elif pred_total > budget:
        alert_level = "medium"
        message = f"Predicted spend ${pred_total:.2f} slightly exceeds your budget (${budget:.2f})."

    recommendation = None
    if pred_total > budget:
        diff = pred_total - budget
        hours_to_reduce = round(diff / 5, 1)
        recommendation = (
            f"Reduce VM runtime by approx {hours_to_reduce} hours/day "
            f"or optimize storage to save costs."
        )

    return {
        "predicted_total": pred_total,
        "budget": budget,
        "alert_level": alert_level,
        "message": message,
        "recommendation": recommendation
    }


@app.get("/estimate")
async def estimate_cost(service: str, usage: float):
    service_map = {
        "vm": ("Virtual Machine (Compute)", 0.08),
        "storage": ("Storage (Blob / Disk)", 0.02),
        "sql": ("SQL Database", 0.15)
    }

    if service not in service_map:
        raise HTTPException(status_code=400, detail="Invalid service")

    service_name, rate = service_map[service]
    cost = usage * rate

    return {"service": service_name, "cost": cost}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
