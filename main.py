# main.py
import os
import numpy as np
from scipy.optimize import curve_fit
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from typing import List
from fastapi.middleware.cors import CORSMiddleware

API_KEY = os.getenv("CLASSIC_TOKEN")

app = FastAPI(title="4PL Curve Fitting API")
origins = [
    
    "https://your-frontend-domain.com" ,
     "https://script.google.com",  
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,       
    allow_credentials=True,
    allow_methods=["*"],         
    allow_headers=["*"],        
)
class FourPLRequest(BaseModel):
    x_data: List[float]
    y_data: List[float]

class FourPLResponse(BaseModel):
    a: float
    b: float
    c: float
    d: float
    r2: float


def fourPL(x, a, b, c, d):
    """Four-parameter logistic function"""
    return d + (a - d) / (1.0 + (x / c) ** b)

def fit_fourPL(x_data: np.ndarray, y_data: np.ndarray):
    a_guess = float(np.min(y_data))
    d_guess = float(np.max(y_data))
    c_guess = float(np.median(x_data))
    b_guess = -1.0
    initial_guesses = [a_guess, b_guess, c_guess, d_guess]

    popt, _ = curve_fit(
        fourPL, x_data, y_data,
        p0=initial_guesses,
        bounds=([-np.inf, -np.inf, 0, -np.inf], [np.inf, 0, np.inf, np.inf]),
        maxfev=10000
    )

    a, b, c, d = popt

    y_pred = fourPL(x_data, *popt)
    ss_res = np.sum((y_data - y_pred) ** 2)
    ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    return a, b, c, d, r_squared


@app.post("/fit_4pl", response_model=FourPLResponse)
def fit_4pl_endpoint(payload: FourPLRequest, x_api_key: str = Header(None)):
    if API_KEY is None:
        raise HTTPException(status_code=500, detail="API key not configured on server")

    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Unauthorized")
    x = np.array(payload.x_data, dtype=float)
    y = np.array(payload.y_data, dtype=float)

    a, b, c, d, r2 = fit_fourPL(x, y)

    return FourPLResponse(a=a, b=b, c=c, d=d, r2=r2)


@app.get("/ping")
def ping():
    print("Ping endpoint was called!")
    return {"status": "ok"}
