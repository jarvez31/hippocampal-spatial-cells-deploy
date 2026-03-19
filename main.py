from fastapi import FastAPI
from pydantic import BaseModel
import analysis, model, data
import numpy as np
from typing import List

app = FastAPI()

# loads once when server starts
encoder_model = model.load_autoencoder("autoencoder_aligned_lattice")

class TrajectoryInput(BaseModel):
    trajectory: List[List[float]]  

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/encode")
def encode(input: TrajectoryInput) -> dict:
    trajectory = np.asarray(input.trajectory)
    PI_nor = analysis.compute_PI(trajectory)
    encoded_data = analysis.get_encoded(encoder_model, PI_nor)    

    return {"encoded_data": encoded_data.tolist()}

@app.post("/results")
def results(input: TrajectoryInput) -> dict:
    trajectory = np.asarray(input.trajectory)
    PI_nor = analysis.compute_PI(trajectory)
    encoded_data = analysis.get_encoded(encoder_model, PI_nor)    
    
    rate_maps, spatial_info = analysis.rate_map_analysis(encoded_data, trajectory)

    return {"rate_maps": rate_maps, "spatial_info": spatial_info}