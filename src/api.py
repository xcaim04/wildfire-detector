import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List

from model import PolynomialNeuralNetwork  # tu clase definida en model.py


app = FastAPI(title="Polynomial Neural Network API - Sin persistencia")
