
# prompt, max_tokens, stop,  model parameter and switch between a given set of models
#  http://127.0.0.1:8000/predict?prompt="What is the answer to life, the universe, and everything?"
# import os
# import sys
# import traceback
# from joblib import load
from infer import generate_from_prompt, get_model

import uvicorn
# Using FastAPI over Flask as it generates documentation automatically
from fastapi import FastAPI, Request, Body
from fastapi.logger import logger
# from sample import num_samples
# from fastapi.encoders import jsonable_encoder
# from fastapi.responses import RedirectResponse, JSONResponse
# from fastapi.exceptions import RequestValidationError
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.staticfiles import StaticFiles

# import torch
# from model import GPT
# from model import Model
# from predict import predict
# from config import CONFIG
# from exception_handler import validation_exception_handler, python_exception_handler
# from schema import *


# Initialize API Server
app = FastAPI()


# # Load custom exception handlers
# app.add_exception_handler(RequestValidationError, validation_exception_handler)
# app.add_exception_handler(Exception, python_exception_handler)

@app.on_event("startup")
def startup_event():
    # Initialize the pytorch model
    model, ctx = get_model()
    app.package = {
        "model": model,
        "ctx": ctx
    }


@app.get('/predict')
def predict(prompt: str = "\n", max_new_tokens: int = 10, num_samples:int = 1):
    """
    Perform prediction on input data
    """
    logger.info('API predict called')
    logger.info(f'input: {prompt}')

    # run model inference
    y = generate_from_prompt(app.package["model"], app.package["ctx"],
                             start=prompt, max_new_tokens=max_new_tokens, num_samples=num_samples)
        # prepare json for returning

    logger.info(f'Response: {y}')

    return {
        "prompt": prompt,
        "max_new_tokens": max_new_tokens,
        "num_samples": num_samples, 
        "response_gpt": y
    }


# @app.get('/about')
# def show_about():
#     """
#     Get deployment information, for debugging
#     """

#     def bash(command):
#         output = os.popen(command).read()
#         return output

#     return {
#         "sys.version": sys.version,
#         "torch.__version__": torch.__version__,
#         "torch.cuda.is_available()": torch.cuda.is_available(),
#         "torch.version.cuda": torch.version.cuda,
#         "torch.backends.cudnn.version()": torch.backends.cudnn.version(),
#         "torch.backends.cudnn.enabled": torch.backends.cudnn.enabled,
#         "nvidia-smi": bash('nvidia-smi')
#     }


if __name__ == '__main__':
    # server api
    uvicorn.run("main:app", host="127.0.0.1", port=8000)