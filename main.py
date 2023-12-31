
"""
Sample Usage: http://127.0.0.1:8000/predict?prompt=What%20is%20the%20meaning%20of%20life&max_new_tokens=10&num_samples=1
# prompt, max_tokens, stop,  model parameter and switch between a given set of models

"""

from infer import generate_from_prompt, get_model
import uvicorn
# Using FastAPI over Flask as it generates documentation automatically
from fastapi import FastAPI, Response
from fastapi.logger import logger
from config import CONFIG, update_config


# Initialize API Server
app = FastAPI(
    title="NanoGPT",
    description="NanoGPT Serving endpoint, by Atul Dhingra",
    version="0.0.1",
    terms_of_service=None,
    contact=None,
    license_info=None
)

CONFIG = CONFIG.copy()

@app.get("/live")
def is_live():
    return Response(status_code=200)

@app.get("/ready")
def is_ready():
    if app.package["model"] :
        return Response(status_code=200)
    else:
        return Response(status_code=409)

@app.on_event("startup")
def startup_event():
    logger.info('PyTorch using device: {}'.format(CONFIG['DEVICE']))
    logger.info('Using: {} config'.format(CONFIG['INIT_FROM']))
    model, ctx = get_model(init_from=CONFIG['INIT_FROM'], device=CONFIG['DEVICE'])
    app.package = {
        "model": model,
        "ctx": ctx
    }

@app.post("/model")
def instantiate_new_model(init_from:str="gpt2-medium", device='cpu'):
    CONFIG = update_config(CONFIG, 'INIT_FROM', init_from)
    CONFIG = update_config(CONFIG, 'DEVICE', device)
    assert init_from in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
    model, ctx = get_model(CONFIG['INIT_FROM'], CONFIG['DEVICE'])
    app.package["model"] = model
    app.package["ctx"] = ctx


@app.get('/predict')
def predict(prompt: str = "\n", max_new_tokens: int = 10, num_samples:int = 1):
    """
    Perform prediction on input data
    """
    logger.info('API predict called')
    logger.info(f'input: {prompt}')

    # Run model inference
    y = generate_from_prompt(app.package["model"], app.package["ctx"],
                            start=prompt, max_new_tokens=max_new_tokens,
                            num_samples=num_samples, device=CONFIG['DEVICE'])
    # Prepare json for returning
    app.package.update({
        "prompt": prompt,
        "max_new_tokens": max_new_tokens,
        "num_samples": num_samples,
        "response_gpt": y
    }
)
    logger.info(f'Response: {y}')
    # TODO: Add streaming inferences
    return y

@app.on_event("shutdown")
async def shutdown_event():
    print('Shutdown')


if __name__ == '__main__':
    # server api
    uvicorn.run("main:app", host="127.0.0.1", reload=True, port=8000)