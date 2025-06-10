import os

from dotenv import load_dotenv

import jwt
import uvicorn
from fastapi import FastAPI
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import router

app = FastAPI(
    title='Avocado AI API',
    description='Mental health bot',
    version='0.1',
    docs_url='/',
)

app.include_router(router)

'''app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)'''

if __name__ == "__main__":
    print(os.getenv('ENV'))
    load_dotenv('../../.env')
    print('OPENAI_API_KEY: ', os.getenv('OPENAI_API_KEY'))
    uvicorn.run("app.main:app", host="0.0.0.0", port=1235, reload=True)


@app.exception_handler(jwt.ExpiredSignatureError)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=401,
        content={"message": "token has expired"},
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"message": "An unexpected error occurred.", "details": str(exc)},
    )
