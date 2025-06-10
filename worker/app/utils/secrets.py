import json
import os
from typing import Optional

import boto3
import jwt
from anthropic import Anthropic
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from openai import OpenAI
from pinecone import Pinecone

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


def decode_token(token):
    try:
        payload = jwt.decode(token, os.getenv('JWT_SECRET_KEY'), algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        raise jwt.ExpiredSignatureError("Token has expired")
    except jwt.InvalidTokenError:
        raise Exception("Invalid token")


def get_current_user_id(token: str = Depends(oauth2_scheme)) -> str:
    payload = decode_token(token)

    user_id: Optional[str] = payload.get("user_id")

    if user_id is None:
        raise HTTPException(
            status_code=401, detail="Invalid token: user_id not found"
        )

    return user_id


def get_open_ai_client():
    load_dotenv('../.env')
    return OpenAI(api_key=os.environ['OPENAI_API_KEY'])


def get_claude_client():
    load_dotenv('../.env')
    return Anthropic(api_key=os.getenv('CLAUDE_API_KEY'))


def get_pinecone_client():
    return Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
