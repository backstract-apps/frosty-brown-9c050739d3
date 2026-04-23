from pydantic import BaseModel,Field,field_validator

import datetime

import uuid

from typing import Any, Dict, List,Optional,Tuple,Union

import re

class Users(BaseModel):
    email: str
    password: str
    phone: str


class ReadUsers(BaseModel):
    email: str
    password: str
    phone: str
    class Config:
        from_attributes = True




class PostUsers(BaseModel):
    email: str = Field(..., max_length=255)
    password: str = Field(..., max_length=255)
    phone: Optional[str]=None
    created_at: Optional[Any]=None

    class Config:
        from_attributes = True



class PutUsersId(BaseModel):
    id: str = Field(..., max_length=100)
    email: str = Field(..., max_length=255)
    password: str = Field(..., max_length=255)
    phone: Optional[str]=None
    created_at: Optional[Any]=None

    class Config:
        from_attributes = True



# Query Parameter Validation Schemas

class GetUsersIdQueryParams(BaseModel):
    """Query parameter validation for get_users_id"""
    id: int = Field(..., ge=1, description="Id")

    class Config:
        populate_by_name = True


class DeleteUsersIdQueryParams(BaseModel):
    """Query parameter validation for delete_users_id"""
    id: int = Field(..., ge=1, description="Id")

    class Config:
        populate_by_name = True
