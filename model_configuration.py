from typing import Optional, List
from sqlalchemy import Column, JSON
from sqlmodel import SQLModel, Field
from pydantic import BaseModel

class ModelConfiguration(SQLModel):
    user_option_values: Optional[dict] = Field(sa_column=Column(JSON), default_factory=dict)
    additional_continuous_covariates: List[str] = Field(default_factory=list, sa_column=Column(JSON))

# two seperate classes of same name 
class ModelConfiguration(BaseModel):
    alpha: float = 0.1