from pydantic import BaseModel

class ActionSchema(BaseModel):
    analysis: str
    action: list[float]
    
class ActionSchemaCheck(BaseModel):
    analysis: str
    correct: bool
    action: list[float]

class VideoActionSchema(BaseModel):
    analysis: str
    correct: bool
    action: list[float]

class DataConstructionSchema(BaseModel):
    analysis: str
    response: str