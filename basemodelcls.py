from pydantic import BaseModel
class CropInput(BaseModel):
    country: str
    crop: str
    year: int
    pesticides: float
    rainfall: float
    temperature: float