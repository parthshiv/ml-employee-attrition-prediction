from pydantic import BaseModel

class EmployeeInput(BaseModel):
    age: int
    salary: float
    years_at_company: int
    job_satisfaction: int