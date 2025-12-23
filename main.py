from fastapi import FastAPI

from analyze import router as analyze_router
from checklist import router as checklist_router
from loan import router as loan_router
from solution import router as solution_router

app = FastAPI()

app.include_router(checklist_router)
app.include_router(analyze_router)
app.include_router(loan_router)
app.include_router(solution_router)
