from fastapi import APIRouter, Depends
from ..security import require_permission

router = APIRouter(
    prefix="/risk", tags=["risk"], dependencies=[Depends(require_permission("risk:read"))]
)
