from datetime import datetime, timedelta
from typing import Optional, Dict, List, Callable
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import os
SECRET_KEY=os.getenv("OPE_JWT_SECRET","change-me")
ALGORITHM="HS256"; ACCESS_TOKEN_EXPIRE_MINUTES=30
pwd_context=CryptContext(schemes=["bcrypt"], deprecated="auto")
security=HTTPBearer()
_users={"quant_trader":{"username":"quant_trader","hashed_password":pwd_context.hash("trader123"),"permissions":["pricing:read","pricing:write","risk:read"]}}
class User:
    def __init__(self,username:str,permissions:List[str]): self.username=username; self.permissions=permissions
def authenticate_user(u:str,p:str)->Optional[Dict]:
    d=_users.get(u); 
    if not d or not pwd_context.verify(p,d["hashed_password"]): return None
    return d
def create_access_token(data:dict,expires_delta:Optional[timedelta]=None)->str:
    to_encode={**data}; exp=datetime.utcnow()+(expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)); to_encode.update({"exp":exp}); return jwt.encode(to_encode,SECRET_KEY,algorithm=ALGORITHM)
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security))->User:
    exc=HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials", headers={"WWW-Authenticate":"Bearer"})
    try:
        payload=jwt.decode(credentials.credentials,SECRET_KEY,algorithms=[ALGORITHM]); sub=payload.get("sub")
        if not sub: raise exc
    except JWTError: raise exc
    d=_users.get(sub); 
    if not d: raise exc
    return User(sub,d.get("permissions",[]))
def require_permission(perm:str)->Callable[[User],User]:
    def _dep(user:User=Depends(get_current_user))->User:
        if perm not in user.permissions: raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=f"Permission '{perm}' required")
        return user
    return _dep
