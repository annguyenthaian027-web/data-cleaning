
import os
from datetime import datetime, timedelta, timezone

from jose import JWTError, jwt
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import OAuth2PasswordBearer
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from passlib.context import CryptContext
from bson import ObjectId
from bson.errors import InvalidId

from backend.database import check_database_connection, users_collection

# Cau hinh ma hoa mat khau
pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto"
)

# Cau hinh JWT
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "data_cleaning_secret_key_2026")  # đặt JWT_SECRET_KEY khi triển khai
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/login")


# Tao JWT Token
def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(
        minutes=ACCESS_TOKEN_EXPIRE_MINUTES
    )
    to_encode.update({"exp": expire})

    return jwt.encode(
        to_encode,
        SECRET_KEY,
        algorithm=ALGORITHM
    )


# Lay thong tin nguoi dung tu Token
def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(
            token,
            SECRET_KEY,
            algorithms=[ALGORITHM]
        )

        user_id = payload.get("sub")

        if user_id is None:
            raise HTTPException(
                status_code=401,
                detail="Token khong hop le"
            )

        user = users_collection.find_one({
            "_id": ObjectId(user_id)
        })

        if not user:
            raise HTTPException(
                status_code=401,
                detail="Nguoi dung khong ton tai"
            )

        return user

    except (JWTError, InvalidId):
        raise HTTPException(
            status_code=401,
            detail="Token khong hop le hoac het han"
        )


def require_admin(current_user: dict = Depends(get_current_user)):
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Chi quan tri vien moi duoc phep")
    return current_user


# Cau truc du lieu dang ky
class UserRegister(BaseModel):
    full_name: str
    email: str
    password: str

# Cau truc du lieu dang nhap
class UserLogin(BaseModel):
    email: str
    password: str
app = FastAPI(
    title="Data Cleaning API",
    description="Backend nen tang lam sach du lieu Excel va CSV",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {
        "message": "Data Cleaning Backend dang hoat dong!",
        "status": "success"
    }

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "message": "Backend hoat dong binh thuong"
    }


@app.get("/api/users")
def get_users(admin: dict = Depends(require_admin)):
    users = []

    for user in users_collection.find():
        users.append({
            "id": str(user["_id"]),
            "full_name": user["full_name"],
            "email": user["email"],
            "role": user["role"]
        })

    return {
        "message": "Lay danh sach nguoi dung thanh cong",
        "total": len(users),
        "users": users
    }

@app.get("/api/health/db")
def check_db():
    if check_database_connection():
        return {
            "status": "success",
            "message": "Ket noi MongoDB thanh cong!"
        }

    return {
        "status": "error",
        "message": "Khong the ket noi MongoDB!"
    }

@app.post("/api/register")
def register_user(user: UserRegister):
    if len(user.password) < 6:
        raise HTTPException(status_code=400, detail="Mat khau toi thieu 6 ky tu")
    # Kiem tra email da ton tai
    existing_user = users_collection.find_one({
        "email": user.email
    })

    if existing_user:
        raise HTTPException(
            status_code=400,
            detail="Email da duoc dang ky"
        )

    # Ma hoa mat khau
    hashed_password = pwd_context.hash(user.password)

    # Tao tai khoan moi
    new_user = {
        "full_name": user.full_name,
        "email": user.email,
        "password": hashed_password,
        "role": "user"
    }

    result = users_collection.insert_one(new_user)

    return {
        "message": "Dang ky thanh cong",
        "user_id": str(result.inserted_id),
        "full_name": user.full_name,
        "email": user.email,
        "role": "user"
    }


@app.post("/api/login")
def login_user(user: UserLogin):
    existing_user = users_collection.find_one({
        "email": user.email
    })

    if not existing_user:
        raise HTTPException(
            status_code=401,
            detail="Email hoac mat khau khong dung"
        )

    if not pwd_context.verify(
        user.password,
        existing_user["password"]
    ):
        raise HTTPException(
            status_code=401,
            detail="Email hoac mat khau khong dung"
        )

    access_token = create_access_token({
        "sub": str(existing_user["_id"]),
        "role": existing_user["role"]
    })

    return {
        "message": "Dang nhap thanh cong",
        "access_token": access_token,
        "token_type": "bearer",
        "user_id": str(existing_user["_id"]),
        "full_name": existing_user["full_name"],
        "email": existing_user["email"],
        "role": existing_user["role"]
    }


@app.get("/api/users/{user_id}")
def get_user(user_id: str, admin: dict = Depends(require_admin)):
    try:
        user = users_collection.find_one({
            "_id": ObjectId(user_id)
        })
    except InvalidId:
        raise HTTPException(
            status_code=400,
            detail="User ID khong hop le"
        )

    if not user:
        raise HTTPException(
            status_code=404,
            detail="Khong tim thay nguoi dung"
        )

    return {
        "id": str(user["_id"]),
        "full_name": user["full_name"],
        "email": user["email"],
        "role": user["role"]
    }
@app.get("/api/me")
def get_my_profile(current_user: dict = Depends(get_current_user)):
    return {
        "message": "Xac thuc thanh cong",
        "user_id": str(current_user["_id"]),
        "full_name": current_user["full_name"],
        "email": current_user["email"],
        "role": current_user["role"]
    }

# ======================= API làm sạch dữ liệu (dùng core + lớp AI) =======================
import json
from fastapi import File, Form, UploadFile
from fastapi.responses import Response

from dataclean import CleaningSession, Rule, analyze, export_session, load_dataframe
from dataclean_ai import AIAdvisor, AIConfig

_advisor = AIAdvisor(AIConfig.from_env())


def _read(upload: UploadFile):
    return load_dataframe(upload.file.read(), upload.filename)


@app.post("/api/clean/analyze")
def clean_analyze(file: UploadFile = File(...), user: dict = Depends(get_current_user)):
    df = _read(file)
    rep = analyze(df)
    rv = _advisor.review(df, rep)
    return {"report": rep, "ai": rv.to_dict(), "ai_mode": _advisor.mode}


@app.post("/api/clean/export")
def clean_export(file: UploadFile = File(...), rules: str = Form("[]"), fmt: str = Form("xlsx"),
                 user: dict = Depends(get_current_user)):
    df = _read(file)
    s = CleaningSession(df, file.filename)
    s.apply([Rule.from_dict(r) for r in json.loads(rules)], "Làm sạch qua API")
    data, name, mime = export_session(s, fmt)
    return Response(content=data, media_type=mime,
                    headers={"Content-Disposition": f'attachment; filename="{name}"'})


# ======================= Quan ly user (admin) + tai khoan admin mac dinh =======================
class RoleUpdate(BaseModel):
    role: str


@app.put("/api/users/{user_id}/role")
def set_user_role(user_id: str, body: RoleUpdate, admin: dict = Depends(require_admin)):
    if body.role not in ("user", "admin"):
        raise HTTPException(status_code=400, detail="Role phai la user hoac admin")
    try:
        oid = ObjectId(user_id)
    except InvalidId:
        raise HTTPException(status_code=400, detail="User ID khong hop le")
    res = users_collection.update_one({"_id": oid}, {"$set": {"role": body.role}})
    if res.matched_count == 0:
        raise HTTPException(status_code=404, detail="Khong tim thay nguoi dung")
    return {"message": "Da cap nhat quyen", "user_id": user_id, "role": body.role}


@app.delete("/api/users/{user_id}")
def delete_user(user_id: str, admin: dict = Depends(require_admin)):
    if user_id == str(admin["_id"]):
        raise HTTPException(status_code=400, detail="Khong the tu xoa tai khoan dang dang nhap")
    try:
        oid = ObjectId(user_id)
    except InvalidId:
        raise HTTPException(status_code=400, detail="User ID khong hop le")
    if users_collection.delete_one({"_id": oid}).deleted_count == 0:
        raise HTTPException(status_code=404, detail="Khong tim thay nguoi dung")
    return {"message": "Da xoa nguoi dung"}


@app.on_event("startup")
def seed_admin():
    """Dat ADMIN_EMAIL + ADMIN_PASSWORD de tu tao tai khoan admin lan dau."""
    email, pw = os.getenv("ADMIN_EMAIL"), os.getenv("ADMIN_PASSWORD")
    if email and pw and not users_collection.find_one({"email": email}):
        users_collection.insert_one({"full_name": "Quan tri vien", "email": email,
                                     "password": pwd_context.hash(pw), "role": "admin"})
