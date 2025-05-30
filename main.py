from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from pydantic import BaseModel
from datetime import datetime, timedelta
import jwt
import bcrypt
import openai
import stripe
import os
import uvicorn
from typing import Optional

# Configuración
SECRET_KEY = "your-secret-key-change-this"
ALGORITHM = "HS256"
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///quickcontent.db")

# OpenAI y Stripe (usar variables de entorno en producción)
openai.api_key = os.getenv("OPENAI_API_KEY", "your-openai-key")
stripe.api_key = os.getenv("STRIPE_SECRET_KEY", "your-stripe-key")

# Base de datos
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Modelos de base de datos
class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    plan = Column(String, default="free")  # free, pro, business
    posts_generated = Column(Integer, default=0)
    stripe_customer_id = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    subscription_active = Column(Boolean, default=False)

class Post(Base):
    __tablename__ = "posts"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True)
    content = Column(Text)
    platform = Column(String)  # instagram, facebook, linkedin
    topic = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

# Crear tablas
Base.metadata.create_all(bind=engine)

# Schemas Pydantic
class UserCreate(BaseModel):
    email: str
    password: str

class UserLogin(BaseModel):
    email: str
    password: str

class ContentRequest(BaseModel):
    topic: str
    platform: str
    tone: str = "professional"

class Token(BaseModel):
    access_token: str
    token_type: str

# FastAPI app
app = FastAPI(title="QuickContent AI", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Servir archivos estáticos
app.mount("/static", StaticFiles(directory="static"), name="static")

# Dependencias
security = HTTPBearer()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=30)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security), db: Session = Depends(get_db)):
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise HTTPException(status_code=401, detail="Invalid token")
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    user = db.query(User).filter(User.email == email).first()
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    return user

# Funciones de negocio
def can_generate_post(user: User) -> bool:
    limits = {"free": 5, "pro": 100, "business": 500}
    return user.posts_generated < limits.get(user.plan, 5)

def generate_content_with_ai(topic: str, platform: str, tone: str) -> str:
    """Genera contenido usando OpenAI"""
    prompts = {
        "instagram": f"""Crea un post ESPECÍFICO para Instagram sobre {topic} con tono {tone}:
- Máximo 150 palabras
- Incluye 5-8 hashtags relevantes
- Usa emojis llamativos
- Estilo visual y atractivo
- Enfoque en lifestyle/inspiración
- Incluye call-to-action para engagement""",
        
        "facebook": f"""Crea un post ESPECÍFICO para Facebook sobre {topic} con tono {tone}:
- Entre 150-200 palabras
- Estilo conversacional y personal
- Haz preguntas para generar comentarios
- Incluye storytelling
- Enfoque en comunidad y discusión
- Sin hashtags (Facebook no los usa tanto)""",
        
        "linkedin": f"""Crea un post ESPECÍFICO para LinkedIn sobre {topic} con tono {tone}:
- Entre 200-300 palabras
- Enfoque profesional y de negocios
- Incluye insights valiosos
- Menciona tendencias de industria
- Formato de lista o puntos clave
- Call-to-action profesional"""
    }
    
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompts.get(platform, prompts["instagram"]),
            max_tokens=300,
            temperature=0.8
        )
        return response.choices[0].text.strip()
    except Exception as e:
        # Fallback DIFERENTE para cada plataforma
        fallbacks = {
            "instagram": f"🚀 {topic} es clave para el éxito ✨\n\nDescubre cómo puede transformar tu día a día. La innovación está en los detalles.\n\n¿Cuál es tu experiencia? 💭\n\n#marketing #exito #innovacion #emprendimiento #motivacion",
            "facebook": f"¡Hola amigos! Quería compartir algo interesante sobre {topic}.\n\nEs impresionante cómo este tema puede impactar nuestras vidas de maneras que no imaginamos. He estado investigando y los resultados son fascinantes.\n\n¿Ustedes qué opinan? ¿Han tenido experiencias similares? Me encantaría leer sus comentarios.",
            "linkedin": f"Reflexionando sobre {topic} y su impacto en el mundo empresarial.\n\nTres puntos clave que he observado:\n\n1. La transformación digital está redefiniendo las reglas\n2. Las empresas que se adaptan lideran el mercado\n3. La innovación constante es ahora una necesidad\n\n¿Cómo está impactando esto en tu industria? Comparte tu perspectiva."
        }
        return fallbacks.get(platform, fallbacks["instagram"])
    
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompts.get(platform, prompts["instagram"]),
            max_tokens=200,
            temperature=0.7
        )
        return response.choices[0].text.strip()
    except Exception as e:
        # Fallback si no hay API key o créditos
        return f"🚀 Contenido sobre {topic} para {platform}\n\n¡Hola! Hoy quiero compartir algo interesante sobre {topic}. Es increíble cómo este tema puede impactar nuestro día a día.\n\n¿Qué opinas al respecto? 💭\n\n#marketing #contenido #socialmedia"

# Rutas de API
@app.post("/api/register", response_model=Token)
async def register(user: UserCreate, db: Session = Depends(get_db)):
    # Verificar si el usuario ya existe
    if db.query(User).filter(User.email == user.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Crear usuario
    hashed_password = hash_password(user.password)
    db_user = User(email=user.email, hashed_password=hashed_password)
    db.add(db_user)
    db.commit()
    
    # Crear token
    access_token = create_access_token(data={"sub": user.email})
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/api/login", response_model=Token)
async def login(user: UserLogin, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.email == user.email).first()
    if not db_user or not verify_password(user.password, db_user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    access_token = create_access_token(data={"sub": user.email})
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/api/me")
async def get_user_info(current_user: User = Depends(get_current_user)):
    limits = {"free": 5, "pro": 100, "business": 500}
    return {
        "email": current_user.email,
        "plan": current_user.plan,
        "posts_generated": current_user.posts_generated,
        "posts_remaining": limits.get(current_user.plan, 5) - current_user.posts_generated,
        "subscription_active": current_user.subscription_active
    }

@app.post("/api/generate-content")
async def generate_content(
    request: ContentRequest, 
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Verificar límites
    if not can_generate_post(current_user):
        raise HTTPException(status_code=403, detail="Post limit reached. Upgrade your plan!")
    
    # Generar contenido
    content = generate_content_with_ai(request.topic, request.platform, request.tone)
    
    # Guardar en base de datos
    post = Post(
        user_id=current_user.id,
        content=content,
        platform=request.platform,
        topic=request.topic
    )
    db.add(post)
    
    # Actualizar contador
    current_user.posts_generated += 1
    db.commit()
    
    return {"content": content}

@app.get("/api/posts")
async def get_user_posts(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    posts = db.query(Post).filter(Post.user_id == current_user.id).order_by(Post.created_at.desc()).limit(20).all()
    return [{"id": p.id, "content": p.content, "platform": p.platform, "topic": p.topic, "created_at": p.created_at} for p in posts]

@app.post("/api/create-checkout-session")
async def create_checkout_session(plan: str, current_user: User = Depends(get_current_user)):
    prices = {
        "pro": "price_1234567890",  # Reemplazar con tus Price IDs de Stripe
        "business": "price_0987654321"
    }
    
    try:
        checkout_session = stripe.checkout.Session.create(
            payment_method_types=['card'],
            line_items=[{
                'price': prices[plan],
                'quantity': 1,
            }],
            mode='subscription',
            success_url='http://localhost:8000/success',
            cancel_url='http://localhost:8000/cancel',
            customer_email=current_user.email,
        )
        return {"checkout_url": checkout_session.url}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Ruta principal para servir el frontend
@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
