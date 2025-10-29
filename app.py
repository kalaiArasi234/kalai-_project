# app.py — DR & OCT Screening with MFA (OTP)
# Sidebar removed; navigation in main area to avoid "columns in sidebar" error.

import os
import secrets
import hashlib
import sqlite3
from datetime import datetime, timedelta
from io import BytesIO

import numpy as np
from PIL import Image, ImageOps, ImageFilter
import streamlit as st
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# Deep learning (fundus real model + OCT hook)
import torch
import torch.nn.functional as F
from torchvision import models, transforms

# Optional animation lib (graceful fallback)
try:
    from streamlit_lottie import st_lottie
    import json as _json
except Exception:
    st_lottie, _json = None, None


# ================== CONFIG ==================
DB_PATH = os.environ.get("APP_DB_PATH", "app.db")

# OTP settings
OTP_EXP_MIN = 5
OTP_MAX_ATTEMPTS = 3
OTP_RESEND_COOLDOWN_SEC = 45
OTP_RATE_LIMIT_PER_HOUR = 5

# UI/dev toggles
DEV_SHOW_OTP_ON_PAGE = True  # show OTP in UI for local testing; set False in prod

# Classes
FUNDUS_CLASSES = ["No_DR", "Mild", "Moderate", "Severe", "Proliferative"]
OCT_CLASSES = ["Normal", "DME", "CNV", "Drusen"]

# Model weights (override via env)
WEIGHTS_FUNDUS = os.getenv("FUNDUS_WEIGHTS", "dr_resnet50.pth")
WEIGHTS_OCT    = os.getenv("OCT_WEIGHTS",    "oct_resnet18.pth")

# Assets (optional; app falls back if missing)
ASSETS = {
    "hero_dr": "assets/hero_dr.png",
    "hero_oct": "assets/hero_oct.png",
    "login_side": "assets/login_side.png",
    "eye_lottie": "assets/eye_lottie.json",
    "oct_lottie": "assets/oct_lottie.json",
}

# ================== THEME / CSS ==================
APP_CSS = """
<style>
.main > div { animation: fadein .5s ease-in; }
@keyframes fadein { from{opacity:0; transform:translateY(4px)} to{opacity:1; transform:none} }

h1.app-title {
  background: linear-gradient(90deg, #6ee7b7, #93c5fd, #fca5a5);
  -webkit-background-clip: text; background-clip: text; color: transparent;
}
.block-container { padding-top: 1.1rem !important; }

.card {
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 16px; padding: 18px 16px; backdrop-filter: blur(6px);
  transition: transform .15s ease, box-shadow .15s ease, border-color .15s ease;
}
.card:hover { transform: translateY(-2px); border-color: rgba(255,255,255,0.18); }
.stButton > button { border-radius: 9999px; transition: transform .08s ease, box-shadow .08s ease; }
.stButton > button:hover { transform: translateY(-1px); box-shadow: 0 8px 20px rgba(0,0,0,.25); }

[data-testid="stFileUploader"] { border-radius: 14px; }
[data-testid="stFileUploader"] > div:first-child {
  animation: pulse 2s infinite;
  border-radius: 14px;
}
@keyframes pulse {
  0% { box-shadow: 0 0 0 0 rgba(99,102,241,0.25); }
  70% { box-shadow: 0 0 0 10px rgba(99,102,241,0); }
  100% { box-shadow: 0 0 0 0 rgba(99,102,241,0); }
}
.badge {
  display:inline-block; padding: 2px 8px; border-radius:9999px;
  background: rgba(148,163,184,.2); font-size:.75rem;
  border: 1px solid rgba(148,163,184,.35);
}
</style>
"""


# ================== DB ==================
def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def init_db():
    conn = get_db()
    conn.execute("""
    CREATE TABLE IF NOT EXISTS users(
        user_id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT,
        phone TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        is_active INTEGER DEFAULT 1
    );
    """)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS otps(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        otp_hash TEXT NOT NULL,
        salt TEXT NOT NULL,
        expires_at TEXT NOT NULL,
        attempts_left INTEGER NOT NULL,
        request_id TEXT NOT NULL,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(user_id) REFERENCES users(user_id)
    );
    """)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS scans(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        modality TEXT NOT NULL,
        file_name TEXT,
        prediction TEXT,
        probs_json TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(user_id) REFERENCES users(user_id)
    );
    """)
    conn.commit(); conn.close()


# ================== OPTIONAL SENDERS (Email/SMS) ==================
def send_email_otp(to_addr: str, code: str):
    # Configure via env: SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS
    import smtplib
    from email.message import EmailMessage
    host = os.getenv("SMTP_HOST"); user = os.getenv("SMTP_USER"); pwd = os.getenv("SMTP_PASS")
    port = int(os.getenv("SMTP_PORT", "587"))
    if not (host and user and pwd):
        return False, "SMTP not configured"
    try:
        msg = EmailMessage()
        msg["Subject"] = "Your One-Time Code"
        msg["From"] = user
        msg["To"] = to_addr
        msg.set_content(f"Your OTP is {code}. It expires in {OTP_EXP_MIN} minutes.")
        with smtplib.SMTP(host, port) as s:
            s.starttls(); s.login(user, pwd); s.send_message(msg)
        return True, None
    except Exception as e:
        return False, str(e)

def send_sms_otp(phone: str, code: str):
    # Twilio (optional): TWILIO_SID, TWILIO_TOKEN, TWILIO_FROM
    try:
        from twilio.rest import Client
    except Exception:
        return False, "Twilio not installed"
    sid=os.getenv("TWILIO_SID"); tok=os.getenv("TWILIO_TOKEN"); from_ = os.getenv("TWILIO_FROM")
    if not (sid and tok and from_):
        return False, "Twilio not configured"
    try:
        Client(sid, tok).messages.create(to=phone, from_=from_, body=f"Your OTP: {code}")
        return True, None
    except Exception as e:
        return False, str(e)


# ================== AUTH (OTP) ==================
def _hash_code(code: str, salt: str) -> str:
    return hashlib.sha256((salt + code).encode()).hexdigest()

def _find_or_create_user(identifier: str):
    conn = get_db()
    if "@" in identifier:
        row = conn.execute("SELECT user_id FROM users WHERE email=?", (identifier,)).fetchone()
        uid = row[0] if row else conn.execute("INSERT INTO users(email) VALUES (?)", (identifier,)).lastrowid
    else:
        row = conn.execute("SELECT user_id FROM users WHERE phone=?", (identifier,)).fetchone()
        uid = row[0] if row else conn.execute("INSERT INTO users(phone) VALUES (?)", (identifier,)).lastrowid
    conn.commit(); conn.close()
    return uid

def _rate_limit_ok(uid:int)->bool:
    conn = get_db()
    one_hour_ago = (datetime.utcnow() - timedelta(hours=1)).isoformat()
    cnt = conn.execute("SELECT COUNT(*) FROM otps WHERE user_id=? AND created_at>=?", (uid, one_hour_ago)).fetchone()[0]
    conn.close()
    return cnt < OTP_RATE_LIMIT_PER_HOUR

def send_otp(identifier: str):
    uid = _find_or_create_user(identifier)
    if not _rate_limit_ok(uid):
        return None, "Too many OTP requests in the last hour. Please try later."
    conn = get_db()
    last = conn.execute("SELECT created_at FROM otps WHERE user_id=? ORDER BY id DESC LIMIT 1", (uid,)).fetchone()
    if last:
        delta = (datetime.utcnow() - datetime.fromisoformat(last[0])).total_seconds()
        if delta < OTP_RESEND_COOLDOWN_SEC:
            conn.close()
            return None, f"You can resend in {OTP_RESEND_COOLDOWN_SEC-int(delta)}s."

    code = f"{secrets.randbelow(1_000_000):06d}"
    salt = secrets.token_hex(8)
    expires_at = (datetime.utcnow() + timedelta(minutes=OTP_EXP_MIN)).isoformat()
    request_id = secrets.token_urlsafe(12)

    conn.execute("""INSERT INTO otps(user_id, otp_hash, salt, expires_at, attempts_left, request_id)
                    VALUES (?,?,?,?,?,?)""",
                 (uid, _hash_code(code, salt), salt, expires_at, OTP_MAX_ATTEMPTS, request_id))
    conn.commit(); conn.close()

    # DEV: print to console
    print(f"[DEV] OTP for {identifier} (user_id={uid}) → {code} (expires {OTP_EXP_MIN}m)")

    # Optional delivery (ignored errors in dev)
    if "@" in identifier:
        send_email_otp(identifier, code)
    else:
        send_sms_otp(identifier, code)

    return {"user_id": uid, "request_id": request_id, "code": code if DEV_SHOW_OTP_ON_PAGE else None}, None

def verify_otp(user_id:int, request_id:str, code:str):
    conn = get_db()
    row = conn.execute("""SELECT id, otp_hash, salt, expires_at, attempts_left
                          FROM otps WHERE user_id=? AND request_id=? ORDER BY id DESC LIMIT 1""",
                       (user_id, request_id)).fetchone()
    if not row:
        conn.close(); return False, "Invalid request. Please resend OTP."
    _id, otp_hash, salt, expires_at, attempts_left = row
    if datetime.utcnow() > datetime.fromisoformat(expires_at):
        conn.close(); return False, "Code expired, please request a new one."
    if attempts_left <= 0:
        conn.close(); return False, "Too many attempts. Please request a new OTP."
    if _hash_code(code, salt) == otp_hash:
        conn.execute("DELETE FROM otps WHERE id=?", (_id,)); conn.commit(); conn.close()
        return True, None
    attempts_left -= 1
    conn.execute("UPDATE otps SET attempts_left=? WHERE id=?", (attempts_left, _id))
    conn.commit(); conn.close()
    return False, f"Invalid code. Attempts left: {attempts_left}"


# ================== SESSION ==================
def set_session(user_id:int, identifier:str, as_guest:bool=False):
    st.session_state["auth.user_id"] = user_id
    st.session_state["auth.identifier"] = identifier
    st.session_state["auth.is_authenticated"] = True
    st.session_state["auth.is_guest"] = as_guest
    st.session_state.setdefault("guest_runs", 0)

def clear_session():
    for k in list(st.session_state.keys()):
        if k.startswith("auth.") or k.startswith("ui."):
            del st.session_state[k]

def is_authed(): 
    return st.session_state.get("auth.is_authenticated", False)


# ================== MODELS (REAL FUNDUS + OCT HOOK) ==================
def _pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

_DEVICE = _pick_device()
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

# FUNDUS
_fundus_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD)
])

def preprocess_fundus(img: Image.Image) -> torch.Tensor:
    if img.mode != "RGB":
        img = img.convert("RGB")
    return _fundus_tf(img)  # [3,224,224]

@st.cache_resource(show_spinner=False)
def load_fundus_model():
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(FUNDUS_CLASSES))
    try:
        state = torch.load(WEIGHTS_FUNDUS, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(WEIGHTS_FUNDUS, map_location="cpu")
    model.load_state_dict(state)
    model.eval().to(_DEVICE)
    return model

def predict_fundus(x: torch.Tensor):
    x = x.unsqueeze(0).to(_DEVICE)  # [1,3,224,224]
    model = load_fundus_model()
    with torch.no_grad():
        logits = model(x)
        probs  = F.softmax(logits, dim=1).cpu().numpy()[0].tolist()
    idx = int(np.argmax(probs))
    return FUNDUS_CLASSES[idx], probs

# OCT
_oct_tf = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.Lambda(lambda im: im.convert("L")),
    transforms.Lambda(lambda im: Image.merge("RGB", (im, im, im))),
    transforms.ToTensor(),
    transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD)
])

def preprocess_oct(img: Image.Image) -> torch.Tensor:
    return _oct_tf(img)  # [3,256,256]

@st.cache_resource(show_spinner=False)
def load_oct_model():
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(OCT_CLASSES))
    if os.path.exists(WEIGHTS_OCT):
        try:
            state = torch.load(WEIGHTS_OCT, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(WEIGHTS_OCT, map_location="cpu")
        model.load_state_dict(state)
    model.eval().to(_DEVICE)
    return model

def predict_oct(x_like):
    # accept tensor or numpy; convert to tensor
    if isinstance(x_like, np.ndarray):
        x = torch.from_numpy(x_like)
        if x.dim() == 3 and x.shape[0] in (1, 3):
            pass
        else:
            raise ValueError("OCT numpy should be CHW")
    elif isinstance(x_like, torch.Tensor):
        x = x_like
    else:
        raise ValueError("Unsupported input type for predict_oct")
    x = x.unsqueeze(0).to(_DEVICE)  # [1,3,256,256]
    model = load_oct_model()
    with torch.no_grad():
        logits = model(x)
        probs  = F.softmax(logits, dim=1).cpu().numpy()[0].tolist()
    idx = int(np.argmax(probs))
    return OCT_CLASSES[idx], probs


# ================== STORAGE / REPORT ==================
def save_scan(user_id:int, modality:str, file_name:str, prediction:str, probs:list):
    import json
    conn = get_db()
    conn.execute("""INSERT INTO scans(user_id, modality, file_name, prediction, probs_json)
                    VALUES (?,?,?,?,?)""",
                 (user_id, modality, file_name, prediction, json.dumps(probs)))
    conn.commit(); conn.close()

def load_history(user_id:int):
    conn = get_db()
    rows = conn.execute("""SELECT id, created_at, modality, file_name, prediction, probs_json
                           FROM scans WHERE user_id=? ORDER BY id DESC""",(user_id,)).fetchall()
    conn.close(); return rows

def build_pdf(identifier:str, modality:str, file_name:str, prediction:str, probs:list)->bytes:
    buf = BytesIO(); c = canvas.Canvas(buf, pagesize=A4); w,h = A4; y=h-50
    c.setFont("Helvetica-Bold", 18); c.drawString(40,y,"Diabetic Retinopathy / OCT Report"); y-=30
    c.setFont("Helvetica",11)
    c.drawString(40,y,f"User: {identifier}"); y-=16
    c.drawString(40,y,f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"); y-=16
    c.drawString(40,y,f"Modality: {modality.upper()}"); y-=16
    c.drawString(40,y,f"File: {file_name or '-'}"); y-=24
    c.setFont("Helvetica-Bold",12); c.drawString(40,y,f"Prediction: {prediction}"); y-=18
    c.setFont("Helvetica",11); c.drawString(40,y,"Class probabilities:"); y-=16
    classes = FUNDUS_CLASSES if modality=="fundus" else OCT_CLASSES
    for cls,p in zip(classes, probs): c.drawString(60,y,f"{cls:15s} {p*100:6.2f}%"); y-=14
    y-=20; c.setFont("Helvetica-Oblique",10)
    c.drawString(40,y,"Disclaimer: Screening tool only; not a medical diagnosis.")
    c.showPage(); c.save(); buf.seek(0); return buf.read()


# ================== UI HELPERS ==================
def load_lottie(path_key:str):
    if st_lottie is None or _json is None: return None
    p = ASSETS.get(path_key)
    if not p or not os.path.exists(p): return None
    try:
        with open(p,"r",encoding="utf-8") as f: return _json.load(f)
    except Exception: return None

def hero_section():
    st.markdown(APP_CSS, unsafe_allow_html=True)
    st.markdown('<h1 class="app-title">Diabetic Retinopathy & OCT Screening</h1>', unsafe_allow_html=True)
    st.write("Upload retinal or OCT images for quick local screening. MFA protected.")
    st.markdown('<span class="badge">ResNet-based • Local Inference • MFA-secured</span>', unsafe_allow_html=True)
    col1, col2 = st.columns([1.2, 1])
    with col2:
        anim = load_lottie("eye_lottie")
        if anim: st_lottie(anim, height=180, speed=1, loop=True, quality="high")
        elif os.path.exists(ASSETS["hero_dr"]): st.image(ASSETS["hero_dr"], use_container_width=True)

def oct_banner():
    col1, col2 = st.columns([1,1])
    with col2:
        anim = load_lottie("oct_lottie")
        if anim: st_lottie(anim, height=180, speed=1, loop=True)
        elif os.path.exists(ASSETS["hero_oct"]): st.image(ASSETS["hero_oct"], use_container_width=True)


# ================== PAGES ==================
def page_login():
    hero_section()
    st.divider()

    left, right = st.columns([1.2, 0.8])

    # --- SEND OTP FORM ---
    with left:
        st.subheader("Sign in")
        st.caption("Use email or phone to receive a one-time code (OTP).")
        with st.form("send_otp_form", clear_on_submit=False):
            identifier = st.text_input("Email or phone", placeholder="you@example.com or 9876543210")
            send = st.form_submit_button("Send OTP")
        if send and identifier:
            result, err = send_otp(identifier.strip())
            if err:
                st.error(err)
            else:
                st.session_state["auth.pending"] = {
                    "identifier": identifier.strip(),
                    "user_id": result["user_id"],
                    "request_id": result["request_id"],
                    "sent_at": datetime.utcnow().isoformat()
                }
                st.success("OTP sent. Check your email/SMS (dev: console).")
                if DEV_SHOW_OTP_ON_PAGE and result.get("code"):
                    st.info(f"DEV OTP: {result['code']}")

        # --- VERIFY OTP FORM ---
        if "auth.pending" in st.session_state:
            st.write("")
            st.markdown("**Enter OTP**")
            with st.form("verify_otp_form", clear_on_submit=False):
                code = st.text_input("6-digit code", max_chars=6)
                verify = st.form_submit_button("Verify")
            if verify:
                p = st.session_state["auth.pending"]
                ok, err = verify_otp(p["user_id"], p["request_id"], code.strip())
                if ok:
                    set_session(p["user_id"], p["identifier"])
                    del st.session_state["auth.pending"]
                    st.success("Welcome! Signed in successfully.")
                    st.balloons()
                    st.rerun()
                else:
                    st.error(err)

            if st.button("Resend OTP"):
                ident = st.session_state["auth.pending"]["identifier"]
                result, err = send_otp(ident)
                if err:
                    st.error(err)
                else:
                    st.session_state["auth.pending"].update({
                        "user_id": result["user_id"],
                        "request_id": result["request_id"],
                        "sent_at": datetime.utcnow().isoformat()
                    })
                    st.success("OTP re-sent.")

        # --- GUEST FORM ---
        st.write("")
        st.caption("Prefer to try without sign-in?")
        with st.form("guest_form"):
            go_guest = st.form_submit_button("Continue as guest (limited)")
        if go_guest:
            gid = _find_or_create_user("guest@example.com")
            set_session(gid, "guest@example.com", as_guest=True)
            st.rerun()

    with right:
        if os.path.exists(ASSETS["login_side"]):
            st.image(ASSETS["login_side"], caption="AI-assisted retinal screening", use_container_width=True)
        else:
            st.info("Tip: add an illustration as assets/login_side.png for this panel.")


def page_scan():
    # All content in main area (no sidebar anywhere)
    hero_section()
    st.divider()

    is_guest = st.session_state.get("auth.is_guest", False)
    colm = st.columns([1,1,1])
    with colm[0]:
        st.metric("Status", "Signed in" if not is_guest else "Guest", help="Guest is limited to 1 Fundus run")
    with colm[1]: st.metric("Models", "Fundus + OCT")
    with colm[2]: st.metric("Security", "MFA", help="OTP login enabled")

    scan_type = st.radio("Choose modality", ["Fundus", "OCT"], horizontal=True, index=0)
    if is_guest and scan_type == "OCT":
        st.info("OCT is disabled for guests. Please login."); scan_type = "Fundus"

    if scan_type == "OCT": oct_banner()
    cols = st.columns([1.2, .8])
    with cols[0]:
        st.markdown("#### Upload Image")
        uploaded = st.file_uploader("PNG/JPG", type=["png","jpg","jpeg"], accept_multiple_files=False)
        if uploaded:
            img = Image.open(uploaded)
            st.image(img, caption="Preview", use_container_width=True)
            if scan_type == "Fundus":
                x = preprocess_fundus(img)
                go = st.button("Analyze Fundus", type="primary", use_container_width=True)
                if go:
                    if is_guest and st.session_state.get("guest_runs",0) >= 1:
                        st.warning("Guest limit reached. Please login to continue.")
                    else:
                        pred, probs = predict_fundus(x)
                        st.session_state["ui.prediction_result"] = {
                            "modality":"fundus","file_name":uploaded.name,"prediction":pred,"probs":probs
                        }
                        if is_guest: st.session_state["guest_runs"] += 1
                        st.success(f"Prediction: {pred}")
            else:
                x = preprocess_oct(img)
                go = st.button("Analyze OCT", type="primary", use_container_width=True)
                if go:
                    pred, probs = predict_oct(x)  # tensor accepted
                    st.session_state["ui.prediction_result"] = {
                        "modality":"oct","file_name":uploaded.name,"prediction":pred,"probs":probs
                    }
                    st.success(f"Prediction: {pred}")
    with cols[1]:
        st.markdown("#### Results")
        if "ui.prediction_result" not in st.session_state:
            st.info("Upload an image and click **Analyze** to see results here.")
        else:
            res = st.session_state["ui.prediction_result"]
            st.success(f"**{res['modality'].upper()} → {res['prediction']}**")
            classes = FUNDUS_CLASSES if res["modality"]=="fundus" else OCT_CLASSES
            st.write("**Probabilities**")
            st.table({ "Class": classes, "Probability": [f"{p*100:.2f}%" for p in res["probs"]] })
            st.caption("_Disclaimer: Screening tool only; not a medical diagnosis._")
            c1,c2,c3 = st.columns(3)
            with c1:
                st.button("Run another", on_click=lambda: st.session_state.pop("ui.prediction_result", None))
            with c2:
                st.download_button(
                    "Download PDF",
                    data=build_pdf(st.session_state["auth.identifier"], res["modality"], res["file_name"], res["prediction"], res["probs"]),
                    file_name="report.pdf",
                    mime="application/pdf"
                )
            with c3:
                st.button("Save to history",
                          disabled=st.session_state.get("auth.is_guest", False),
                          on_click=lambda: save_scan(
                              st.session_state["auth.user_id"], res["modality"], res["file_name"],
                              res["prediction"], res["probs"])
                          )


def page_history():
    st.header("🗂 History")
    if st.session_state.get("auth.is_guest", False):
        st.info("History is only for logged-in users.")
        return
    rows = load_history(st.session_state["auth.user_id"])
    if not rows:
        st.info("No scans yet."); return
    import json
    for (id_, created_at, modality, fname, pred, probs_json) in rows:
        probs = json.loads(probs_json); classes = FUNDUS_CLASSES if modality=="fundus" else OCT_CLASSES
        with st.expander(f"[{created_at}] {modality.upper()} — {pred}  •  {fname or '-'}"):
            st.table({"Class": classes, "Probability":[f"{p*100:.2f}%" for p in probs]})


# ================== ROUTER (main-only navigation) ==================
def router_main_only():
    """All navigation + pages render in the main area only (no sidebar)."""
    if not is_authed():
        page_login()
        return

    st.markdown(APP_CSS, unsafe_allow_html=True)

    # Top navigation (main area)
    nav_col, action_col = st.columns([0.85, 0.15])
    with nav_col:
        selected = st.radio(
            "Navigation",
            ["Scan", "History", "Logout"],
            horizontal=True,
            index=0,
            label_visibility="collapsed",
        )
    with action_col:
        st.caption("")

    if selected == "Logout":
        clear_session()
        st.success("Logged out.")
        st.rerun()
        return

    if selected == "Scan":
        page_scan()
    elif selected == "History":
        page_history()
    else:
        page_scan()


# ================== MAIN ==================
def main():
    st.set_page_config(page_title="DR & OCT Screening (MFA)", page_icon="👁️", layout="wide")
    init_db()
    router_main_only()

if __name__ == "__main__":
    main()
