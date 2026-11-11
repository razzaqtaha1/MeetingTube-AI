import html
import re
import time

import streamlit as st
from dotenv import load_dotenv

from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import extract_all
from core.rag_engine import build_rag_chain, ask_question

load_dotenv()

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Video Assistant",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS (Light Theme) ───────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css');
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

/* ═════════ Design Tokens ═════════ */
:root {
    --bg: #f6f7fb;
    --surface: #ffffff;
    --surface-2: #f1f2f9;
    --border: #e4e6f2;
    --border-hover: #b9a8ff;
    --accent: #6d4aff;
    --accent-soft: #efeaff;
    --accent-2: #0ea5e9;
    --accent-2-soft: #e0f4fe;
    --accent-3: #ec4899;
    --text: #1d1b3b;
    --text-muted: #6b6a8c;
    --success: #10b981;
    --success-soft: #dcfaee;
    --warning: #f59e0b;
    --warning-soft: #fef3d6;
    --pink-soft: #fde6f2;
    --danger: #ef4444;
    --radius: 20px;
    --shadow-sm: 0 1px 2px rgba(29, 27, 59, .04), 0 4px 14px rgba(29, 27, 59, .05);
    --shadow-md: 0 10px 30px -8px rgba(109, 74, 255, .22);
    --shadow-lg: 0 22px 50px -14px rgba(109, 74, 255, .35);
    --ease: cubic-bezier(.22, 1, .36, 1);
}

/* ═════════ Base ═════════ */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: var(--text);
}

.stApp {
    background:
        radial-gradient(800px 480px at 5% -8%, rgba(109, 74, 255, .13), transparent 60%),
        radial-gradient(700px 460px at 100% 0%, rgba(14, 165, 233, .12), transparent 60%),
        radial-gradient(700px 500px at 60% 115%, rgba(236, 72, 153, .08), transparent 60%),
        var(--bg) !important;
    background-attachment: fixed !important;
    color: var(--text) !important;
}

/* Soft dotted pattern */
.stApp::before {
    content: '';
    position: fixed; inset: 0;
    background-image: radial-gradient(rgba(109, 74, 255, .16) 1px, transparent 1.2px);
    background-size: 26px 26px;
    mask-image: radial-gradient(ellipse at 50% 30%, black 10%, transparent 75%);
    -webkit-mask-image: radial-gradient(ellipse at 50% 30%, black 10%, transparent 75%);
    animation: dotsMove 30s linear infinite;
    pointer-events: none;
    z-index: 0;
}
@keyframes dotsMove { to { background-position: 26px 26px; } }

/* Floating pastel blobs */
.orb { position: fixed; border-radius: 50%; filter: blur(80px); pointer-events: none; z-index: 0; animation: floatOrb 20s ease-in-out infinite; }
.orb-1 { width: 340px; height: 340px; background: #c9bbff; opacity: .5; top: -90px; left: 12%; }
.orb-2 { width: 300px; height: 300px; background: #a5e3fb; opacity: .5; bottom: -70px; right: 8%; animation-delay: -7s; }
.orb-3 { width: 220px; height: 220px; background: #ffc4e1; opacity: .4; top: 48%; right: 32%; animation-delay: -13s; }
@keyframes floatOrb {
    0%, 100% { transform: translate(0, 0) scale(1); }
    33%      { transform: translate(40px, -30px) scale(1.12); }
    66%      { transform: translate(-30px, 40px) scale(.94); }
}

/* Hide default Streamlit chrome */
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent !important; }
.block-container { padding-top: 2rem !important; max-width: 1300px; position: relative; z-index: 1; }

/* ═════════ Sidebar ═════════ */
[data-testid="stSidebar"] {
    background: rgba(255, 255, 255, .85) !important;
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    border-right: 1px solid var(--border) !important;
    box-shadow: 6px 0 30px rgba(109, 74, 255, .05);
}
[data-testid="stSidebar"] * { color: var(--text); }

/* ═════════ Typography ═════════ */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    color: var(--text) !important;
}

.hero-title {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: clamp(2.2rem, 5vw, 3.5rem);
    font-weight: 800;
    line-height: 1.08;
    letter-spacing: -0.035em;
    margin: 0;
    background: linear-gradient(110deg, #3b2bb3 0%, #6d4aff 30%, #0ea5e9 55%, #ec4899 80%, #6d4aff 100%);
    background-size: 250% auto;
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shine 8s linear infinite, fadeDown .8s var(--ease) both;
}
@keyframes shine { to { background-position: 250% center; } }

.hero-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: .74rem;
    color: var(--text-muted);
    letter-spacing: .26em;
    text-transform: uppercase;
    margin-top: .7rem;
    animation: fadeDown .9s var(--ease) .1s both;
}
.hero-sub .dot {
    display: inline-block; width: 7px; height: 7px; border-radius: 50%;
    background: var(--success); margin-right: .6rem;
    box-shadow: 0 0 0 4px rgba(16, 185, 129, .18);
    animation: pulse 1.8s infinite;
}

.sb-logo { display: flex; align-items: center; gap: .8rem; margin-bottom: .2rem; }
.sb-logo-icon {
    width: 46px; height: 46px; border-radius: 14px;
    display: grid; place-items: center; font-size: 1.45rem;
    background: linear-gradient(135deg, #6d4aff, #0ea5e9);
    box-shadow: 0 10px 24px rgba(109, 74, 255, .35);
    animation: bob 3.5s ease-in-out infinite;
}
@keyframes bob { 0%,100% { transform: translateY(0) rotate(0); } 50% { transform: translateY(-5px) rotate(-4deg); } }
.sb-logo-text { font-family: 'Plus Jakarta Sans', sans-serif; font-weight: 800; font-size: 1.2rem; line-height: 1.1; }
.sb-logo-text small { display:block; font-family:'JetBrains Mono',monospace; font-weight:400; font-size:.6rem; letter-spacing:.2em; color: var(--text-muted); text-transform: uppercase; margin-top: 3px; }

/* ═════════ Cards ═════════ */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.5rem 1.6rem;
    margin-bottom: 1.1rem;
    position: relative;
    overflow: hidden;
    box-shadow: var(--shadow-sm);
    transition: transform .35s var(--ease), border-color .3s, box-shadow .35s;
    animation: riseIn .7s var(--ease) both;
}
.card:hover {
    transform: translateY(-5px);
    border-color: var(--border-hover);
    box-shadow: var(--shadow-lg);
}
/* Colored accent strip on the left */
.card::before {
    content: '';
    position: absolute; top: 0; left: 0; bottom: 0; width: 4px;
    background: linear-gradient(180deg, var(--accent), var(--accent-2));
    border-radius: 4px 0 0 4px;
}
/* Light sweep on hover */
.card::after {
    content: '';
    position: absolute; top: 0; left: -120%;
    width: 60%; height: 100%;
    background: linear-gradient(100deg, transparent, rgba(109, 74, 255, .06), transparent);
    transform: skewX(-20deg);
    transition: left .8s var(--ease);
    pointer-events: none;
}
.card:hover::after { left: 150%; }

.card-title {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: .74rem;
    font-weight: 700;
    letter-spacing: .14em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 1rem;
    display: flex; align-items: center; gap: .6rem;
}
.card-title .ico {
    width: 32px; height: 32px; border-radius: 10px;
    display: grid; place-items: center; font-size: 1rem;
    background: var(--accent-soft);
    transition: transform .4s var(--ease);
}
.card:hover .card-title .ico { transform: rotate(-8deg) scale(1.12); }
.card-content { font-size: .93rem; line-height: 1.8; color: var(--text); word-break: break-word; }
.card-content p { margin: 0 0 .6rem; }
.card-content ul { margin: 0; padding-left: 1.15rem; }
.card-content li { margin-bottom: .5rem; }
.card-content li::marker { color: var(--accent); }
.card-content strong { color: #000; font-weight: 700; }

/* Colour variants */
.card.v-green::before { background: linear-gradient(180deg, #10b981, #34d399); }
.card.v-green .ico    { background: var(--success-soft); }
.card.v-green li::marker { color: var(--success); }
.card.v-amber::before { background: linear-gradient(180deg, #f59e0b, #fbbf24); }
.card.v-amber .ico    { background: var(--warning-soft); }
.card.v-amber li::marker { color: var(--warning); }
.card.v-pink::before  { background: linear-gradient(180deg, #ec4899, #f9a8d4); }
.card.v-pink .ico     { background: var(--pink-soft); }
.card.v-pink li::marker { color: var(--accent-3); }

/* Staggered entrance */
.d1 { animation-delay: .05s; } .d2 { animation-delay: .15s; }
.d3 { animation-delay: .25s; } .d4 { animation-delay: .35s; }
@keyframes riseIn   { from { opacity: 0; transform: translateY(26px); } to { opacity: 1; transform: none; } }
@keyframes fadeDown { from { opacity: 0; transform: translateY(-14px); } to { opacity: 1; transform: none; } }

/* Title banner */
.title-banner {
    background: linear-gradient(120deg, #6d4aff 0%, #7c5cff 40%, #0ea5e9 100%);
    background-size: 200% 200%;
    animation: riseIn .7s var(--ease) both, gradShift 10s ease infinite;
    border-radius: 24px;
    padding: 1.9rem 2.1rem;
    margin-bottom: 1.3rem;
    position: relative; overflow: hidden;
    box-shadow: 0 20px 44px -14px rgba(109, 74, 255, .55);
}
@keyframes gradShift { 0%,100% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } }
.title-banner::after {
    content: '\\f008'; font-family: 'Font Awesome 6 Free'; font-weight: 900; position: absolute; right: 1.6rem; top: 50%;
    transform: translateY(-50%) rotate(-10deg); font-size: 5.5rem; opacity: .18;
}
.title-banner::before {
    content: ''; position: absolute; width: 220px; height: 220px; border-radius: 50%;
    background: rgba(255,255,255,.12); top: -90px; left: -60px;
}
.title-banner .tb-label { font-family:'JetBrains Mono',monospace; font-size:.68rem; letter-spacing:.24em; text-transform:uppercase; color: rgba(255,255,255,.8); margin-bottom:.6rem; position: relative; }
.title-banner .tb-title { font-family:'Plus Jakarta Sans',sans-serif; font-size: clamp(1.3rem, 3vw, 2rem); font-weight: 800; line-height: 1.25; color: #fff; position: relative; letter-spacing: -.01em; }

/* ═════════ Badges ═════════ */
.badge {
    display: inline-block;
    padding: .3rem .75rem;
    border-radius: 999px;
    font-family: 'JetBrains Mono', monospace;
    font-size: .64rem;
    font-weight: 500;
    letter-spacing: .12em;
    text-transform: uppercase;
    transition: transform .25s var(--ease);
}
.badge:hover { transform: translateY(-2px) scale(1.05); }
.badge-purple { background: var(--accent-soft);   color: var(--accent);   border: 1px solid #d8ccff; }
.badge-cyan   { background: var(--accent-2-soft); color: #0284c7;         border: 1px solid #b9e6fb; }
.badge-green  { background: var(--success-soft);  color: #059669;         border: 1px solid #b4efd6; }

/* ═════════ Inputs & Buttons ═════════ */
.stTextInput > div > div > input {
    background: #fff !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 14px !important;
    color: var(--text) !important;
    font-family: 'Inter', sans-serif !important;
    padding: .75rem .95rem !important;
    box-shadow: var(--shadow-sm) !important;
    transition: border-color .25s, box-shadow .25s !important;
}
.stTextInput > div > div > input::placeholder { color: #a3a2bf !important; }
.stTextInput > div > div > input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 4px rgba(109, 74, 255, .15) !important;
}
.stTextInput > div > div { background: transparent !important; border: none !important; }

.stSelectbox [data-baseweb="select"] > div {
    background: #fff !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 14px !important;
    color: var(--text) !important;
    box-shadow: var(--shadow-sm) !important;
}
.stSelectbox [data-baseweb="select"] * { color: var(--text) !important; }
[data-baseweb="popover"] ul, [data-baseweb="menu"] { background: #fff !important; border-radius: 14px !important; }
[data-baseweb="popover"] li { color: var(--text) !important; }
[data-baseweb="popover"] li:hover { background: var(--accent-soft) !important; }

.stButton > button {
    position: relative; overflow: hidden;
    background: linear-gradient(135deg, #6d4aff, #8b6cff 50%, #0ea5e9) !important;
    background-size: 200% 200% !important;
    color: #fff !important;
    border: none !important;
    border-radius: 14px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 700 !important;
    font-size: .88rem !important;
    letter-spacing: .04em !important;
    padding: .75rem 1.5rem !important;
    transition: transform .25s var(--ease), box-shadow .25s, background-position .6s !important;
    box-shadow: 0 10px 24px -6px rgba(109, 74, 255, .5) !important;
}
.stButton > button:hover {
    transform: translateY(-3px) !important;
    background-position: 100% 50% !important;
    box-shadow: 0 18px 34px -8px rgba(109, 74, 255, .6) !important;
}
.stButton > button:active { transform: translateY(0) scale(.97) !important; }
.stButton > button p { color: #fff !important; font-weight: 700 !important; }

.stButton > button[kind="secondary"] {
    background: #fff !important;
    border: 1.5px solid var(--border) !important;
    box-shadow: var(--shadow-sm) !important;
}
.stButton > button[kind="secondary"] p { color: var(--text-muted) !important; }
.stButton > button[kind="secondary"]:hover { border-color: var(--danger) !important; box-shadow: 0 10px 24px -8px rgba(239, 68, 68, .35) !important; }
.stButton > button[kind="secondary"]:hover p { color: var(--danger) !important; }

/* Expander */
[data-testid="stExpander"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    box-shadow: var(--shadow-sm);
    animation: riseIn .7s var(--ease) .15s both;
}
[data-testid="stExpander"] summary { font-family: 'Plus Jakarta Sans', sans-serif; font-weight: 700; color: var(--text) !important; }
[data-testid="stExpander"] summary:hover { color: var(--accent) !important; }

/* ═════════ Font Awesome icon styling ═════════ */
.sb-logo-icon i { color: #fff; font-size: 1.25rem; }
.card-title .ico i { color: var(--accent); font-size: .9rem; }
.card.v-green .ico i { color: var(--success); }
.card.v-amber .ico i { color: var(--warning); }
.card.v-pink  .ico i { color: var(--accent-3); }
.status-bar > span i { width: 1.2rem; text-align: center; margin-right: .2rem; color: var(--accent); }
.status-bar.s-done > span i { color: var(--success); }
.status-bar .check i { font-size: .6rem; color: #fff; }
.section-title i { color: var(--accent); font-size: 1.1rem; }
.tb-label i, .chat-label i { margin-right: .35rem; }
.chat-empty .ce-ico i { font-size: 2.3rem; color: var(--accent); }
.empty-ico i {
    font-size: 3.1rem;
    background: linear-gradient(135deg, #6d4aff, #0ea5e9);
    -webkit-background-clip: text; background-clip: text;
    -webkit-text-fill-color: transparent;
}
.feature .f-ico i { font-size: 1.25rem; color: var(--accent); }
.feature:nth-child(2) .f-ico i { color: #0284c7; }
.feature:nth-child(3) .f-ico i { color: var(--accent-3); }

/* Transcript (custom collapsible) */
.fa-details {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    box-shadow: var(--shadow-sm);
    padding: .2rem 1.1rem;
    margin-bottom: 1.1rem;
    animation: riseIn .7s var(--ease) .15s both;
    transition: border-color .3s, box-shadow .3s;
}
.fa-details:hover { border-color: var(--border-hover); box-shadow: var(--shadow-md); }
.fa-details summary {
    cursor: pointer; list-style: none; padding: .85rem 0;
    font-family: 'Plus Jakarta Sans', sans-serif; font-weight: 700; color: var(--text);
    display: flex; align-items: center; gap: .6rem;
}
.fa-details summary::-webkit-details-marker { display: none; }
.fa-details summary::after {
    content: '\\f078'; font-family: 'Font Awesome 6 Free'; font-weight: 900;
    margin-left: auto; font-size: .7rem; color: var(--text-muted);
    transition: transform .3s var(--ease);
}
.fa-details[open] summary::after { transform: rotate(180deg); }
.fa-details summary i { color: var(--accent); }
.fa-details .transcript-box { margin-bottom: 1rem; }

/* ═════════ Pipeline Steps ═════════ */
.status-bar {
    display: flex; align-items: center; gap: .8rem;
    padding: .72rem 1rem;
    background: #fff;
    border-radius: 14px;
    margin: .45rem 0;
    border: 1px solid var(--border);
    font-size: .83rem;
    box-shadow: var(--shadow-sm);
    transition: transform .25s var(--ease), border-color .25s;
    animation: slideIn .5s var(--ease) both;
}
.status-bar:hover { transform: translateX(4px); border-color: var(--border-hover); }
.status-bar.s-active { border-color: var(--accent); background: var(--accent-soft); }
.status-bar.s-done   { border-color: #b4efd6; background: #f4fdf9; }
.status-bar .check {
    margin-left: auto; width: 20px; height: 20px; border-radius: 50%;
    background: var(--success); color: #fff; font-size: .65rem;
    display: grid; place-items: center;
}
@keyframes slideIn { from { opacity: 0; transform: translateX(-16px); } to { opacity: 1; transform: none; } }

.status-dot { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }
.dot-active  { background: var(--accent); box-shadow: 0 0 0 4px rgba(109, 74, 255, .2); animation: pulse 1.2s infinite; }
.dot-done    { background: var(--success); }
.dot-pending { background: #d3d4e6; }
@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%      { opacity: .45; transform: scale(1.3); }
}

/* Progress */
.prog-wrap { height: 8px; border-radius: 99px; background: #e8e9f5; overflow: hidden; margin: .9rem 0 .4rem; }
.prog-fill {
    height: 100%; border-radius: 99px;
    background: linear-gradient(90deg, #6d4aff, #0ea5e9, #ec4899);
    background-size: 200% 100%;
    animation: flow 2s linear infinite;
    transition: width .5s var(--ease);
}
@keyframes flow { to { background-position: 200% 0; } }

/* Running banner */
.running {
    display: flex; align-items: center; gap: 1rem;
    padding: 1.15rem 1.4rem; margin-bottom: 1rem;
    border-radius: var(--radius);
    background: #fff;
    border: 1.5px solid #d8ccff;
    box-shadow: var(--shadow-md);
    animation: riseIn .5s var(--ease) both;
}
.loader {
    width: 28px; height: 28px; border-radius: 50%;
    border: 3.5px solid #e6e0ff;
    border-top-color: var(--accent);
    border-right-color: var(--accent-2);
    animation: spin .85s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }
.running b { font-family: 'Plus Jakarta Sans', sans-serif; color: var(--text); }
.running span { color: var(--text-muted); font-size: .82rem; }

/* ═════════ Chat ═════════ */
.chat-container {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.3rem;
    max-height: 460px;
    overflow-y: auto;
    margin-bottom: 1rem;
    display: flex; flex-direction: column; gap: 1rem;
    box-shadow: var(--shadow-sm);
}
.chat-msg { display: flex; flex-direction: column; gap: .3rem; animation: popIn .45s var(--ease) both; }
.chat-label { font-family: 'JetBrains Mono', monospace; font-size: .62rem; font-weight: 500; letter-spacing: .16em; text-transform: uppercase; }
.chat-bubble { padding: .8rem 1.1rem; font-size: .92rem; line-height: 1.7; max-width: 85%; word-break: break-word; }
.user-label { color: var(--accent); }
.bot-label  { color: #0284c7; }
.user-bubble {
    background: linear-gradient(135deg, #6d4aff, #8b6cff);
    color: #fff;
    border-radius: 18px 18px 4px 18px;
    box-shadow: 0 8px 20px -6px rgba(109, 74, 255, .5);
}
.bot-bubble {
    background: var(--surface-2);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 18px 18px 18px 4px;
}
@keyframes popIn { from { opacity: 0; transform: translateY(10px) scale(.96); } to { opacity: 1; transform: none; } }

.chat-empty { text-align: center; padding: 2.4rem 1rem; }
.chat-empty::before { display: none; }
.chat-empty .ce-ico { font-size: 2.5rem; display: inline-block; animation: bob 3s ease-in-out infinite; }
.chat-empty .ce-txt { color: var(--text-muted); font-size: .9rem; margin-top: .5rem; }

.section-title {
    font-family: 'Plus Jakarta Sans', sans-serif; font-size: 1.35rem; font-weight: 800;
    margin: .4rem 0 1rem; display: flex; align-items: center; gap: .6rem; color: var(--text);
}
.section-title::after { content: ''; flex: 1; height: 1px; background: linear-gradient(90deg, var(--border), transparent); }

/* ═════════ Transcript ═════════ */
.transcript-box {
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.2rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: .78rem;
    line-height: 1.9;
    max-height: 340px;
    overflow-y: auto;
    color: #4a4966;
    white-space: pre-wrap;
    word-break: break-word;
}

/* ═════════ Empty State ═════════ */
.empty {
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    padding: 3.5rem 2rem; text-align: center;
    animation: riseIn .9s var(--ease) both;
}
.empty-ico {
    width: 112px; height: 112px; border-radius: 32px;
    display: grid; place-items: center; font-size: 3.5rem; margin-bottom: 1.6rem;
    background: linear-gradient(135deg, #fff, #efeaff);
    border: 1px solid #d8ccff;
    box-shadow: 0 24px 50px -12px rgba(109, 74, 255, .4), inset 0 -6px 14px rgba(109, 74, 255, .08);
    animation: bob 4s ease-in-out infinite;
}
.empty h2 { font-size: 2rem; font-weight: 800; margin: 0 0 .6rem; }
.empty p  { color: var(--text-muted); font-size: .94rem; max-width: 450px; line-height: 1.8; }
.features { margin-top: 2.2rem; display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 1.1rem; width: 100%; max-width: 780px; }
.feature {
    background: #fff; border: 1px solid var(--border); border-radius: 18px;
    padding: 1.3rem; text-align: left; box-shadow: var(--shadow-sm);
    transition: transform .35s var(--ease), border-color .3s, box-shadow .3s;
}
.feature:hover { transform: translateY(-7px); border-color: var(--border-hover); box-shadow: var(--shadow-lg); }
.feature .f-ico {
    width: 46px; height: 46px; border-radius: 14px; display: grid; place-items: center;
    font-size: 1.4rem; margin-bottom: .8rem; background: var(--accent-soft);
}
.feature:nth-child(2) .f-ico { background: var(--accent-2-soft); }
.feature:nth-child(3) .f-ico { background: var(--pink-soft); }
.feature .f-t { font-family: 'Plus Jakarta Sans', sans-serif; font-weight: 700; font-size: .98rem; margin-bottom: .3rem; }
.feature .f-d { color: var(--text-muted); font-size: .8rem; line-height: 1.6; }

/* ═════════ Misc ═════════ */
hr { border: none !important; border-top: 1px solid var(--border) !important; margin: 1.5rem 0 !important; }
.stProgress > div > div > div { background: var(--accent) !important; }
.stSpinner > div { border-top-color: var(--accent) !important; }
[data-testid="stMarkdownContainer"] p { color: var(--text); }
label, [data-testid="stWidgetLabel"] p { color: var(--text-muted) !important; font-size: .8rem !important; font-weight: 500 !important; }
[data-testid="stAlert"] { border-radius: 14px !important; }

::-webkit-scrollbar { width: 7px; height: 7px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #cfc6ff; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent); }

@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after { animation: none !important; transition: none !important; }
}
</style>

<div class="orb orb-1"></div>
<div class="orb orb-2"></div>
<div class="orb orb-3"></div>
""",
    unsafe_allow_html=True,
)

# ─── Session State Init ──────────────────────────────────────────────────────────
for key, default in {
    "result": None,
    "chat_history": [],
    "processing": False,
    "pipeline_done": False,
    "pipeline_steps": {},
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ─── Helpers ────────────────────────────────────────────────────────────────────
STEPS = [
    ("audio",      "<i class='fa-solid fa-volume-high'></i>",        "Audio Processing"),
    ("transcript", "<i class='fa-solid fa-closed-captioning'></i>",  "Transcription"),
    ("title",      "<i class='fa-solid fa-tag'></i>",                "Title Generation"),
    ("summary",    "<i class='fa-solid fa-align-left'></i>",         "Summarisation"),
    ("extract",    "<i class='fa-solid fa-magnifying-glass'></i>",   "Extraction"),
    ("rag",        "<i class='fa-solid fa-brain'></i>",              "RAG Engine"),
]


def to_html(text) -> str:
    """Safely convert LLM text (plain / markdown-ish) to HTML so it renders
    nicely inside cards without breaking the layout."""
    if text is None:
        return ""
    if isinstance(text, (list, tuple)):
        text = "\n".join(f"- {t}" for t in text)
    text = html.escape(str(text).strip())
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)

    lines, out, in_list = text.split("\n"), [], False
    for line in lines:
        m = re.match(r"^\s*(?:[-*•]|\d+[.)])\s+(.*)", line)
        if m:
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{m.group(1)}</li>")
        else:
            if in_list:
                out.append("</ul>")
                in_list = False
            if line.strip():
                out.append(f"<p>{line}</p>")
    if in_list:
        out.append("</ul>")
    return "".join(out)


def render_steps(container, steps: dict):
    """Render the pipeline status list (works live, inside a placeholder)."""
    done = sum(1 for k, _, _ in STEPS if steps.get(k) == "done")
    pct = int(done / len(STEPS) * 100)
    html_out = f'<div class="prog-wrap"><div class="prog-fill" style="width:{pct}%"></div></div>'
    html_out += f'<div style="font-size:.7rem;color:var(--text-muted);margin-bottom:.4rem;font-family:JetBrains Mono,monospace">{pct}% complete</div>'
    for i, (key, icon, label) in enumerate(STEPS):
        s = steps.get(key, "pending")
        dot = {"active": "dot-active", "done": "dot-done"}.get(s, "dot-pending")
        cls = {"active": "s-active", "done": "s-done"}.get(s, "")
        check = '<span class="check"><i class="fa-solid fa-check"></i></span>' if s == "done" else ""
        html_out += (
            f'<div class="status-bar {cls}" style="animation-delay:{i*0.05}s">'
            f'<div class="status-dot {dot}"></div><span>{icon} {label}</span>{check}</div>'
        )
    container.markdown(html_out, unsafe_allow_html=True)


def card(icon: str, title: str, body: str, variant: str = "", delay: str = "d1") -> str:
    return f"""
    <div class="card {variant} {delay}">
        <div class="card-title"><span class="ico">{icon}</span>{title}</div>
        <div class="card-content">{to_html(body)}</div>
    </div>"""


# ─── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div class="sb-logo">
            <div class="sb-logo-icon"><i class="fa-solid fa-film"></i></div>
            <div class="sb-logo-text">AI Video Assistant<small>MeetingTube-AI</small></div>
        </div>""",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    st.markdown('<span class="badge badge-purple">Input</span>', unsafe_allow_html=True)
    source = st.text_input(
        "YouTube URL or File Path",
        placeholder="https://youtube.com/watch?v=... or /path/to/file.mp4",
    )

    language = st.selectbox("Language", ["english", "urdu"], index=0)

    run_btn = st.button("Analyse", icon=":material/bolt:", use_container_width=True)

    status_area = st.empty()
    if st.session_state.pipeline_steps:
        with status_area.container():
            st.markdown("---")
            st.markdown('<span class="badge badge-green">Pipeline Status</span>', unsafe_allow_html=True)
        steps_box = st.empty()
        render_steps(steps_box, st.session_state.pipeline_steps)
    else:
        steps_box = st.empty()

# ─── Main Area ──────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">AI Video Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub"><span class="dot"></span>Transcribe · Summarise · Chat with your meetings</div>',
    unsafe_allow_html=True,
)
st.markdown("---")

# ── Run Pipeline ────────────────────────────────────────────────────────────────
if run_btn:
    if not source.strip():
        st.error("Please enter a YouTube URL or file path.", icon=":material/error:")
    else:
        st.session_state.pipeline_done = False
        st.session_state.result = None
        st.session_state.chat_history = []
        st.session_state.pipeline_steps = {}

        progress_placeholder = st.empty()

        with status_area.container():
            st.markdown("---")
            st.markdown('<span class="badge badge-green">Pipeline Status</span>', unsafe_allow_html=True)

        def update_step(key, state):
            st.session_state.pipeline_steps[key] = state
            render_steps(steps_box, st.session_state.pipeline_steps)

        try:
            progress_placeholder.markdown(
                """<div class="running"><div class="loader"></div>
                <div><b>Pipeline running…</b><br><span>Check Live status sidebar</span></div></div>""",
                unsafe_allow_html=True,
            )

            update_step("audio", "active")
            chunks = process_input(source)
            update_step("audio", "done")

            update_step("transcript", "active")
            transcript = transcribe_all(chunks, language)
            update_step("transcript", "done")

            update_step("title", "active")
            title = generate_title(transcript)
            update_step("title", "done")

            update_step("summary", "active")
            summary = summarize(transcript)
            update_step("summary", "done")

            update_step("extract", "active")
            extracted = extract_all(transcript)
            action_items = extracted["action_items"]
            decisions = extracted["decisions"]
            questions = extracted["questions"]
            update_step("extract", "done")

            update_step("rag", "active")
            rag_chain = build_rag_chain(transcript)
            update_step("rag", "done")

            st.session_state.result = {
                "title": title,
                "transcript": transcript,
                "summary": summary,
                "action_items": action_items,
                "key_decisions": decisions,
                "open_questions": questions,
                "rag_chain": rag_chain,
            }
            st.session_state.pipeline_done = True
            progress_placeholder.success("Analysis complete!", icon=":material/check_circle:")
            time.sleep(0.6)
            progress_placeholder.empty()
            st.rerun()

        except Exception as e:
            for k, _, _ in STEPS:
                if st.session_state.pipeline_steps.get(k) == "active":
                    st.session_state.pipeline_steps[k] = "pending"
            render_steps(steps_box, st.session_state.pipeline_steps)
            progress_placeholder.error(f"Error: {e}", icon=":material/error:")

# ── Results ──────────────────────────────────────────────────────────────────────
if st.session_state.result:
    r = st.session_state.result

    st.markdown(
        f"""
    <div class="title-banner">
        <div class="tb-label"><i class="fa-solid fa-thumbtack"></i> Session Title</div>
        <div class="tb-title">{html.escape(str(r['title']).strip())}</div>
    </div>""",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([3, 2], gap="medium")

    with col1:
        st.markdown(card("<i class='fa-solid fa-align-left'></i>", "Summary", r["summary"], "", "d1"), unsafe_allow_html=True)

    with col2:
        transcript_html = html.escape(str(r["transcript"])).replace("\n", "<br>")
        st.markdown(
            '<details class="fa-details"><summary><i class="fa-solid fa-file-lines"></i> Full Transcript</summary>'
            f'<div class="transcript-box">{transcript_html}</div></details>',
            unsafe_allow_html=True,
        )

    c1, c2, c3 = st.columns(3, gap="medium")

    with c1:
        st.markdown(card("<i class='fa-solid fa-list-check'></i>", "Action Items", r["action_items"], "v-green", "d2"), unsafe_allow_html=True)
    with c2:
        st.markdown(card("<i class='fa-solid fa-key'></i>", "Key Decisions", r["key_decisions"], "v-amber", "d3"), unsafe_allow_html=True)
    with c3:
        st.markdown(card("<i class='fa-solid fa-circle-question'></i>", "Open Questions", r["open_questions"], "v-pink", "d4"), unsafe_allow_html=True)

    st.markdown("---")

    # ── RAG Chat ──────────────────────────────────────────────────────────────
    st.markdown('<div class="section-title"><i class="fa-solid fa-comments"></i> Chat with your Meeting</div>', unsafe_allow_html=True)

    if st.session_state.chat_history:
        chat_html = '<div class="chat-container">'
        for msg in st.session_state.chat_history:
            content = html.escape(str(msg["content"])).replace("\n", "<br>")
            if msg["role"] == "user":
                chat_html += f"""
                <div class="chat-msg" style="align-items:flex-end">
                    <span class="chat-label user-label">You</span>
                    <div class="chat-bubble user-bubble">{content}</div>
                </div>"""
            else:
                chat_html += f"""
                <div class="chat-msg" style="align-items:flex-start">
                    <span class="chat-label bot-label"><i class="fa-solid fa-robot"></i> Assistant</span>
                    <div class="chat-bubble bot-bubble">{content}</div>
                </div>"""
        chat_html += "</div>"
        st.markdown(chat_html, unsafe_allow_html=True)
    else:
        st.markdown(
            """
        <div class="card chat-empty">
            <div class="ce-ico"><i class="fa-solid fa-comments"></i></div>
            <div class="ce-txt">Ask anything about your meeting transcript</div>
        </div>""",
            unsafe_allow_html=True,
        )

    chat_col1, chat_col2 = st.columns([5, 1], gap="small")
    with chat_col1:
        user_input = st.text_input(
            "Your question",
            placeholder="What were the main decisions made?",
            label_visibility="collapsed",
        )
    with chat_col2:
        send_btn = st.button("Send", icon=":material/send:", use_container_width=True)

    if send_btn and user_input.strip():
        with st.spinner("Thinking…"):
            answer = ask_question(r["rag_chain"], user_input.strip())
        st.session_state.chat_history.append({"role": "user", "content": user_input.strip()})
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.rerun()

    if st.session_state.chat_history:
        if st.button("Clear Chat", icon=":material/delete:", type="secondary"):
            st.session_state.chat_history = []
            st.rerun()

else:
    st.markdown(
        """
    <div class="empty">
        <div class="empty-ico"><i class="fa-solid fa-film"></i></div>
        <h2>Ready to Analyse</h2>
        <p>Paste a YouTube URL or local file path in the sidebar, choose your language,
           and hit <strong>Analyse</strong> to get started.</p>
        <div class="features">
            <div class="feature"><div class="f-ico"><i class="fa-solid fa-closed-captioning"></i></div><div class="f-t">Transcription</div><div class="f-d">Accurately transcribe the entire audio of the video into English text</div></div>
            <div class="feature"><div class="f-ico"><i class="fa-solid fa-wand-magic-sparkles"></i></div><div class="f-t">Smart Summary</div><div class="f-d">Title, summary, action items aur decisions</div></div>
            <div class="feature"><div class="f-ico"><i class="fa-solid fa-brain"></i></div><div class="f-t">RAG Chat</div><div class="f-d">Ask questions about your meeting</div></div>
        </div>
    </div>""",
        unsafe_allow_html=True,
    )