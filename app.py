# -*- coding: utf-8 -*-
"""
农智通 —— 山西杂粮 AI 植保诊断助手
主程序文件：app.py
技术栈：Streamlit + 阿里云百炼 Qwen-VL（兼容 OpenAI 协议）

运行方式：
    Windows PowerShell :  $env:DASHSCOPE_API_KEY="sk-你的Key"; streamlit run app.py
    macOS / Linux      :  export DASHSCOPE_API_KEY=sk-你的Key && streamlit run app.py
"""

import base64
import io
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

import requests
import streamlit as st
from PIL import Image, ImageOps, ImageEnhance

# ============================================================
# 0. 页面基础配置（必须是第一个 st 调用）
# ============================================================
st.set_page_config(
    page_title="农智通 - 山西杂粮AI植保诊断助手",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# 1. 常量与路径
# ============================================================
DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "").strip()
QWEN_VL_MODEL = os.environ.get("QWEN_VL_MODEL", "qwen-vl-max")
QWEN_TEXT_MODEL = os.environ.get("QWEN_TEXT_MODEL", "qwen-turbo")
DASHSCOPE_API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"

BASE_DIR = Path(__file__).resolve().parent
KB_DIR = BASE_DIR / "knowledge_base"
KB_FILE = KB_DIR / "杂粮病虫害知识库.json"
HISTORY_DIR = BASE_DIR / "history"
HISTORY_FILE = HISTORY_DIR / "diagnosis_history.json"

for _d in (KB_DIR, HISTORY_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Pillow 版本兼容
try:
    _LANCZOS = Image.Resampling.LANCZOS
except AttributeError:  # Pillow < 9.1
    _LANCZOS = Image.LANCZOS

# 农业病害图片的细小病斑非常依赖分辨率，不能像普通缩略图一样过度压缩。
# 当前百炼视觉模型文档显示，多数模型支持单图较高分辨率输入；这里采用较保守的 2048 长边。
MAX_IMAGE_SIDE = 2048
JPEG_QUALITY = 93
MAX_IMAGES = 3

# =========================
# 病害标准图片库
# 目录结构：
# disease_database/
#   谷子/
#     谷瘟病/
#       001.jpg
#       002.jpg
#     谷子白发病/
#       001.jpg
# =========================
DISEASE_IMAGE_DB_DIR = BASE_DIR / "disease_database"
IMAGE_SEARCH_TOP_K = 5
DISEASE_IMAGE_DB_DIR.mkdir(parents=True, exist_ok=True)

# 视觉诊断采用“逐图初判 + 多图综合”两阶段，避免多张图片互相污染判断。
PER_IMAGE_TIMEOUT = 120
SYNTHESIS_TIMEOUT = 120

# ============================================================
# 3.5 常见杂粮病虫害候选库
# 注意：这里是“视觉候选清单”，不是最终诊断结论。
# 它的作用是防止模型只在原有极小知识库中猜病害。
# ============================================================
COMMON_DISEASES = {
    "谷子": [
        "谷瘟病", "谷子白发病", "谷子黑穗病", "谷子纹枯病", "谷锈病",
        "谷子线虫病", "谷子红叶病", "粟灰螟", "粟叶甲", "粟芒蝇", "黏虫", "蚜虫"
    ],
    "高粱": [
        "高粱丝黑穗病", "高粱炭疽病", "高粱叶斑病", "高粱纹枯病", "高粱靶斑病",
        "高粱蚜", "玉米螟", "黏虫", "棉铃虫"
    ],
    "荞麦": [
        "荞麦霜霉病", "荞麦白粉病", "荞麦叶枯病", "荞麦轮纹病", "荞麦褐斑病",
        "荞麦立枯病", "荞麦根结线虫病"
    ],
    "糜子": [
        "糜子黑穗病", "糜子红叶病", "蚜虫", "黏虫", "螟虫"
    ],
    "燕麦": [
        "燕麦锈病", "燕麦黑穗病", "燕麦叶斑病", "蚜虫", "黏虫"
    ],
    "豆类": [
        "豆类锈病", "大豆灰斑病", "大豆紫斑病", "大豆霜霉病", "大豆根腐病",
        "大豆立枯病", "大豆菌核病", "大豆孢囊线虫病", "大豆蚜", "豆荚螟", "大豆食心虫"
    ],
}

# 视觉识别时给模型的少量“高区分度”症状提示。
# 不用于代替专业诊断，只用于帮助模型关注关键视觉特征。
VISUAL_SIGNATURES = {
    "谷瘟病": "叶片常见梭形/纺锤形病斑，中央灰白或灰褐，边缘深褐；穗部受害可出现死穗等表现。",
    "谷子白发病": "典型表现包括灰背、白尖、枪杆、白发、刺猬头等系统性症状；不是普通叶面白色粉层。",
    "谷子黑穗病": "穗部异常并出现黑色粉状物，正常籽粒结构被病组织替代。",
    "谷锈病": "叶片或叶鞘出现黄褐至锈褐色孢子堆，破裂后可见锈色粉末。",
    "高粱炭疽病": "叶片初见紫红色小斑点，后形成椭圆或长条病斑，边缘紫红、中央浅褐至灰白，可有黑色小点。",
    "高粱丝黑穗病": "穗部明显异常，病穗内部或表面形成黑色粉状孢子结构。",
    "大豆灰斑病": "叶片可出现圆形灰色斑点，后期边缘褐色、中部灰褐，部分表现近似蛙眼状。",
    "荞麦霜霉病": "叶片常出现黄化斑，湿度较高时叶片背面可出现霜状霉层。",
}

DEFAULT_RESULT = {
    "crop_name": "未知作物",
    "disease_name": "未能识别",
    "severity": "未知",
    "confidence": "低",
    "symptoms": "未能识别具体症状",
    "cause": "请结合专业知识判断",
    "treatment": "建议咨询当地农技人员",
    "prevention": "注意田间管理",
    "note": "AI 诊断结果仅供参考，具体用药请咨询当地农技人员。",
}

# ============================================================
# 2. 自定义样式（界面美化）
# ============================================================
CUSTOM_CSS = """
<style>
    .block-container { padding-top: 1.6rem; padding-bottom: 2.5rem; max-width: 1250px; }

    p.hero-title {
        text-align: center; font-size: 5em; font-weight: 800;
        color: #2e7d32; margin-bottom: 0.1em; letter-spacing: 2px;
    }
    .hero-sub { text-align: center; font-size: 1.35em; color: #555; margin-top: 0; }
    .hero-tag { text-align: center; font-size: 1.02em; color: #888; margin-top: 0.6em; }

    .feature-card {
        text-align: center; padding: 26px 16px; border-radius: 16px;
        border: 2px solid #e0e0e0; background: #ffffff;
        transition: all 0.25s ease; height: 100%;
    }
    .feature-card:hover { transform: translateY(-4px); box-shadow: 0 8px 20px rgba(0,0,0,0.08); }
    .feature-card h2 { margin: 0; font-size: 2.2em; }
    .feature-card h3 { margin: 0.4em 0; color: #333; }
    .feature-card p  { color: #777; font-size: 0.95em; line-height: 1.7; margin: 0; }

    .green  { border-color: #4caf50; }
    .blue   { border-color: #2196f3; }
    .orange { border-color: #ff9800; }
    .purple { border-color: #9c27b0; }

    .result-box {
        background: #f6fbf6; border-left: 5px solid #4caf50;
        border-radius: 10px; padding: 14px 18px; margin-bottom: 10px;
    }
    .result-box b { color: #2e7d32; }

    div[data-testid="stMetricValue"] { font-size: 1.5rem; color: #2e7d32; }
    .stButton > button { border-radius: 10px; }
</style>
"""

# ============================================================
# 3. 知识库
# ============================================================
DEFAULT_KB = [
    {
        "crop": "谷子", "disease": "谷子白发病",
        "symptoms": "系统性症状可表现为灰背、白尖、枪杆、白发、刺猬头等，病株生长异常，穗部和叶片可出现典型变色与病组织。",
        "cause": "由禾生指梗霜霉引起，是种传、土传的系统性侵染病害；具体发生与品种、播期、种源及田间条件有关。",
        "treatment": "以抗病品种、种子处理和及时拔除病株为主要措施；发现灰背、白尖等典型病株应尽快带出田外处理。",
        "prevention": "选用无病种子和抗病品种，适期播种，轮作倒茬，及时清除病株，降低田间菌源。",
    },
    {
        "crop": "高粱", "disease": "高粱炭疽病",
        "symptoms": "叶片上出现圆形或椭圆形褐色病斑，边缘紫红色，中央灰白色，严重时叶片枯死",
        "cause": "由炭疽菌引起，高温多雨季节易爆发，连作地块发病重",
        "treatment": "1.用50%多菌灵可湿性粉剂500倍液喷雾；2.或用75%百菌清可湿性粉剂600倍液",
        "prevention": "实行轮作；选用抗病品种；及时清除病残体并深翻",
    },
    {
        "crop": "高粱", "disease": "高粱叶斑病",
        "symptoms": "叶片上出现长方形或梭形褐色病斑，边缘深褐色，中间灰褐色，病斑沿叶脉扩展",
        "cause": "由真菌侵染引起，高温高湿条件有利发病",
        "treatment": "1.用50%代森锰锌可湿性粉剂500倍液喷雾；2.或用1:1:200波尔多液",
        "prevention": "合理密植改善通风条件；避免大水漫灌；收获后清除病残体",
    },
    {
        "crop": "荞麦", "disease": "荞麦霜霉病",
        "symptoms": "叶片正面出现黄色斑点，背面产生灰白色霜状霉层，严重时叶片枯黄脱落",
        "cause": "由霜霉菌引起，低温高湿环境易发病，连阴雨天气加重病情",
        "treatment": "1.用72%霜脲锰锌可湿性粉剂800倍液喷雾；2.或用64%杀毒矾可湿性粉剂500倍液",
        "prevention": "选择排水良好的地块；避免密植；发现病株及时拔除并销毁",
    },
    {
        "crop": "荞麦", "disease": "荞麦白粉病",
        "symptoms": "叶片和茎秆上出现白色粉状斑，逐渐扩大连成片，严重时整株被白粉覆盖",
        "cause": "由白粉菌引起，干旱与灌溉不当交替时易发病",
        "treatment": "1.用15%三唑酮可湿性粉剂1000倍液喷雾；2.或用40%多硫悬浮剂600倍液",
        "prevention": "合理施肥，避免氮肥过量；注意田间排水；保持植株通风透光",
    },
    {
        "crop": "糜子", "disease": "糜子黑穗病",
        "symptoms": "穗部被黑色粉状物取代，病穗比正常穗小，黑粉随风传播侵染健康植株",
        "cause": "由黑穗菌引起，种子带菌是主要传播途径",
        "treatment": "1.选用无病种子；2.用种子重量0.3%的50%福美双可湿性粉剂拌种",
        "prevention": "从无病田留种；轮作倒茬；及时拔除病株并带出田外销毁",
    },
    {
        "crop": "糜子", "disease": "糜子红叶病",
        "symptoms": "叶片由绿变红或紫红色，植株矮小，分蘖减少，抽穗困难",
        "cause": "多由蚜虫传播病毒引起，或土壤缺磷、低温胁迫导致",
        "treatment": "1.及时防治蚜虫（10%吡虫啉1500倍液）；2.叶面喷施0.2%磷酸二氢钾+尿素缓解",
        "prevention": "选用抗病品种；及时防蚜；增施有机肥，平衡施肥",
    },
    {
        "crop": "燕麦", "disease": "燕麦锈病",
        "symptoms": "叶片和茎秆上出现红褐色或深褐色疱状隆起，破裂后散出锈色粉末",
        "cause": "由锈菌引起，温暖潮湿环境易发病，可随气流远距离传播",
        "treatment": "1.用15%三唑酮可湿性粉剂1500倍液喷雾；2.或用20%萎锈灵乳油400倍液",
        "prevention": "选用抗病品种；合理施肥增强抗性；发病初期及时喷药控制",
    },
    {
        "crop": "豆类", "disease": "豆类锈病",
        "symptoms": "叶片两面出现褐色或黑色疱状隆起，破裂后散出褐色粉末，严重时叶片早枯",
        "cause": "由锈菌引起，温湿度适宜时传播迅速",
        "treatment": "1.用15%三唑酮可湿性粉剂1000倍液喷雾；2.或用25%敌力脱乳油2000倍液",
        "prevention": "选用抗病品种；合理密植通风透光；避免大水漫灌",
    },
    {
        "crop": "豆类", "disease": "豆类灰斑病",
        "symptoms": "叶片上出现圆形或近圆形灰色或灰褐色病斑，边缘深褐色，直径2-5毫米",
        "cause": "由真菌引起，低温高湿条件有利发病，风雨传播",
        "treatment": "1.用50%多菌灵可湿性粉剂800倍液喷雾；2.或用75%百菌清600倍液",
        "prevention": "轮作倒茬2-3年；选用无病种子；及时清除病残体",
    },
    {
        "crop": "谷子", "disease": "谷子粟灰螟",
        "symptoms": "苗期心叶枯死形成枯心苗，茎秆被蛀空，茎基部有虫粪排出",
        "cause": "粟灰螟幼虫钻蛀危害，高温干旱年份发生重",
        "treatment": "1.卵孵化盛期用2.5%溴氰菊酯乳油2000倍液喷雾；2.或用5%甲维盐3000倍液",
        "prevention": "秋季深翻灭茬；清除田间残株；灯光诱杀成虫",
    },
]


def init_knowledge_base():
    """知识库不存在则写入默认数据，返回当前知识库列表"""
    if not KB_FILE.exists():
        try:
            with open(KB_FILE, "w", encoding="utf-8") as f:
                json.dump(DEFAULT_KB, f, ensure_ascii=False, indent=2)
        except OSError as e:
            st.warning(f"知识库文件写入失败：{e}")
            return list(DEFAULT_KB)
    return load_knowledge_base()


def load_knowledge_base():
    """读取知识库 JSON"""
    if not KB_FILE.exists():
        return list(DEFAULT_KB)
    try:
        with open(KB_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else list(DEFAULT_KB)
    except (json.JSONDecodeError, OSError):
        return list(DEFAULT_KB)


def _bigrams(text: str):
    """生成 2-gram / 3-gram 分词，用于中文模糊匹配"""
    clean = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9]", "", text or "")
    tokens = set()
    for n in (2, 3):
        for i in range(len(clean) - n + 1):
            tokens.add(clean[i:i + n])
    return tokens


def search_knowledge_base(keyword: str, limit: int = 5):
    """在知识库中按权重搜索匹配条目"""
    if not keyword or not keyword.strip():
        return []
    kw = keyword.strip()
    kb = load_knowledge_base()
    tokens = _bigrams(kw)
    scored = []
    for item in kb:
        score = 0
        crop = item.get("crop", "")
        disease = item.get("disease", "")
        symptoms = item.get("symptoms", "")
        if kw in crop:
            score += 3
        if kw in disease:
            score += 6
        if kw in symptoms:
            score += 2
        for t in tokens:
            if t in disease:
                score += 2
            if t in crop:
                score += 1
            if t in symptoms:
                score += 1
        if score > 0:
            scored.append((score, item))
    scored.sort(key=lambda x: -x[0])
    return [it for _, it in scored[:limit]]


# ------------------------------------------------------------
# 知识库升级：纠正已知错误并补充常见病虫害名称。
# 不会覆盖用户自定义条目，只更新明确已知的问题项并补齐缺失项。
# ------------------------------------------------------------
SUPPLEMENTAL_KB = [
    {
        "crop": "谷子", "disease": "谷瘟病",
        "symptoms": "叶片常见梭形或纺锤形病斑，中央灰白或灰褐，边缘深褐；穗部受害可形成死穗。",
        "cause": "真菌性病害，发病与温湿条件、品种抗性、田间栽培条件等有关；仅凭单张图片不能确定具体病原来源。",
        "treatment": "优先结合病害发生期进行田间管理和登记药剂防控；疑似初发病斑时及时请当地植保人员确认。",
        "prevention": "选用抗病品种，合理施肥，加强田间监测，减少病残体和适宜发病条件。",
    },
    {
        "crop": "谷子", "disease": "谷子黑穗病",
        "symptoms": "穗部异常，正常籽粒结构被黑色粉状病组织替代。",
        "cause": "种传/侵染性病害，具体发生与种源和栽培条件有关。",
        "treatment": "及时拔除病株并带出田外规范处理，种子处理和抗病品种选择需依据当地植保建议。",
        "prevention": "使用健康种子，轮作，加强田间清洁。",
    },
    {
        "crop": "高粱", "disease": "高粱炭疽病",
        "symptoms": "叶片初为紫红色小斑点，后形成椭圆形或长条形病斑，边缘紫红、中央浅褐至灰白，可出现黑色小点。",
        "cause": "真菌性病害，温湿条件和田间病源等会影响发生。",
        "treatment": "根据病情和当地登记情况采取综合防治，不凭单张图片直接指定唯一药剂。",
        "prevention": "选抗病品种，合理轮作，加强田间通风和病残体管理。",
    },
    {
        "crop": "荞麦", "disease": "荞麦霜霉病",
        "symptoms": "叶片出现黄色或黄绿色斑块，湿润条件下叶片背面可见灰白色至霜状霉层。",
        "cause": "霜霉类病害，低温、高湿和连续阴雨等条件可能有利于发生。",
        "treatment": "及时清除重病株并改善通风排湿；药剂需依据当地登记和植保部门建议选择。",
        "prevention": "合理密植，保持排水和通风条件，避免长期高湿。",
    },
    {
        "crop": "豆类", "disease": "大豆灰斑病",
        "symptoms": "叶片出现圆形灰色斑点，病斑边缘可逐渐转褐，中部灰褐，部分病斑具有蛙眼状外观。",
        "cause": "真菌性叶部病害，发生程度受品种、温湿度和田间条件影响。",
        "treatment": "根据病情和当地登记药剂进行综合防治，注意不要仅凭叶斑颜色确定病名。",
        "prevention": "轮作、清除病残体、选择适宜品种并加强田间监测。",
    },
]


def upgrade_knowledge_base_file():
    """启动时迁移已知错误知识，并补齐关键常见病虫害条目。"""
    if not KB_FILE.exists():
        return
    try:
        with open(KB_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            return

        changed = False
        repaired = []
        for item in data:
            disease = item.get("disease", "") if isinstance(item, dict) else ""
            if disease == "谷子白发病":
                correct = next(x for x in DEFAULT_KB if x.get("disease") == "谷子白发病")
                for k in ("symptoms", "cause", "treatment", "prevention"):
                    if item.get(k) != correct.get(k):
                        item[k] = correct.get(k)
                        changed = True
            if disease == "谷子倒伏病":
                # 该条目并非稳定的标准病害分类，容易让视觉模型把“倒伏”当病名。
                changed = True
                continue
            repaired.append(item)

        existing = {(x.get("crop", ""), x.get("disease", "")) for x in repaired if isinstance(x, dict)}
        for extra in SUPPLEMENTAL_KB:
            key = (extra["crop"], extra["disease"])
            if key not in existing:
                repaired.append(extra)
                changed = True

        if changed:
            with open(KB_FILE, "w", encoding="utf-8") as f:
                json.dump(repaired, f, ensure_ascii=False, indent=2)
    except (json.JSONDecodeError, OSError, StopIteration):
        pass


# ============================================================
# 4. 图片处理（诊断优化区）
# ============================================================
def read_uploaded_bytes(uploaded_file) -> bytes:
    """安全读取 Streamlit 上传对象，读取前后都复位指针。"""
    try:
        uploaded_file.seek(0)
    except Exception:
        pass

    data = uploaded_file.read()

    try:
        uploaded_file.seek(0)
    except Exception:
        pass

    if not data:
        raise ValueError("文件内容为空")
    return data


def load_pil_image(data: bytes) -> Image.Image:
    """从字节流加载图片并立即解码，避免后续文件流失效。"""
    img = Image.open(io.BytesIO(data))
    img.load()
    return img


def to_jpeg_bytes(pil_img: Image.Image,
                  max_side: int = MAX_IMAGE_SIDE,
                  quality: int = JPEG_QUALITY) -> bytes:
    """
    为视觉模型做“尽量少损失病斑细节”的预处理：
      1. EXIF 方向纠正；
      2. 透明图铺白底；
      3. 最长边最多缩到 2048，而不是 1280；
      4. 轻度锐化，避免手机照片缩放后细节过软；
      5. JPEG 质量 93，避免过度压缩病斑边缘。
    """
    img = ImageOps.exif_transpose(pil_img)

    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        bg = Image.new("RGB", rgba.size, (255, 255, 255))
        bg.paste(rgba, mask=rgba.getchannel("A"))
        img = bg
    else:
        img = img.convert("RGB")

    w, h = img.size
    if w < 240 or h < 240:
        raise ValueError(f"图片分辨率过低：{w}×{h}，建议使用更清晰的原图")

    if max(w, h) > max_side:
        scale = max_side / float(max(w, h))
        img = img.resize(
            (max(1, int(w * scale)), max(1, int(h * scale))),
            _LANCZOS,
        )

    # 轻度锐化，不做大幅度对比度/饱和度增强，避免把病斑颜色人为改变。
    img = ImageEnhance.Sharpness(img).enhance(1.08)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True, progressive=True)
    return buf.getvalue()


def bytes_to_base64(data: bytes) -> str:
    return base64.b64encode(data).decode("utf-8")


def show_image(img, caption=None):
    """兼容新旧版本 Streamlit 的图片展示。"""
    try:
        st.image(img, caption=caption, use_container_width=True)
    except TypeError:
        st.image(img, caption=caption, use_column_width=True)


def image_basic_info(pil_img: Image.Image) -> dict:
    """返回不依赖 AI 的基础图片信息，用于质量提示。"""
    w, h = pil_img.size
    return {
        "width": w,
        "height": h,
        "megapixels": round((w * h) / 1_000_000, 2),
        "aspect_ratio": round(w / h, 3) if h else 0,
    }


# ============================================================
# 5. AI 调用
# ============================================================
def api_key_ready() -> bool:
    key = DASHSCOPE_API_KEY
    return bool(key) and key.lower() not in ("your-api-key-here", "sk-xxx", "none")


def _headers():
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
    }


def _friendly_http_error(status: int, body: str = "") -> str:
    if status in (401, 403):
        return "API Key 无效或权限不足，请检查环境变量 DASHSCOPE_API_KEY 是否配置正确。"
    if status == 400:
        return "请求格式有误。常见原因包括图片过大、图片格式不支持或模型参数不兼容。"
    if status == 404:
        return f"视觉模型不存在或未开通：{QWEN_VL_MODEL}。可在环境变量 QWEN_VL_MODEL 中改成你百炼账号已开通的视觉模型。"
    if status == 429:
        return "调用过于频繁或额度不足，请稍后重试。"
    if status >= 500:
        return f"AI 服务端异常（{status}），请稍后重试。"
    return f"AI 服务返回异常状态（{status}），请稍后重试。"


def _post_with_retry(payload, retries: int = 2, connect_timeout: int = 15,
                     read_timeout: int = 120):
    last_exc = None
    for attempt in range(retries + 1):
        try:
            resp = requests.post(
                DASHSCOPE_API_URL,
                json=payload,
                headers=_headers(),
                timeout=(connect_timeout, read_timeout),
            )
        except requests.exceptions.Timeout as e:
            last_exc = e
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
                continue
            raise
        except requests.exceptions.RequestException as e:
            last_exc = e
            if attempt < retries:
                time.sleep(1.2 * (attempt + 1))
                continue
            raise

        if resp.status_code in (429, 500, 502, 503, 504) and attempt < retries:
            time.sleep(1.5 * (attempt + 1))
            continue
        return resp

    raise RuntimeError(f"请求失败：{last_exc}")


def extract_json_from_text(content: str):
    """稳健提取模型返回的第一个完整 JSON 对象。"""
    if not content:
        return None

    text = content.strip().lstrip("\ufeff")

    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()

    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass

    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start:i + 1]
                try:
                    return json.loads(candidate)
                except (json.JSONDecodeError, TypeError):
                    return None
    return None


def _visual_signature_text() -> str:
    lines = []
    for name, desc in VISUAL_SIGNATURES.items():
        lines.append(f"- {name}：{desc}")
    return "\n".join(lines)


def _common_candidate_text(crop_hint: str = "自动识别") -> str:
    if crop_hint and crop_hint != "自动识别":
        names = COMMON_DISEASES.get(crop_hint, [])
    else:
        names = []
        for crop, values in COMMON_DISEASES.items():
            names.extend(values)
    return "、".join(names)


def _make_image_part(jpeg: bytes):
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{bytes_to_base64(jpeg)}"},
    }


VISUAL_ANALYSIS_PROMPT = (
    "你是农智通的农业视觉初筛专家。你的任务不是立刻拍脑袋给出病名，而是从图片中提取可验证的视觉证据。\n"
    "严格执行：\n"
    "1. 先判断作物类别；\n"
    "2. 只描述图片真正能看到的症状：病斑形状、颜色、边缘、霉层、孢子粉、虫体、虫孔、穗部异常、叶片卷曲/黄化、茎秆损伤等；\n"
    "3. 给出最多3个候选病虫害，必须写出每个候选为什么支持、什么证据不足；\n"
    "4. 不允许因为‘看起来像’就把猜测写成确定结论；\n"
    "5. 如果图片模糊、主体太远、病斑太小或光线异常，要明确标记 image_quality；\n"
    "6. 允许识别知识库之外的常见病虫害，不要强行套用原有知识库。\n"
    "7. 输出严格 JSON，不要 Markdown。\n"
    "JSON格式："
    '{"crop":"","image_quality":"良好/一般/较差","observations":[""],'
    '"candidate_diseases":[{"name":"","support":"","contradiction":"","confidence":"高/中/低"}],'
    '"missing_evidence":[""],"need_better_photo":false}'
)


FINAL_SYSTEM_PROMPT = (
    "你是‘农智通’农业病虫害综合诊断专家，服务对象主要是山西及华北地区杂粮种植者。\n"
    "你的核心原则：视觉证据优先、知识库辅助、证据不足时宁可保守，不要编造。\n\n"
    "【非常重要】\n"
    "1. 图片中的直接可见现象与病因不是一回事。症状可以直接描述；病原、传播途径、诱发条件只能写‘可能原因’，不能把模型猜测写成事实。\n"
    "2. 不得把知识库里的错误或不相关条目照抄为原因。只有作物、症状和候选病害相互吻合时才使用知识库信息。\n"
    "3. 不得强制从知识库中选一个病。知识库之外的常见病虫害也可以诊断。\n"
    "4. 对相似病害必须进行排除比较；至少给出一个主要鉴别依据。\n"
    "5. 如果图片证据不足，disease_name 输出‘暂不能确定’，confidence 输出‘低’，并明确告诉用户需要补拍什么部位。\n"
    "6. 防治建议必须与最终诊断一致；诊断不确定时，不要给出容易导致误用农药的确定性处方。\n"
    "7. 严禁凭空捏造图片中不存在的症状。\n"
    "8. 只输出 JSON，不要 Markdown。\n\n"
    "输出字段："
    '{"crop_name":"","disease_name":"","disease_type":"病害/虫害/非侵染因素/暂不能确定",'
    '"severity":"轻度/中度/重度/未知","confidence":"高/中/低",'
    '"symptoms":"只写图片可见症状","cause":"可能原因；与证据对应，不确定就明确不确定",'
    '"treatment":"防治建议","prevention":"预防措施",'
    '"diagnosis_basis":"为什么这样判断","alternative_diagnoses":[""],'
    '"distinguishing_points":"与最相似病害如何区分","need_more_images":[],"image_quality":"良好/一般/较差",'
    '"note":"AI辅助诊断，仅供参考；具体用药应以当地农技人员和农药标签为准。"}'
)



def _simple_image_feature(img: Image.Image):
    """
    轻量图片检索特征：
    - 低分辨率灰度结构
    - RGB平均颜色
    - HSV近似颜色统计
    这是“相似病例检索分数”，不是诊断概率。
    不依赖额外向量数据库，便于比赛项目直接部署。
    """
    import math

    img = ImageOps.exif_transpose(img).convert("RGB").resize((24, 24))
    px = list(img.getdata())

    # 灰度结构
    gray = []
    for r, g, b in px:
        gray.append((0.299*r + 0.587*g + 0.114*b) / 255.0)

    # 颜色均值/标准差
    means = []
    stds = []
    for channel in range(3):
        vals = [p[channel] / 255.0 for p in px]
        m = sum(vals) / len(vals)
        means.append(m)
        stds.append((sum((v-m)**2 for v in vals) / len(vals)) ** 0.5)

    # 灰度缩成 8x8
    small = ImageOps.grayscale(img).resize((8, 8))
    structure = [v / 255.0 for v in small.getdata()]

    feat = means + stds + structure
    norm = math.sqrt(sum(v*v for v in feat))
    return [v / norm for v in feat] if norm else feat


def _feature_similarity(a, b):
    if not a or not b or len(a) != len(b):
        return 0.0
    return sum(x*y for x, y in zip(a, b))


@st.cache_resource(show_spinner=False)
def build_disease_image_index():
    """自动扫描 disease_database/作物/病害/图片。"""
    records = []
    if not DISEASE_IMAGE_DB_DIR.exists():
        return records

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for p in DISEASE_IMAGE_DB_DIR.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in exts:
            continue

        rel = p.relative_to(DISEASE_IMAGE_DB_DIR)
        parts = rel.parts

        if len(parts) >= 3:
            crop, disease = parts[0], parts[1]
        elif len(parts) == 2:
            crop, disease = parts[0], parts[0]
        else:
            crop, disease = "未知作物", "未知病害"

        try:
            img = Image.open(p)
            feature = _simple_image_feature(img)
            records.append({
                "path": str(p),
                "crop": crop,
                "disease": disease,
                "feature": feature,
            })
        except Exception:
            continue

    return records


def search_similar_disease_images(query_img: Image.Image,
                                  top_k: int = IMAGE_SEARCH_TOP_K,
                                  crop_filter: str = "自动识别"):
    """
    返回最相似的标准病例图片。
    注意：相似度仅用于检索，不等于疾病发生概率。
    """
    records = build_disease_image_index()
    if not records:
        return []

    q = _simple_image_feature(query_img)
    scored = []

    for rec in records:
        if crop_filter and crop_filter != "自动识别" and rec["crop"] != crop_filter:
            continue

        score = _feature_similarity(q, rec["feature"])
        scored.append({
            "path": rec["path"],
            "crop": rec["crop"],
            "disease": rec["disease"],
            "similarity": score,
        })

    scored.sort(key=lambda x: x["similarity"], reverse=True)
    return scored[:top_k]


def aggregate_case_results(results):
    """按病害聚合 Top-K 案例，避免一张偶然相似图片决定结果。"""
    groups = {}
    for item in results:
        name = item["disease"]
        groups.setdefault(name, [])
        groups[name].append(item["similarity"])

    rows = []
    for disease, scores in groups.items():
        rows.append({
            "disease": disease,
            "count": len(scores),
            "max_similarity": max(scores),
            "mean_similarity": sum(scores) / len(scores),
        })

    rows.sort(
        key=lambda x: (x["count"], x["mean_similarity"], x["max_similarity"]),
        reverse=True
    )
    return rows


def build_image_case_context(case_results):
    if not case_results:
        return "【标准病例图片检索】当前图库没有找到可用相似病例。"

    lines = [
        "【标准病例图片检索】以下结果只表示视觉相似案例，不是最终诊断概率。"
    ]
    for i, item in enumerate(case_results, 1):
        lines.append(
            f"{i}. {item['crop']} / {item['disease']} / "
            f"视觉相似度={item['similarity']:.3f}"
        )

    groups = aggregate_case_results(case_results)
    if groups:
        lines.append("按病害聚合：")
        for g in groups[:5]:
            lines.append(
                f"- {g['disease']}：Top-K中{g['count']}张，"
                f"平均相似度={g['mean_similarity']:.3f}"
            )

    return "\n".join(lines)


def call_qwen_vl_once(jpeg: bytes, system_prompt: str, text_msg: str,
                      max_tokens: int = 1800):
    """单次视觉模型调用；单图调用用于降低多图互相干扰。"""
    payload = {
        "model": QWEN_VL_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [_make_image_part(jpeg), {"type": "text", "text": text_msg}]},
        ],
        "temperature": 0.15,
        "max_tokens": max_tokens,
    }

    resp = _post_with_retry(payload, retries=2, read_timeout=PER_IMAGE_TIMEOUT)
    if resp.status_code != 200:
        print(f"[Qwen-VL {resp.status_code}] {resp.text[:500]}")
        raise RuntimeError(_friendly_http_error(resp.status_code, resp.text))

    try:
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError) as e:
        raise RuntimeError(f"AI 返回内容无法解析：{e}")

    if isinstance(content, list):
        content = "".join(seg.get("text", "") for seg in content if isinstance(seg, dict))
    return content or ""


def retrieve_diagnosis_kb(preliminary: list, crop_hint: str = "自动识别", limit: int = 8):
    """根据视觉初筛结果检索知识库，只把高度相关的条目交给最终模型。"""
    kb = load_knowledge_base()
    candidates = []
    for item in preliminary:
        if not isinstance(item, dict):
            continue
        crop = str(item.get("crop", ""))
        if crop:
            candidates.extend(search_knowledge_base(crop, limit=3))
        for c in item.get("candidate_diseases", []) or []:
            if isinstance(c, dict):
                name = str(c.get("name", ""))
            else:
                name = str(c)
            if name:
                candidates.extend(search_knowledge_base(name, limit=3))

    if crop_hint and crop_hint != "自动识别":
        candidates.extend([it for it in kb if it.get("crop") == crop_hint][:4])

    unique = []
    seen = set()
    for item in candidates:
        key = (item.get("crop", ""), item.get("disease", ""))
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
        if len(unique) >= limit:
            break
    return unique


def build_kb_context(kb_results: list) -> str:
    if not kb_results:
        return "本地知识库未找到足够匹配项；不得因此强行认定某个病害。"
    parts = ["【本地知识库候选信息，仅作为辅助，不是最终结论】"]
    for item in kb_results:
        parts.append(
            f"- {item.get('crop', '')} / {item.get('disease', '')}："
            f"症状={item.get('symptoms', '')}；"
            f"可能原因={item.get('cause', '')}；"
            f"防治={item.get('treatment', '')}；"
            f"预防={item.get('prevention', '')}"
        )
    return "\n".join(parts)


def synthesize_diagnosis(jpeg_list: list, preliminary: list, kb_results: list,
                         user_question: str = "", image_case_context: str = "") -> dict:
    """第二阶段：让视觉模型看到原图 + 初筛证据 + 知识库候选后综合判断。"""
    content_parts = []
    for idx, jpeg in enumerate(jpeg_list, 1):
        content_parts.append({"type": "text", "text": f"【原始图片 {idx}】请把这张图片作为独立证据，不要与其他图片混淆。"})
        content_parts.append(_make_image_part(jpeg))

    preliminary_text = json.dumps(preliminary, ensure_ascii=False, indent=2)
    kb_text = build_kb_context(kb_results)
    common_text = _common_candidate_text()
    signature_text = _visual_signature_text()

    prompt = (
        "现在进行最终综合诊断。\n\n"
        f"【用户补充信息】\n{user_question or '无'}\n\n"
        f"【第一阶段逐图视觉初筛】\n{preliminary_text}\n\n"
        f"{kb_text}\n\n"
        f"{image_case_context}\n\n"
        "【常见病虫害候选清单】\n" + common_text + "\n\n"
        "【高区分度视觉特征参考】\n" + signature_text + "\n\n"
        "请注意：候选清单不是答案；必须根据图片实际证据判断。"
    )
    content_parts.append({"type": "text", "text": prompt})

    payload = {
        "model": QWEN_VL_MODEL,
        "messages": [
            {"role": "system", "content": FINAL_SYSTEM_PROMPT},
            {"role": "user", "content": content_parts},
        ],
        "temperature": 0.2,
        "max_tokens": 2200,
    }

    resp = _post_with_retry(payload, retries=2, read_timeout=SYNTHESIS_TIMEOUT)
    if resp.status_code != 200:
        print(f"[Qwen-VL synthesis {resp.status_code}] {resp.text[:500]}")
        raise RuntimeError(_friendly_http_error(resp.status_code, resp.text))

    try:
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError) as e:
        raise RuntimeError(f"综合诊断返回无法解析：{e}")

    if isinstance(content, list):
        content = "".join(seg.get("text", "") for seg in content if isinstance(seg, dict))

    parsed = extract_json_from_text(content)
    if isinstance(parsed, dict):
        for key, default in DEFAULT_RESULT.items():
            val = parsed.get(key)
            if val is None or (isinstance(val, str) and not val.strip()):
                parsed[key] = default
        parsed.setdefault("diagnosis_basis", "")
        parsed.setdefault("alternative_diagnoses", [])
        parsed.setdefault("distinguishing_points", "")
        parsed.setdefault("need_more_images", [])
        parsed.setdefault("image_quality", "一般")
        parsed["_error"] = False
        parsed["_preliminary"] = preliminary
        parsed["_kb_results"] = kb_results
        parsed["_raw"] = content
        return parsed

    raise RuntimeError("AI 综合诊断结果不是合法 JSON。")


def call_qwen_vl(jpeg_list, user_question: str = "", detail: str = "标准", image_case_context: str = ""):
    """
    两阶段视觉诊断：
      第一阶段：每张图片单独分析，避免多图互相污染；
      第二阶段：原图 + 逐图结果 + 本地知识库候选一起综合判断。
    """
    base = dict(DEFAULT_RESULT)

    if not api_key_ready():
        return {**base, "_error": True, "disease_name": "配置缺失",
                "symptoms": "未检测到有效的 DASHSCOPE_API_KEY，请先在环境变量中配置阿里云百炼 API Key。"}

    if not jpeg_list:
        return {**base, "_error": True, "disease_name": "图片缺失",
                "symptoms": "未检测到有效的图片，请重新上传作物照片。"}

    preliminary = []
    try:
        common_text = _common_candidate_text()
        signature_text = _visual_signature_text()
        for idx, jpeg in enumerate(jpeg_list, 1):
            detail_hint = "标准"
            if detail == "详细":
                detail_hint = "请更加仔细地区分相似病害，但仍不得在证据不足时猜测。"
            text_msg = (
                f"这是本次诊断的第 {idx} 张图片。请只分析这一张图片。\n"
                f"作物提示：{user_question or '自动识别'}\n"
                f"常见候选清单：{common_text}\n"
                f"视觉特征参考：{signature_text}\n"
                f"诊断模式：{detail_hint}"
            )
            raw = call_qwen_vl_once(jpeg, VISUAL_ANALYSIS_PROMPT, text_msg, max_tokens=1500)
            parsed = extract_json_from_text(raw)
            if not isinstance(parsed, dict):
                parsed = {
                    "crop": "",
                    "image_quality": "一般",
                    "observations": [],
                    "candidate_diseases": [],
                    "missing_evidence": ["该图片初筛结果无法结构化解析"],
                    "need_better_photo": True,
                    "raw": raw[:1000],
                }
            parsed["image_index"] = idx
            preliminary.append(parsed)
    except Exception as e:
        return {**base, "_error": True, "disease_name": "视觉初筛失败",
                "symptoms": str(e)}

    kb_results = retrieve_diagnosis_kb(preliminary, crop_hint=user_question, limit=8)

    try:
        return synthesize_diagnosis(jpeg_list, preliminary, kb_results, user_question=user_question, image_case_context=image_case_context)
    except Exception as e:
        # 第二阶段失败时，不把第一阶段完全丢掉，给出可排查结果。
        first = preliminary[0] if preliminary else {}
        candidates = first.get("candidate_diseases", []) if isinstance(first, dict) else []
        candidate_names = []
        for c in candidates:
            if isinstance(c, dict):
                candidate_names.append(c.get("name", ""))
            else:
                candidate_names.append(str(c))
        return {
            **base,
            "_error": True,
            "disease_name": "综合诊断失败",
            "symptoms": "；".join(first.get("observations", [])) if isinstance(first, dict) else str(e),
            "note": f"最终综合阶段失败：{e}。视觉初筛候选：{'、'.join([x for x in candidate_names if x]) or '暂无'}",
            "_preliminary": preliminary,
            "_kb_results": kb_results,
        }


def call_qwen_text(prompt: str) -> str:
    """调用通义千问文本模型（知识库问答）。"""
    if not api_key_ready():
        return "⚠️ 未检测到有效的 DASHSCOPE_API_KEY，请先配置阿里云百炼 API Key 后再使用。"
    if not prompt or not prompt.strip():
        return "⚠️ 未收到有效的问题描述，请补充你的种植问题。"

    system_prompt = (
        "你是「农智通」农业知识问答助手，专门解答山西杂粮种植相关的病虫害防治问题。"
        "优先引用用户问题中给出的作物和症状，不要把不存在的症状当成事实。"
        "回答请包含【病因分析】【防治建议】【预防措施】三个方面；"
        "涉及农药时必须提醒用户按当地登记标签和农技人员意见执行。"
    )

    payload = {
        "model": QWEN_TEXT_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt.strip()},
        ],
        "temperature": 0.4,
        "max_tokens": 1500,
    }

    try:
        resp = _post_with_retry(payload)
    except requests.exceptions.Timeout:
        return "⚠️ AI 服务响应超时，请稍后重试。"
    except requests.exceptions.RequestException as e:
        print(f"[Qwen-Text 网络异常] {e}")
        return "⚠️ 网络连接不稳定，请稍后重试。"
    except Exception as e:
        print(f"[Qwen-Text 未知异常] {e}")
        return "⚠️ 服务暂时不可用，请稍后重试。"

    if resp.status_code != 200:
        print(f"[Qwen-Text {resp.status_code}] {resp.text[:300]}")
        return "⚠️ " + _friendly_http_error(resp.status_code, resp.text)

    try:
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError):
        return "⚠️ AI 返回内容无法解析，请稍后重试。"

    if isinstance(content, list):
        content = "".join(seg.get("text", "") for seg in content if isinstance(seg, dict))

    return content or "⚠️ 未获取到有效回答，请换个说法再试一次。"


# ============================================================
# 6. 历史记录
# ============================================================
def load_history():
    if not HISTORY_FILE.exists():
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def save_history(record: dict):
    history = load_history()
    record["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    history.insert(0, record)
    history = history[:50]
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except OSError as e:
        print(f"[历史保存失败] {e}")


def write_history(history):
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except OSError as e:
        print(f"[历史写入失败] {e}")


# ============================================================
# 7. 页面：首页
# ============================================================
def render_home():
    st.markdown("""
        <div style="padding: 30px 20px 10px 20px;">
            <p class="hero-title">🌾 农智通</p>
            <p class="hero-sub">山西杂粮 AI 植保诊断助手</p>
            <p class="hero-tag">拍照识病害 · 智能出方案 · 零门槛使用</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""
            <div class="feature-card green">
                <h2>📷</h2><h3>拍照诊断</h3>
                <p>上传或拍摄作物照片<br>AI 分析病斑与虫害<br>生成诊断与防治建议</p>
            </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""
            <div class="feature-card blue">
                <h2>💬</h2><h3>知识问答</h3>
                <p>文字描述病情<br>智能检索知识库<br>生成防治建议</p>
            </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""
            <div class="feature-card orange">
                <h2>📚</h2><h3>知识库</h3>
                <p>山西杂粮专属<br>病虫害防治资料<br>随时查阅</p>
            </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown("""
            <div class="feature-card purple">
                <h2>📋</h2><h3>问诊历史</h3>
                <p>自动保存诊断记录<br>随时回看对比<br>跟踪病情变化</p>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 关于农智通")
    st.markdown("""
农智通是一款专为**山西杂粮作物**（谷子、糜子、高粱、荞麦、燕麦、豆类等 30 多种）量身定制的
AI 植保诊断 Web 应用。针对山西作为"小杂粮王国"却缺乏专属 AI 植保工具、山区农技人员覆盖不足的痛点，
本项目利用**阿里云百炼 Qwen-VL 视觉大模型**，结合本地构建的杂粮病虫害知识库，实现"拍照即诊断"的便捷服务。

系统采用纯 Web 架构，农户无需安装 App、无需注册登录，用手机拍照后即可获得病害名称与防治方案。
界面针对农村老年用户做了适老化设计（大字号、少层级、强对比），真正实现零门槛、秒开即用的智慧农业服务。
    """)

    st.markdown("---")
    st.caption("本系统为华北五省大学生计算机应用大赛参赛作品 | 技术栈：Streamlit + Qwen-VL + Python")


# ============================================================
# 8. 页面：AI 拍照诊断
# ============================================================
def _build_diagnosis_report(result: dict, question: str) -> str:
    lines = [
        "=" * 46,
        "        农智通 · AI 植保诊断报告",
        "=" * 46,
        f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "-" * 46,
        f"【作物】      {result.get('crop_name', '未知')}",
        f"【病害】      {result.get('disease_name', '未能识别')}",
        f"【严重程度】  {result.get('severity', '未知')}",
        f"【AI 置信度】 {result.get('confidence', '中')}",
        "-" * 46,
        "【症状描述】",
        result.get("symptoms", "无"),
        "",
        "【可能原因】",
        result.get("cause", "无"),
        "",
        "【防治方案】",
        result.get("treatment", "无"),
        "",
        "【预防措施】",
        result.get("prevention", "无"),
        "",
        "【判断依据】",
        result.get("diagnosis_basis", "无"),
        "",
        "【鉴别要点】",
        result.get("distinguishing_points", "无"),
        "",
        "【其他候选】",
        "、".join(result.get("alternative_diagnoses", []) or []) or "无",
        "",
        "-" * 46,
        f"补充说明：{question or '无'}",
        f"温馨提示：{result.get('note', '')}",
        "=" * 46,
        "本报告由 AI 生成，仅供参考；具体用药请咨询当地农技人员。",
    ]
    return "\n".join(lines)


def render_diagnosis():
    st.markdown("### 📷 AI 拍照诊断")
    st.caption("建议近距离拍清病斑/虫体；多张图片会逐张分析后再综合，避免互相干扰")
    db_records = build_disease_image_index()
    st.caption(
        f"标准病例图库：{len(db_records)} 张图片 / "
        f"{len(set((x['crop'], x['disease']) for x in db_records))} 个作物-病害类别"
        if db_records else
        "标准病例图库尚未添加图片：请按 disease_database/作物/病害/图片.jpg 建库。"
    )
    st.markdown("---")

    images = []  # [(name, PIL.Image)]

    tab_upload, tab_camera = st.tabs(["📁 本地上传", "📸 直接拍照"])

    with tab_upload:
        files = st.file_uploader(
            f"请选择作物病害照片（最多 {MAX_IMAGES} 张，支持 jpg / png / bmp / webp）",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            accept_multiple_files=True,
            key="diag_files",
            help="建议拍摄叶片正反面、茎秆基部以及整株远景，识别更准确",
        )
        if files:
            for f in files[:MAX_IMAGES]:
                try:
                    raw = read_uploaded_bytes(f)
                    images.append((f.name, load_pil_image(raw)))
                except Exception as e:
                    st.error(f"图片「{f.name}」读取失败：{e}")

    with tab_camera:
        cam = st.camera_input("用摄像头拍摄病害部位", key="diag_cam")
        if cam is not None:
            try:
                raw = read_uploaded_bytes(cam)
                images.append(("camera.jpg", load_pil_image(raw)))
            except Exception as e:
                st.error(f"拍照图片读取失败：{e}")

    if not images:
        st.info("👆 请先上传或拍摄一张作物病害照片")
        return

    st.markdown("---")
    left, right = st.columns([1.1, 1])

    with left:
        st.subheader("图片预览")
        cols = st.columns(min(len(images), 3))
        for idx, (name, img) in enumerate(images):
            with cols[idx % len(cols)]:
                show_image(img, caption=name)

    with right:
        st.subheader("诊断设置")
        crop_select = st.selectbox(
            "作物类型（选填，可提高识别准确率）",
            ["自动识别", "谷子", "糜子", "高粱", "荞麦", "燕麦", "豆类", "其他"],
            index=0,
        )
        extra_desc = st.text_area(
            "补充描述（选填）",
            placeholder="例如：这片地种的是谷子，叶子从底部开始发黄，最近连下了三天雨……",
            height=90,
        )
        detail_level = st.radio("诊断详细程度", ["标准", "详细"], horizontal=True)

    st.markdown("")
    do_diagnose = st.button("🔍 开始诊断", type="primary", use_container_width=True)

    if do_diagnose:
        with st.spinner("正在压缩并编码图片……"):
            jpeg_list = []
            failed = []
            for name, img in images:
                try:
                    jpeg_list.append(to_jpeg_bytes(img))
                except Exception as e:
                    failed.append(f"{name}: {e}")

        if failed:
            st.warning("以下图片处理失败，已跳过：\n" + "\n".join(failed))

        if not jpeg_list:
            st.error("没有可用的图片，请重新上传。")
            return

        total_kb = sum(len(b) for b in jpeg_list) / 1024
        st.caption(f"已处理 {len(jpeg_list)} 张图片，合计约 {total_kb:.0f} KB")

        # 新增：标准病例图库检索
        case_results = []
        for _, img in images:
            case_results.extend(
                search_similar_disease_images(
                    img,
                    top_k=IMAGE_SEARCH_TOP_K,
                    crop_filter=crop_select,
                )
            )

        # 多张用户图片合并后，保留全局最相似的 Top-K
        case_results.sort(key=lambda x: x["similarity"], reverse=True)
        case_results = case_results[:IMAGE_SEARCH_TOP_K]
        image_case_context = build_image_case_context(case_results)

        with st.expander("🖼️ 标准病例图片检索", expanded=False):
            if case_results:
                for i, item in enumerate(case_results, 1):
                    st.write(
                        f"{i}. **{item['crop']} / {item['disease']}** "
                        f"（视觉相似度 {item['similarity']:.3f}）"
                    )
            else:
                st.info(
                    "当前没有可用的相似标准病例。请建立 "
                    "`disease_database/作物/病害/图片.jpg` 图片库。"
                )

        progress = st.progress(0)
        status = st.empty()

        # 构造提问
        user_question = ""
        if crop_select != "自动识别":
            user_question += f"这张图片中的作物是{crop_select}。"
        if extra_desc.strip():
            user_question += f"补充信息：{extra_desc.strip()}"

        status.text("第 1 步：逐张分析图片，提取病斑和虫害视觉证据……")
        progress.progress(0.25)
        result = call_qwen_vl(
            jpeg_list,
            user_question,
            detail=detail_level,
            image_case_context=image_case_context,
        )

        status.text("第 2 步：综合比较候选病虫害与本地知识库……")
        progress.progress(0.85)
        status.text("第 3 步：生成最终诊断与鉴别依据……")
        progress.progress(1.0)
        progress.empty()
        status.empty()

        st.session_state["diag_result"] = result
        st.session_state["diag_question"] = user_question
        st.session_state["diag_case_results"] = case_results

        if not result.get("_error"):
            save_history({
                "type": "photo_diagnosis",
                "crop": result.get("crop_name", "未知"),
                "disease": result.get("disease_name", "未知"),
                "severity": result.get("severity", "未知"),
                "symptoms": result.get("symptoms", ""),
                "treatment": result.get("treatment", ""),
                "prevention": result.get("prevention", ""),
                "images": [n for n, _ in images],
            })

    # ---------- 结果展示（从 session_state 读取，重跑不丢） ----------
    result = st.session_state.get("diag_result")
    if not result:
        return

    st.markdown("---")

    if result.get("_error"):
        st.error("❌ 诊断未能完成")
        st.markdown(f"**原因：** {result.get('symptoms', '未知错误')}")
        st.info("排查建议：\n"
                "1. 确认环境变量 `DASHSCOPE_API_KEY` 已正确配置且账户有余额；\n"
                "2. 确认百炼控制台已开通环境变量 `QWEN_VL_MODEL` 对应的视觉模型；\n"
                "3. 换一张光线充足、主体清晰的照片重试。")
        return

    st.success("✅ 诊断完成！")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.success(f"**作物：** {result.get('crop_name', '未知')}")
    with col_b:
        st.error(f"**病害：** {result.get('disease_name', '未能识别')}")
    with col_c:
        st.warning(f"**AI 置信度：** {result.get('confidence', '中')} ｜ **严重度：** {result.get('severity', '未知')}")

    st.markdown("")
    case_results = st.session_state.get("diag_case_results", [])
    if case_results:
        st.markdown("#### 🖼️ 相似标准病例")
        groups = aggregate_case_results(case_results)
        if groups:
            cols = st.columns(min(len(groups), 3))
            for idx, g in enumerate(groups[:3]):
                with cols[idx]:
                    st.metric(
                        g["disease"],
                        f"{g['mean_similarity']:.3f}",
                        f"{g['count']} 张相似案例"
                    )
        st.caption(
            "以上为图库视觉相似度，仅用于辅助诊断，不代表该病害的真实发生概率。"
        )

    st.markdown("#### 🔬 症状描述")
    st.write(result.get("symptoms", "未能识别具体症状"))

    st.markdown("#### 🧐 可能原因")
    st.write(result.get("cause", "请结合专业知识判断"))

    st.markdown("#### 💊 防治方案")
    st.warning(result.get("treatment", "建议咨询当地农技人员"))

    st.markdown("#### 🛡️ 预防措施")
    st.info(result.get("prevention", "注意田间管理"))

    st.markdown("#### 🔎 判断依据")
    st.write(result.get("diagnosis_basis", "未提供"))

    st.markdown("#### ⚖️ 鉴别与其他可能")
    st.write(result.get("distinguishing_points", "未提供"))
    alternatives = result.get("alternative_diagnoses", []) or []
    if alternatives:
        st.caption("其他候选：" + "、".join(alternatives))

    more_images = result.get("need_more_images", []) or []
    if more_images:
        st.warning("建议补拍：" + "；".join(str(x) for x in more_images))

    st.markdown("---")
    st.caption(result.get("note", "AI 诊断结果仅供参考，具体用药请咨询当地农技人员。"))

    # ---------- 报告下载 ----------
    report_text = _build_diagnosis_report(
        result, st.session_state.get("diag_question", "")
    )
    fname = f"农智通诊断报告_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    dcol1, dcol2 = st.columns([1, 3])
    with dcol1:
        st.download_button(
            "📥 下载诊断报告",
            data=report_text.encode("utf-8"),
            file_name=fname,
            mime="text/plain",
            use_container_width=True,
        )
    with dcol2:
        with st.expander("📄 预览报告全文"):
            st.text(report_text)

    # ---------- 知识库交叉验证 ----------
    disease_keyword = result.get("disease_name", "")
    if disease_keyword and disease_keyword not in ("未能识别", "识别结果（未结构化）"):
        kb_results = search_knowledge_base(disease_keyword, limit=2)
        if kb_results:
            st.markdown("---")
            st.markdown("### 📚 知识库辅助匹配（非最终判定）")
            for item in kb_results:
                with st.expander(f"{item.get('crop', '')} - {item.get('disease', '')}", expanded=False):
                    st.write(f"**症状：** {item.get('symptoms', '')}")
                    st.write(f"**防治：** {item.get('treatment', '')}")
                    st.write(f"**预防：** {item.get('prevention', '')}")


# ============================================================
# 9. 页面：知识问答
# ============================================================
QUICK_QUESTIONS = [
    "谷子叶子发黄是怎么回事？",
    "高粱穗子上有白色粉末是什么病？",
    "荞麦叶片上有褐色斑点怎么治？",
    "豆类叶子卷曲怎么办？",
    "糜子倒伏了是什么原因？",
    "燕麦叶片上有锈色粉末怎么办？",
]


def _answer_qa(question: str, crop: str = "不限"):
    kb_results = search_knowledge_base(question, limit=3)

    parts = []
    if crop and crop != "不限":
        parts.append(f"用户种植的作物是{crop}。")

    if kb_results:
        kb_context = "【本地知识库匹配到的相关信息】\n"
        for item in kb_results:
            kb_context += (
                f"- {item.get('crop', '')} {item.get('disease', '')}："
                f"症状={item.get('symptoms', '')}；"
                f"防治={item.get('treatment', '')}；"
                f"预防={item.get('prevention', '')}\n"
            )
        parts.append(kb_context)

    parts.append(f"用户问题：{question}")
    answer = call_qwen_text("\n".join(parts))
    return answer, kb_results


def render_qa():
    st.markdown("### 💬 农业知识智能问答")
    st.caption("用文字描述你的种植问题，AI 结合本地知识库生成防治建议")
    st.markdown("---")

    if "qa_messages" not in st.session_state:
        st.session_state.qa_messages = []

    with st.expander("❓ 常见问题（点击直接提问）", expanded=not st.session_state.qa_messages):
        cols = st.columns(3)
        clicked = None
        for i, q in enumerate(QUICK_QUESTIONS):
            with cols[i % 3]:
                if st.button(q, key=f"quick_q_{i}", use_container_width=True):
                    clicked = q

    if clicked:
        st.session_state.qa_messages.append({"role": "user", "content": clicked})
        with st.spinner("正在检索知识库并生成回答……"):
            answer, kb_results = _answer_qa(clicked)
        st.session_state.qa_messages.append({"role": "assistant", "content": answer})
        save_history({
            "type": "qa",
            "question": clicked,
            "answer": answer[:200],
            "crop": "不限",
        })

    # 历史对话
    for msg in st.session_state.qa_messages:
        with st.chat_message(msg["role"], avatar="🧑‍🌾" if msg["role"] == "user" else "🌾"):
            st.markdown(msg["content"])

    prompt = st.chat_input("请描述您遇到的种植问题，例如：谷子叶片从底部发黄，有褐色斑点……")

    if prompt:
        st.session_state.qa_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑‍🌾"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar="🌾"):
            placeholder = st.empty()
            with st.spinner("正在检索知识库并生成回答……"):
                answer, kb_results = _answer_qa(prompt)
            placeholder.markdown(answer)

        st.session_state.qa_messages.append({"role": "assistant", "content": answer})

        if kb_results:
            st.info(f"已匹配本地知识库 {len(kb_results)} 条相关条目")
        else:
            st.warning("未匹配到知识库条目，回答基于 AI 通用知识生成，建议咨询当地农技人员。")

        save_history({
            "type": "qa",
            "question": prompt,
            "answer": answer[:200],
            "crop": "不限",
        })

    if st.session_state.qa_messages:
        st.markdown("---")
        if st.button("🗑️ 清空当前对话", type="secondary"):
            st.session_state.qa_messages = []
            st.rerun()


# ============================================================
# 10. 页面：知识库浏览
# ============================================================
def render_knowledge():
    st.markdown("### 📚 杂粮病虫害知识库")
    st.caption("本地整理的山西杂粮常见病虫害防治资料")
    st.markdown("---")

    kb = load_knowledge_base()
    if not kb:
        st.info("知识库为空。")
        return

    crops = sorted({item.get("crop", "其他") for item in kb})
    col1, col2 = st.columns([1, 2])
    with col1:
        crop_filter = st.selectbox("按作物筛选", ["全部"] + crops)
    with col2:
        kw = st.text_input("关键词搜索", placeholder="输入病害名称或症状关键词，如：白粉、锈病")

    results = kb
    if crop_filter != "全部":
        results = [it for it in results if it.get("crop") == crop_filter]
    if kw.strip():
        hit = search_knowledge_base(kw.strip(), limit=50)
        hit_set = {id(it) for it in hit}
        results = [it for it in results if id(it) in hit_set]

    st.markdown(f"共 **{len(results)}** 条记录")
    st.markdown("---")

    for item in results:
        with st.expander(f"🌱 {item.get('crop', '')} — {item.get('disease', '')}"):
            st.markdown(f"**症状表现：** {item.get('symptoms', '暂无')}")
            st.markdown(f"**发病原因：** {item.get('cause', '暂无')}")
            st.markdown(f"**防治方案：** {item.get('treatment', '暂无')}")
            st.markdown(f"**预防措施：** {item.get('prevention', '暂无')}")

    st.markdown("---")
    st.download_button(
        "📥 导出知识库 JSON",
        data=json.dumps(kb, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name="农智通_杂粮病虫害知识库.json",
        mime="application/json",
    )


# ============================================================
# 11. 页面：问诊历史
# ============================================================
def render_history():
    st.markdown("### 📋 问诊历史记录")
    st.caption("查看所有历史诊断和问答记录")
    st.markdown("---")

    history = load_history()
    if not history:
        st.info("暂无问诊记录，去「AI 拍照诊断」试试吧～")
        return

    total = len(history)
    photo_cnt = sum(1 for r in history if r.get("type") == "photo_diagnosis")
    qa_cnt = total - photo_cnt

    m1, m2, m3 = st.columns(3)
    m1.metric("总记录数", total)
    m2.metric("拍照诊断", photo_cnt)
    m3.metric("知识问答", qa_cnt)

    st.markdown("---")

    keyword = st.text_input("🔎 搜索历史记录", placeholder="按作物、病害或问题关键词搜索")
    if keyword.strip():
        k = keyword.strip()
        history = [
            r for r in history
            if k in str(r.get("crop", "")) or k in str(r.get("disease", ""))
            or k in str(r.get("question", "")) or k in str(r.get("symptoms", ""))
        ]

    for i, record in enumerate(history):
        rtype = "📷 拍照诊断" if record.get("type") == "photo_diagnosis" else "💬 知识问答"
        title = f"#{i + 1} ｜ {record.get('timestamp', '未知时间')} ｜ {rtype}"
        with st.expander(title):
            if record.get("type") == "photo_diagnosis":
                st.markdown(f"**作物：** {record.get('crop', '未知')}")
                st.markdown(f"**病害：** {record.get('disease', '未知')}")
                st.markdown(f"**严重度：** {record.get('severity', '未知')}")
                st.markdown(f"**症状：** {record.get('symptoms', '暂无')}")
                st.markdown(f"**防治方案：** {record.get('treatment', '暂无')}")
                if record.get("prevention"):
                    st.markdown(f"**预防措施：** {record.get('prevention')}")
                if record.get("images"):
                    st.caption("图片：" + "、".join(record["images"]))
            else:
                st.markdown(f"**作物：** {record.get('crop', '不限')}")
                st.markdown(f"**问题：** {record.get('question', '暂无')}")
                st.markdown(f"**回答：** {record.get('answer', '暂无')}")

            if st.button("🗑️ 删除本条", key=f"del_{i}_{record.get('timestamp', i)}"):
                full = load_history()
                target_ts = record.get("timestamp")
                full = [r for r in full if r.get("timestamp") != target_ts]
                write_history(full)
                st.success("已删除")
                st.rerun()

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "📥 导出全部历史（JSON）",
            data=json.dumps(load_history(), ensure_ascii=False, indent=2).encode("utf-8"),
            file_name=f"农智通_问诊历史_{datetime.now().strftime('%Y%m%d')}.json",
            mime="application/json",
            use_container_width=True,
        )
    with c2:
        if st.button("🧹 清空所有记录", type="secondary", use_container_width=True):
            write_history([])
            st.success("已清空所有记录")
            st.rerun()


# ============================================================
# 12. 主程序
# ============================================================
def main():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    kb = init_knowledge_base()
    upgrade_knowledge_base_file()
    kb = load_knowledge_base()

    # ---------------- 侧边栏 ----------------
    with st.sidebar:
        st.markdown("## 🌾 农智通")
        st.caption("山西杂粮 AI 植保诊断助手")
        st.markdown("---")

        page = st.radio(
            "功能导航",
            [
                "🏠 首页",
                "📷 AI拍照诊断",
                "💬 知识问答",
                "📚 知识库",
                "📋 问诊历史",
            ],
            label_visibility="collapsed",
        )

        st.markdown("---")

        if not api_key_ready():
            st.error(
                "⚠️ 未检测到 API Key\n\n"
                "AI 功能不可用，请先设置环境变量：\n\n"
                "`DASHSCOPE_API_KEY`"
            )
            with st.expander("如何配置？"):
                st.code(
                    '# Windows PowerShell\n'
                    '$env:DASHSCOPE_API_KEY="sk-你的Key"\n'
                    'streamlit run app.py\n\n'
                    '# macOS / Linux\n'
                    'export DASHSCOPE_API_KEY=sk-你的Key\n'
                    'streamlit run app.py',
                    language="bash",
                )
        else:
            st.success("✅ API Key 已配置")
            st.info("**温馨提示**\n\nAI 诊断结果仅供参考，具体用药请咨询当地农技人员。")

        st.markdown("---")
        st.metric("知识库收录", f"{len(kb)} 条")
        st.metric("历史记录", f"{len(load_history())} 条")

        st.markdown("---")
        st.caption("© 农智通 · 华北五省大学生计算机应用大赛作品")

    # ---------------- 路由 ----------------
    if page == "🏠 首页":
        render_home()
    elif page == "📷 AI拍照诊断":
        render_diagnosis()
    elif page == "💬 知识问答":
        render_qa()
    elif page == "📚 知识库":
        render_knowledge()
    elif page == "📋 问诊历史":
        render_history()


if __name__ == "__main__":
    main()

