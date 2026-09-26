# -*- coding: utf-8 -*-
"""
build_disease_database.py
农智通 · 病害标准图片数据库一键生成脚本
运行方式：双击运行，或在命令行执行 python build_disease_database.py
会在脚本所在目录生成 disease_database/ 和 disease_database_final.zip
只依赖 Python 标准库，无需安装任何第三方包。
"""

import json
import shutil
import zipfile
from pathlib import Path

# ============================================================
# 配置：22 种病害（5 种作物）
# ============================================================
DISEASES = {
    "谷子": ["谷瘟病", "谷子白发病", "谷子粒黑穗病", "谷子锈病", "谷子纹枯病", "谷子细菌性褐条病"],
    "高粱": ["高粱丝黑穗病", "高粱散黑穗病", "高粱坚黑穗病", "高粱炭疽病", "高粱北方炭疽病", "高粱靶斑病", "高粱纹枯病"],
    "荞麦": ["荞麦褐斑病", "荞麦霜霉病", "荞麦轮纹病"],
    "糜子": ["糜子黑穗病", "糜子白发病", "糜子叶斑病"],
    "燕麦": ["燕麦冠锈病", "燕麦秆锈病", "燕麦叶斑病"],
}

DB_DIR = Path("disease_database")
ZIP_PATH = Path("disease_database_final.zip")


# ============================================================
# metadata.json 模板（所有不确定信息一律写"待专项核验"）
# ============================================================
def make_metadata(crop: str, disease: str) -> dict:
    return {
        "crop": crop,
        "disease": disease,
        "aliases": [],
        "pathogen": {
            "category": "待专项核验",
            "scientific_name": "待专项核验",
            "verified": False,
            "source": "待专项核验"
        },
        "symptoms": {
            "whole_plant": "待专项核验",
            "leaf": "待专项核验",
            "stem": "待专项核验",
            "spike_or_panicle": "待专项核验",
            "grain_or_fruit": "待专项核验",
            "early_stage": "待专项核验",
            "late_stage": "待专项核验"
        },
        "diagnostic_features": [],
        "differential_diagnosis": [],
        "epidemiology": {
            "transmission": "待专项核验",
            "favorable_conditions": "待专项核验",
            "high_risk_growth_stage": "待专项核验",
            "overwintering_or_survival": "待专项核验"
        },
        "severity": {
            "level_1": "待专项核验",
            "level_2": "待专项核验",
            "level_3": "待专项核验"
        },
        "control": {
            "prevention": [],
            "agronomic": [],
            "biological": [],
            "chemical": [],
            "notes": "应依据当地植保部门指导及登记农药标签执行。"
        },
        "image_dataset": {
            "minimum_recommended_images": 30,
            "image_types": [
                "整株", "叶片", "病斑近照", "茎秆", "叶鞘", "穗部",
                "不同严重程度", "不同生育期", "不同角度", "不同光照", "不同背景"
            ],
            "verified_count": 0,
            "source_license": "待补充"
        },
        "verification": {
            "verified_by": "",
            "verified_date": "",
            "source_documents": [],
            "status": "reviewing"
        }
    }


# ============================================================
# README.txt 内容
# ============================================================
def build_readme(total_crops: int, total_diseases: int) -> str:
    lines = []
    lines.append("农智通 · 病害标准图片数据库（disease_database）")
    lines.append("=" * 48)
    lines.append("")
    lines.append("一、这是什么")
    lines.append("------------")
    lines.append("disease_database 是「农智通」程序的标准病害图片库框架。")
    lines.append("程序启动时会自动扫描本目录下的所有图片文件（.jpg/.jpeg/.png/.bmp/.webp），")
    lines.append("用于「标准病例图片检索」功能，为 AI 视觉诊断提供相似病例参考。")
    lines.append("")
    lines.append("二、目录结构（程序实际读取方式）")
    lines.append("---------------------------------")
    lines.append("disease_database/")
    lines.append("  <作物>/          第1级目录 = 作物名")
    lines.append("    <病害>/        第2级目录 = 病害名")
    lines.append("      001.jpg      第3级 = 标准病例图片（支持 jpg/jpeg/png/bmp/webp）")
    lines.append("      002.jpg")
    lines.append("      ...")
    lines.append("      metadata.json  病害知识扩展数据库（程序不直接读取，供人工维护/扩展）")
    lines.append("      .gitkeep        占位文件（保证空目录可被 git/压缩包保留）")
    lines.append("")
    lines.append("程序通过 rglob 递归扫描，按「第1级=作物、第2级=病害」归类图片。")
    lines.append("metadata.json 与 .gitkeep 不会被当作图片扫描。")
    lines.append("")
    lines.append(f"三、包含的作物与病害（共 {total_crops} 种作物 / {total_diseases} 种病害）")
    lines.append("-" * 46)
    for crop, dlist in DISEASES.items():
        lines.append("")
        lines.append(f"【{crop}】")
        for d in dlist:
            lines.append(f"  - {d}")
    lines.append("")
    lines.append("四、metadata.json 的作用")
    lines.append("------------------------")
    lines.append("每个病害目录下的 metadata.json 是该病害的结构化知识档案，")
    lines.append("包含：病原信息、症状（整株/叶/茎/穗/籽粒、早/晚期）、诊断特征、")
    lines.append("鉴别诊断、流行规律、严重程度分级、综合防治（预防/农业/生物/化学）、")
    lines.append("图片数据集要求、核验状态等。")
    lines.append("")
    lines.append('当前所有字段默认值均为「待专项核验」，status 为 "reviewing"。')
    lines.append("请由植保专业人员依据权威资料逐项核实后再改为 verified，")
    lines.append("严禁将推测内容当作已验证事实写入。")
    lines.append("")
    lines.append("五、标准图片放置位置与要求")
    lines.append("---------------------------")
    lines.append("把真实标准图片放入对应目录，例如：")
    lines.append("  disease_database/谷子/谷瘟病/001.jpg")
    lines.append("  disease_database/谷子/谷瘟病/002.jpg")
    lines.append("")
    lines.append("每种病害建议至少 30 张，尽量覆盖：")
    lines.append("  - 整株、叶片、病斑近照、茎秆、叶鞘、穗部")
    lines.append("  - 不同严重程度（轻/中/重）")
    lines.append("  - 不同生育期（苗期/拔节/抽穗/灌浆/成熟）")
    lines.append("  - 不同角度、不同光照、不同背景")
    lines.append("")
    lines.append("图片必须是真实病例或来源可靠且已确认授权的图片，")
    lines.append("严禁使用 AI 生成图或无关图片充当标准病例。")
    lines.append("")
    lines.append("六、如何接入农智通项目")
    lines.append("-----------------------")
    lines.append("将整个 disease_database 文件夹放到农智通 app.py 所在目录（BASE_DIR）下即可：")
    lines.append("")
    lines.append("  你的项目/")
    lines.append("    app.py")
    lines.append("    disease_database/")
    lines.append("      ...")
    lines.append("")
    lines.append("程序启动时会自动创建（若不存在）并扫描该目录，")
    lines.append("在「AI 拍照诊断」页面会显示图库图片数量与类别数。")
    lines.append("")
    lines.append("七、需要人工复核的资料")
    lines.append("-----------------------")
    lines.append("1. 各 metadata.json 中全部「待专项核验」字段；")
    lines.append("2. 病原菌学名、传播方式、越冬方式、流行条件；")
    lines.append("3. 所有化学防治药剂名称、剂量、施药次数（须与当地登记标签一致）；")
    lines.append("4. 每张标准图片的病例真实性与来源授权；")
    lines.append("5. 病害中文名称与地方俗称的对应关系（aliases 字段）。")
    lines.append("")
    lines.append("八、重要声明")
    lines.append("------------")
    lines.append("本数据库仅为图片库框架与知识档案模板，")
    lines.append("不构成任何植保诊断或用药建议。")
    lines.append("AI 诊断结果仅供参考，具体防治请咨询当地农技人员。")
    lines.append("")
    return "\n".join(lines)


# ============================================================
# 主流程
# ============================================================
def main():
    # 1. 清理旧目录
    if DB_DIR.exists():
        shutil.rmtree(DB_DIR)
        print("已删除旧的 disease_database/")
    DB_DIR.mkdir(parents=True)

    # 2. 创建目录结构 + metadata.json + .gitkeep
    index_entries = []
    for crop, dlist in DISEASES.items():
        crop_dir = DB_DIR / crop
        crop_dir.mkdir(parents=True, exist_ok=True)
        for d in dlist:
            ddir = crop_dir / d
            ddir.mkdir(parents=True, exist_ok=True)
            meta = make_metadata(crop, d)
            with open(ddir / "metadata.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            (ddir / ".gitkeep").touch()
            index_entries.append({
                "crop": crop,
                "disease": d,
                "path": f"{crop}/{d}/metadata.json"
            })
            print(f"  创建: {crop}/{d}/")

    total_crops = len(DISEASES)
    total_diseases = len(index_entries)

    # 3. knowledge_index.json
    with open(DB_DIR / "knowledge_index.json", "w", encoding="utf-8") as f:
        json.dump({
            "version": "1.0",
            "total_crops": total_crops,
            "total_diseases": total_diseases,
            "diseases": index_entries
        }, f, ensure_ascii=False, indent=2)
    print(f"\n已创建 knowledge_index.json（索引 {total_diseases} 种病害）")

    # 4. README.txt
    with open(DB_DIR / "README.txt", "w", encoding="utf-8") as f:
        f.write(build_readme(total_crops, total_diseases))
    print("已创建 README.txt")

    # 5. 打包 ZIP（内部根目录必须是 disease_database/）
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(DB_DIR.rglob("*")):
            if p.is_file():
                zf.write(p, arcname=str(p))  # p 本身已含 disease_database/ 前缀
    print(f"已打包: {ZIP_PATH} ({ZIP_PATH.stat().st_size / 1024:.1f} KB)")

    # 6. 自动检查
    print("\n" + "=" * 46)
    print("自动检查")
    print("=" * 46)

    errors = []

    # 检查1: 5 个作物目录
    crops_found = [d.name for d in DB_DIR.iterdir() if d.is_dir()]
    if len(crops_found) != 5:
        errors.append(f"作物目录数错误: {len(crops_found)}，应为 5")
    else:
        print(f"[通过] 5 个作物目录: {crops_found}")

    # 检查2: 22 个病害目录
    disease_dirs = [d for d in DB_DIR.rglob("*") if d.is_dir() and d.parent != DB_DIR]
    if len(disease_dirs) != total_diseases:
        errors.append(f"病害目录数错误: {len(disease_dirs)}，应为 {total_diseases}")
    else:
        print(f"[通过] {total_diseases} 个病害目录")

    # 检查3: 22 个 metadata.json
    metas = list(DB_DIR.rglob("metadata.json"))
    if len(metas) != total_diseases:
        errors.append(f"metadata.json 数量错误: {len(metas)}，应为 {total_diseases}")
    else:
        print(f"[通过] {total_diseases} 个 metadata.json")

    # 检查4: 全部可正常解析
    parse_fail = []
    for m in metas:
        try:
            data = json.loads(m.read_text(encoding="utf-8"))
            if not data.get("crop") or not data.get("disease"):
                parse_fail.append(str(m))
        except Exception:
            parse_fail.append(str(m))
    if parse_fail:
        errors.append(f"metadata.json 解析失败: {parse_fail}")
    else:
        print(f"[通过] {total_diseases} 个 metadata.json 全部可正常解析")

    # 检查5: knowledge_index.json 可解析且索引完整
    ki = json.loads((DB_DIR / "knowledge_index.json").read_text(encoding="utf-8"))
    if ki["total_diseases"] != total_diseases or len(ki["diseases"]) != total_diseases:
        errors.append("knowledge_index.json 索引数量不符")
    else:
        missing = [e for e in ki["diseases"] if not (DB_DIR / e["path"]).exists()]
        if missing:
            errors.append(f"knowledge_index.json 路径不存在: {missing}")
        else:
            print(f"[通过] knowledge_index.json 索引全部 {total_diseases} 种病害且路径存在")

    # 检查6: ZIP 可正常打开，且无嵌套错误
    with zipfile.ZipFile(ZIP_PATH) as zf:
        bad = zf.testzip()
        if bad:
            errors.append(f"ZIP 内文件损坏: {bad}")
        names = zf.namelist()
        if any(n.startswith("disease_database/disease_database/") for n in names):
            errors.append("ZIP 内存在错误嵌套 disease_database/disease_database/")
        if not any(n == "disease_database/README.txt" for n in names):
            errors.append("ZIP 内缺少 disease_database/README.txt")
        if not any(n == "disease_database/knowledge_index.json" for n in names):
            errors.append("ZIP 内缺少 disease_database/knowledge_index.json")
    if not errors:
        print(f"[通过] ZIP 正常打开，内部文件数: {len(names)}，无嵌套错误")

    # 汇总
    print("=" * 46)
    if errors:
        print("检查未通过：")
        for e in errors:
            print(f"  ❌ {e}")
    else:
        print(f"✅ 全部检查通过！")
        print(f"   作物: {total_crops} 种")
        print(f"   病害: {total_diseases} 种")
        print(f"   metadata.json: {total_diseases} 个")
        print(f"   输出: {DB_DIR.resolve()}")
        print(f"   输出: {ZIP_PATH.resolve()}")

    input("\n按回车键退出...")


if __name__ == "__main__":
    main()
