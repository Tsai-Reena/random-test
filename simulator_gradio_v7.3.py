# simulator_gradio.py
# -*- coding: utf-8 -*-

import json
import io
import os
from contextlib import redirect_stdout
from typing import Dict, List, Tuple, Any
import tempfile
import datetime

import gradio as gr
import pandas as pd

import simulator


# =============================
# 不寫死 fallback 的預設讀取/合併/驗證
# =============================
SYSTEM_DEFAULTS_PATH = os.getenv("SYSTEM_DEFAULTS_PATH", "system_defaults.json")
USER_DEFAULTS_PATH   = os.getenv("USER_DEFAULTS_PATH", "user_defaults.json")

REQUIRED_KEYS = [
    "inventory","draw_price","exchanges","max_to_color","stop_any",
    "stop_immediately","per_round_draws","num_runs","seed","auto_seed",
]

def _load_json_required(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"[ERROR] 系統預設檔不存在：{path}")
    with open(path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except Exception as e:
            raise ValueError(f"[ERROR] 系統預設檔格式錯誤：{path} - {e}")

def _load_json_optional(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _validate_defaults(d: dict):
    missing = [k for k in REQUIRED_KEYS if k not in d]
    if missing:
        raise KeyError(f"[ERROR] 預設缺少必要欄位：{missing}\n請補齊 {SYSTEM_DEFAULTS_PATH}")

def _coerce_types(d: dict) -> dict:
    out = dict(d)
    si = out.get("stop_immediately")
    if isinstance(si, bool):
        out["stop_immediately"] = 1 if si else 0
    elif isinstance(si, str):
        out["stop_immediately"] = 1 if si.strip().lower() in ("1","true","y","yes","是") else 0
    else:
        out["stop_immediately"] = int(si)
    as_ = out.get("auto_seed")
    if isinstance(as_, str):
        out["auto_seed"] = as_.strip().lower() in ("1","true","y","yes","是")
    else:
        out["auto_seed"] = bool(as_)
    if out.get("max_to_color", None) is None:
        out["max_to_color"] = None
    else:
        out["max_to_color"] = str(out["max_to_color"])
    return out

def _merge_defaults() -> dict:
    system = _load_json_required(SYSTEM_DEFAULTS_PATH)
    _validate_defaults(system)
    system = _coerce_types(system)
    user = _load_json_optional(USER_DEFAULTS_PATH) or {}
    merged = {**system, **user} if user else system
    merged = _coerce_types(merged)
    _validate_defaults(merged)
    return merged

def defaults_df():
    return _merge_defaults()["inventory"]

def defaults_other():
    d = _merge_defaults()
    return (
        int(d["draw_price"]),
        str(d["exchanges"]),
        (None if d["max_to_color"] in (None, "", "無限制") else str(d["max_to_color"])),
        list(d["stop_any"]),
        int(d["stop_immediately"]),
        int(d["per_round_draws"]),
        int(d["num_runs"]),
        int(d["seed"]),
        bool(d["auto_seed"]),
    )


# =============================
# 工具：解析輸入與表格處理
# =============================
def _to_rows(table: Any) -> List[List[Any]]:
    if isinstance(table, pd.DataFrame):
        return table.values.tolist()
    return table or []

def _row_has_content(r: list) -> bool:
    if not r: return False
    color = (str(r[0]).strip() if len(r) > 0 and r[0] is not None else "")
    qty_raw = r[1] if len(r) > 1 else None
    qty_has = bool(qty_raw is not None and str(qty_raw).strip() != "")
    return bool(color) or qty_has

def _as_int(x, default=0):
    try:
        return int(x)
    except Exception:
        return default

def _drop_empty_rows(rows: List[List[Any]]) -> List[List[Any]]:
    cleaned = []
    for r in rows:
        if not _row_has_content(r):
            continue
        color = (r[0] if len(r) > 0 else "") or ""
        qty = _as_int(r[1] if len(r) > 1 else 0, 0)
        price = _as_int(r[2] if len(r) > 2 else 0, 0)
        del_flag = bool(r[3]) if len(r) > 3 else False
        cleaned.append([color.strip(), qty, price, del_flag])
    return cleaned

def df_to_inventory(table) -> Dict[str, int]:
    rows = _drop_empty_rows(_to_rows(table))
    inv = {}
    for r in rows:
        color = r[0]; qty = _as_int(r[1], 0)
        if color and qty >= 0: inv[color] = qty
    return inv

def df_to_prices(table) -> Dict[str, int]:
    rows = _drop_empty_rows(_to_rows(table))
    prices = {}
    for r in rows:
        color = r[0]; price = _as_int(r[2], 0)
        if color: prices[color] = price
    return prices

def apply_delete(table):
    rows = _to_rows(table); kept = []
    for r in rows:
        del_flag = bool(r[3]) if len(r) >= 4 else False
        if not del_flag:
            color = r[0] if len(r) > 0 else ""
            qty = r[1] if len(r) > 1 else 0
            price = r[2] if len(r) > 2 else 0
            kept.append([color, qty, price, False])
    return _drop_empty_rows(kept)

def delete_last_row(table):
    rows = _drop_empty_rows(_to_rows(table))
    for i in range(len(rows)-1, -1, -1):
        if _row_has_content(rows[i]): rows.pop(i); break
    return rows

def colors_from_table(table) -> List[str]:
    rows = _drop_empty_rows(_to_rows(table))
    seen, out = set(), []
    for r in rows:
        c = (r[0] or "").strip()
        if c and c not in seen:
            seen.add(c); out.append(c)
    return out

def update_stop_any_options(table, current_values):
    opts = colors_from_table(table)
    if not isinstance(current_values, list): current_values = []
    new_values = [v for v in current_values if v in opts]
    return gr.update(choices=opts, value=new_values)

def update_max_to_options(table, current_value):
    colors = colors_from_table(table)
    choices = ["無限制"] + colors
    value = current_value if current_value in choices else "無限制"
    return gr.update(choices=choices, value=value)

def build_stop_all_dict(stop_all_rows: Any) -> Dict[str, int]:
    rows = _to_rows(stop_all_rows); res = {}
    for r in (rows or []):
        if not r or len(r) < 2: continue
        color = (str(r[0]).strip() if r[0] is not None else "")
        try: qty = int(r[1])
        except Exception: qty = 0
        if color and qty >= 1: res[color] = qty
    return res

def sync_stop_all_choices(table, current_selected, current_table):
    colors = colors_from_table(table)
    selected = [c for c in (current_selected or []) if c in colors]
    prev_rows = _to_rows(current_table); prev = {}
    for r in (prev_rows or []):
        if not r or len(r) < 2: continue
        c = (str(r[0]).strip() if r[0] is not None else "")
        try: q = int(r[1])
        except Exception: q = 1
        if c: prev[c] = q
    new_rows = [[c, prev.get(c, 1)] for c in selected]
    return (gr.update(choices=colors, value=selected), new_rows)

# =============================
# 兌換規則：文字/表格 互通工具
# =============================
def parse_exchanges(s: str) -> List[Tuple[str, int, str]]:
    s = (s or "").strip()
    if not s: return []
    try:
        obj = json.loads(s)
        if isinstance(obj, list):
            out = []
            for triple in obj:
                if not (isinstance(triple,(list,tuple)) and len(triple)==3):
                    raise ValueError("無效的 JSON 兌換規則。")
                out.append((str(triple[0]), int(triple[1]), str(triple[2])))
            return out
    except Exception:
        pass
    out = []
    for rule in s.split(";"):
        rule = rule.strip()
        if not rule: continue
        if "->" not in rule or "*" not in rule:
            raise ValueError(f'無效的兌換規則: "{rule}" (需要 "A*2->B")')
        left, to_c = rule.split("->", 1)
        from_c, ratio = left.split("*", 1)
        out.append((from_c.strip(), int(ratio.strip()), to_c.strip()))
    return out

def exchanges_from_ui(ex_mode, ex_text, ex_df):
    if ex_mode == "表格輸入":
        rows = _to_rows(ex_df) or []; out = []
        for r in rows:
            if not r: continue
            a = (str(r[0]).strip() if len(r)>0 and r[0] is not None else "")
            b = r[1] if len(r)>1 else None
            c = (str(r[2]).strip() if len(r)>2 and r[2] is not None else "")
            if not a or not c: continue
            try: b = int(b)
            except Exception: b = None
            if b and b>0: out.append((a,b,c))
        return out
    else:
        return parse_exchanges(ex_text or "")

def exchanges_text_from_df(ex_df):
    rows = _to_rows(ex_df) or []; parts = []
    for r in rows:
        if not r: continue
        a = (str(r[0]).strip() if len(r)>0 and r[0] is not None else "")
        b = r[1] if len(r)>1 else None
        c = (str(r[2]).strip() if len(r)>2 and r[2] is not None else "")
        try: b = int(b)
        except Exception: b = None
        if a and b and b>0 and c:
            parts.append(f"{a}*{b}->{c}")
    return "; ".join(parts)

def exchanges_df_from_text(ex_text):
    triples = parse_exchanges(ex_text or "")
    return [[a,b,c] for (a,b,c) in triples]

# =============================
# 執行模擬
# =============================
def run_simulation(
    base_inventory_table, draw_price,
    exchanges_mode, exchanges_text, exchanges_table,
    max_to_color,
    stop_any_list, stop_all_table, stop_immediately_choice,
    per_round_draws, num_runs, seed, auto_seed
):
    try:
        base_inventory = df_to_inventory(base_inventory_table)
        prices = df_to_prices(base_inventory_table)
        if not base_inventory:
            return "[ERROR] 基礎庫存不能為空。", gr.update()
        if not prices:
            return "[ERROR] 價格不能為空。", gr.update()
        for c in base_inventory:
            if c not in prices:
                return f'[ERROR] 缺少 "{c}" 的價格設定。', gr.update()

        exchanges = exchanges_from_ui(exchanges_mode, exchanges_text, exchanges_table)
        stop_any = list(stop_any_list or [])
        stop_all = build_stop_all_dict(stop_all_table)

        try:
            stop_immediately = bool(int(stop_immediately_choice))
        except Exception:
            stop_immediately = str(stop_immediately_choice).startswith("是")

        draw_price = int(draw_price)
        per_round_draws = int(per_round_draws)
        num_runs = int(num_runs)

        if auto_seed:
            seed = int.from_bytes(os.urandom(4), "little")
            seed_update = gr.update(value=seed, interactive=False)
        else:
            seed = int(seed)
            seed_update = gr.update(value=seed, interactive=True)

        max_to = None if max_to_color == "無限制" else (max_to_color or None)

        buf = io.StringIO()
        with redirect_stdout(buf):
            simulator.simulate_many(
                num_runs=num_runs,
                base_inventory=base_inventory,
                prices=prices,
                draw_price=draw_price,
                seed=seed,
                exchanges=exchanges,
                max_exchange_to_color=max_to,
                stop_immediately=stop_immediately,
                per_round_draws=per_round_draws,
                stop_on_any=stop_any,
                stop_on_all=stop_all,
            )
        return buf.getvalue(), seed_update
    except Exception as e:
        return f"[ERROR] {e}", gr.update()

def clear_output():
    return ""

def export_current_settings(
    inv_df, draw_price,
    exchanges_mode, exchanges_text, exchanges_table,
    max_to_color,
    stop_any, stop_all_table, stop_immediately,
    per_round_draws, num_runs, seed, auto_seed
):
    try:
        inventory_rows = _drop_empty_rows(_to_rows(inv_df))
        max_to_val = None if max_to_color == "無限制" else (max_to_color or None)
        stop_all_dict = build_stop_all_dict(stop_all_table)
        try:
            stop_immediately_val = int(stop_immediately)
        except Exception:
            stop_immediately_val = 1 if str(stop_immediately).startswith("是") else 0

        if exchanges_mode == "表格輸入":
            exchanges_str = exchanges_text_from_df(exchanges_table)
        else:
            exchanges_str = str(exchanges_text or "").strip()

        settings = {
            "inventory": inventory_rows,
            "draw_price": int(draw_price),
            "exchanges": exchanges_str,
            "max_to_color": max_to_val,
            "stop_any": list(stop_any or []),
            "stop_immediately": stop_immediately_val,
            "per_round_draws": int(per_round_draws),
            "num_runs": int(num_runs),
            "seed": int(seed),
            "auto_seed": bool(auto_seed),
            "stop_all": stop_all_dict,
        }
        tmp_dir = tempfile.gettempdir()
        out_path = os.path.join(tmp_dir, "user_defaults.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(settings, f, ensure_ascii=False, indent=2)
        return out_path
    except Exception:
        return None

def export_report_txt(report_text: str):
    """
    讀取右側輸出框內容，寫成 UTF-8 TXT 後回傳檔案路徑，給 gr.File 下載。
    """
    try:
        text = (report_text or "").strip()
        if not text:
            return None  # 沒內容就不產生檔案

        today_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        tmp_dir = tempfile.gettempdir()
        out_path = os.path.join(tmp_dir, f"simulation_report_{today_str}.txt")

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        return out_path
    except Exception:
        return None

def export_settings_and_report(
    inv_df, draw_price,
    exchanges_mode, exchanges_text, exchanges_table,
    max_to_color,
    stop_any, stop_all_table, stop_immediately,
    per_round_draws, num_runs, seed, auto_seed,
    report_text
):
    """
    一鍵匯出：同時產生設定 JSON 與報告 TXT
    回傳：(json_path, txt_path)
    """
    json_path = export_current_settings(
        inv_df, draw_price,
        exchanges_mode, exchanges_text, exchanges_table,
        max_to_color,
        stop_any, stop_all_table, stop_immediately,
        per_round_draws, num_runs, seed, auto_seed
    )
    txt_path = export_report_txt(report_text)
    return json_path, txt_path

# =============================
# 版面配置（置頂操作列 + 右側 sticky）
# =============================
with gr.Blocks(title="一番賞模擬器", css="""
/* ---------- 文字區字型 ---------- */
#outputbox textarea {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 13px;
}

/* ---------- 隱藏 Dataframe 內建 New row ---------- */
#inv_group button[aria-label="New row"],
#inv_group button[aria-label="Add row"],
#inv_group button[title="New row"],
#inv_group button[title="Add row"] { display: none !important; }

/* ---------- 主表卡片 ---------- */
#inv_group .gr-group, #inv_group .gr-panel, #inv_group .gr-box {
  background: transparent !important; box-shadow: none !important; border: none !important;
}
#inv_card { background: #fff; border: 1px solid #e5e7eb; border-radius: 10px; overflow: hidden; }
#inv_card .wrap, #inv_card table { background: #fff; }
#inv_card thead th { background: #f9fafb; }

/* ---------- 主表底部自訂按鈕列（仍保留但我們沒用） ---------- */
#df_actions { display:flex; justify-content:flex-end; gap:8px; background:#fff; border-top:1px solid #e5e7eb; padding:10px; }
.df-mini .gr-button { height:36px; padding:0 14px; font-size:14px; line-height:1; background:#f3f4f6; border:1px solid #e5e7eb; }
.df-mini .gr-button:hover { background:#e5e7eb; }

/* ---------- AND 區塊卡片 ---------- */
#and_group .gr-group, #and_group .gr-panel, #and_group .gr-box {
  background: transparent !important; box-shadow: none !important; border: none !important;
}
#and_card { background:#fff; border:1px solid #e5e7eb; border-radius:10px; overflow:hidden; }
#and_card .section-title { padding:10px 12px; background:#f9fafb; border-bottom:1px solid #e5e7eb; font-weight:600; }
#and_group button[aria-label="New row"],
#and_group button[aria-label="Add row"],
#and_group button[title="New row"],
#and_group button[title="Add row"] { display:none !important; }

/* 是否 radio button */
#stop_now .wrap { display:flex; gap:12px; align-items:center; flex-wrap:wrap; }

/* ---------- 右側結果 sticky ---------- */
#right_sticky { position: sticky; top: 12px; align-self: start; }
#output_card { max-height: calc(100vh - 24px); overflow: auto; }
#outputbox textarea { min-height: calc(100vh - 140px); }

/* 右側結果下方按鈕列 */
#actions_right { margin-top: 8px; gap: 8px; }
""") as demo:
    gr.Markdown("## 一番賞模擬器 — 左側輸入參數，右側顯示模擬結果")

    with gr.Row():
        with gr.Column(scale=1):

            # ---- 主清單（庫存/價格）卡片 ----
            with gr.Group(elem_id="inv_group"):
                with gr.Column(elem_id="inv_card"):
                    inv_df = gr.Dataframe(
                        headers=["顏色", "數量", "價格", "是否刪除"],
                        datatype=["str", "number", "number", "bool"],
                        row_count=(0, "dynamic"),
                        col_count=(4, "fixed"),
                        value=defaults_df(),
                        label="基礎庫存（填顏色/數量/價格；勾選『是否刪除』後用下方『批量刪除』）",
                    )
                    with gr.Row(elem_id="df_actions"):
                        delete_last_btn = gr.Button("刪除最後一行", elem_classes=["df-mini"])
                        apply_delete_btn = gr.Button("批量刪除", elem_classes=["df-mini"])

            draw_price = gr.Number(label="單抽價格 (整數)", value=defaults_other()[0])

            # ---- 兌換規則：模式切換 + 兩種輸入 ----
            exchanges_mode = gr.Radio(
                label="兌換規則輸入方式",
                choices=["表格輸入", "文字輸入"],
                value="表格輸入",
            )

            exchanges_text = gr.Textbox(
                label="兌換規則（文字）JSON 或 'A*2->B; C*5->D')",
                lines=3,
                value=defaults_other()[1],
                visible=False,
            )

            exchanges_table = gr.Dataframe(
                label="兌換規則（表格）",
                headers=["來源顏色", "比例", "目標顏色"],
                datatype=["str", "number", "str"],
                col_count=(3, "fixed"),
                row_count=(0, "dynamic"),
                value=exchanges_df_from_text(defaults_other()[1]),
                interactive=True,
                wrap=True,
            )

            def _toggle_exchange_input(mode, cur_text, cur_df):
                if mode == "表格輸入":
                    return gr.update(visible=False, value=cur_text), gr.update(visible=True, value=exchanges_df_from_text(cur_text))
                else:
                    return gr.update(visible=True, value=exchanges_text_from_df(cur_df)), gr.update(visible=False)

            exchanges_mode.change(
                fn=_toggle_exchange_input,
                inputs=[exchanges_mode, exchanges_text, exchanges_table],
                outputs=[exchanges_text, exchanges_table],
            )

            # ---- 最大兌換目標顏色：Radio（含「無限制」） ----
            initial_colors = colors_from_table(defaults_df())
            _max_to = defaults_other()[2]
            _max_to_value = "無限制" if _max_to in (None, "", "無限制") else _max_to
            max_to_color = gr.Radio(
                label="最大兌換目標顏色",
                choices=["無限制"] + initial_colors,
                value=_max_to_value,
            )

            # ---- 任一目標（OR） ----
            stop_any = gr.CheckboxGroup(
                label="達成任一目標即停止（複選）",
                choices=["Gold", "Silver"],
                value=["Gold", "Silver"],
            )

            # ---- 全部達成（AND） ----
            with gr.Group(elem_id="and_group"):
                with gr.Column(elem_id="and_card"):
                    gr.Markdown("全部達成條件（AND）", elem_classes=["section-title"])
                    stop_all_colors = gr.CheckboxGroup(
                        label="必須全部達成（先勾顏色）",
                        choices=initial_colors,
                        value=[],
                    )
                    stop_all_table = gr.Dataframe(
                        label="全部達成的最少數量（僅顯示已勾選的顏色）",
                        headers=["顏色", "至少數量"],
                        datatype=["str", "number"],
                        col_count=(2, "fixed"),
                        row_count=(0, "dynamic"),
                        value=[],
                        interactive=True,
                    )

            # ✅ 單選（1=是、0=否）
            stop_immediately = gr.Radio(
                choices=[("是", 1), ("否", 0)],
                value=defaults_other()[4],
                label="條件達成後是否立即停止？",
                type="value",
                interactive=True,
                elem_id="stop_now",
            )

            per_round_draws = gr.Number(label="若未立即停止時，每回合抽數 (整數)", value=defaults_other()[5])
            num_runs = gr.Number(label="模擬次數 (整數)", value=defaults_other()[6])

            # seed 與 auto_seed
            _seed, _auto = defaults_other()[7], defaults_other()[8]
            seed = gr.Number(label="隨機種子 (整數)", value=_seed, interactive=not _auto)
            auto_seed = gr.Checkbox(label="自動產生隨機種子（每次執行都不同）", value=_auto)

        with gr.Column(scale=1, elem_id="right_sticky"):
            with gr.Group(elem_id="output_card"):
                output_box = gr.Textbox(
                    label="模擬結果:",
                    lines=28,
                    interactive=False,
                    elem_id="outputbox",
                )

            # 結果下方的操作列
            with gr.Row(elem_id="actions_right"):
                run_btn   = gr.Button("執行模擬", variant="primary")
                clear_btn = gr.Button("清空輸出")
                load_btn  = gr.Button("載入預設值")
                export_both_btn = gr.Button("下載設定與報告（JSON + TXT）")

            # 下載檔案元件（匯出後出現）
            export_file = gr.File(label="下載 user_defaults.json", interactive=False)
            report_file = gr.File(label="下載 simulation_report.txt", interactive=False)

    # ---- 表格刪除控制 ----
    delete_last_btn.click(fn=delete_last_row, inputs=[inv_df], outputs=[inv_df])
    apply_delete_btn.click(fn=apply_delete, inputs=[inv_df], outputs=[inv_df])

    # ---- 執行 / 清空 / 匯出 / 載入 ----
    run_btn.click(
        fn=run_simulation,
        inputs=[
            inv_df, draw_price,
            exchanges_mode, exchanges_text, exchanges_table,
            max_to_color,
            stop_any, stop_all_table, stop_immediately,
            per_round_draws, num_runs, seed, auto_seed
        ],
        outputs=[output_box, seed],
    )
    clear_btn.click(fn=clear_output, inputs=None, outputs=[output_box])

    export_both_btn.click(
        fn=export_settings_and_report,
        inputs=[
            inv_df, draw_price,
            exchanges_mode, exchanges_text, exchanges_table,
            max_to_color,
            stop_any, stop_all_table, stop_immediately,
            per_round_draws, num_runs, seed, auto_seed,
            output_box,  # 把右側輸出框內容也傳進去，產 TXT 用
        ],
        outputs=[export_file, report_file],
    )

    def _load_defaults_with_all_ui():
        ddf = defaults_df()
        colors = [row[0] for row in ddf]
        (draw_p, exg, max_to, stop_any_list, stop_now, prd, runs, sd, use_auto) = defaults_other()
        valid_stop_any = [c for c in (stop_any_list or []) if c in colors]
        radio_choices = ["無限制"] + colors
        radio_value = "無限制" if max_to in (None, "", "無限制") else (max_to if max_to in radio_choices else "無限制")
        ex_text = exg
        ex_tbl  = exchanges_df_from_text(exg)
        return (
            ddf,
            gr.update(choices=colors, value=valid_stop_any),
            gr.update(choices=colors, value=[]),
            [],
            draw_p,
            ex_text, ex_tbl,
            gr.update(choices=radio_choices, value=radio_value),
            stop_now, prd, runs,
            gr.update(value=sd, interactive=not use_auto),
            use_auto,
        )

    load_btn.click(
        fn=_load_defaults_with_all_ui,
        inputs=None,
        outputs=[
            inv_df, stop_any, stop_all_colors, stop_all_table,
            draw_price,
            exchanges_text, exchanges_table,
            max_to_color, stop_immediately,
            per_round_draws, num_runs, seed, auto_seed
        ],
    )

    # ---- 同步：主表改變 -> 更新 stop_any / stop_all / max_to_color ----
    inv_df.change(fn=update_stop_any_options, inputs=[inv_df, stop_any], outputs=[stop_any])
    inv_df.change(fn=sync_stop_all_choices, inputs=[inv_df, stop_all_colors, stop_all_table], outputs=[stop_all_colors, stop_all_table])
    inv_df.change(fn=update_max_to_options, inputs=[inv_df, max_to_color], outputs=[max_to_color])
    stop_all_colors.change(fn=sync_stop_all_choices, inputs=[inv_df, stop_all_colors, stop_all_table], outputs=[stop_all_colors, stop_all_table])

    # seed 互動鎖定
    def _toggle_seed_interactive(checked: bool):
        return gr.update(interactive=not checked)
    auto_seed.change(fn=_toggle_seed_interactive, inputs=[auto_seed], outputs=[seed])

if __name__ == "__main__":
    demo.launch(favicon_path="icon.png")
