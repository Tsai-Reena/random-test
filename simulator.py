# simulator.py
# -*- coding: utf-8 -*-

import random
from collections import defaultdict, Counter
from copy import deepcopy
from typing import Dict, List, Tuple


# =========================
# 資料結構與基本工具
# =========================
# 註解全部繁體中文；print 一律英文


def draw_one(inventory: Dict[str, int], rng: random.Random) -> str:
    """
    從當前庫存中隨機抽出一顆球（依數量比例）
    - 庫存為 {顏色: 數量}
    - 回傳被抽中的顏色字串
    """
    total = sum(inventory.values())
    if total <= 0:
        return ""  # 無球可抽
    # 以 cumulative sum 快速做加權抽樣
    r = rng.randint(1, total)
    cum = 0
    for color, cnt in inventory.items():
        cum += cnt
        if r <= cum:
            return color
    # 理論上不會到這行
    return ""


def apply_exchanges(
    counts: Dict[str, int],
    exchanges: List[Tuple[str, int, str]],
    prices: Dict[str, int],
    max_to_color: str = None
) -> Dict[str, int]:
    """
    依據兌換規則反覆兌換直到無法再兌換為止
    - counts: 當前各顏色數量
    - exchanges: 規則列表 [(from_color, ratio, to_color), ...]
    - prices: 各顏色單價（成本用）
    - max_to_color: 若指定，則禁止兌換到此顏色之上的更高級（例如 "Silver" 表示最多換到銀）
      （此參數用於類似「最多換到銀，無法換到金」）
    備註：成本以「兌換後的最終球數乘以其單價」計算，因此此函式只需輸出最終 counts
    """
    # 複製，避免原始資料被改動
    result = counts.copy()

    # 建立一個可兌換目標集合（若有 max_to_color 限制，將超出者濾掉）
    def rule_allowed(to_c: str) -> bool:
        if max_to_color is None:
            return True
        # 若指定最多換到某色，則禁止 to_c 超過該等級
        # 這裡假設價格愈高等級愈高，以價格判斷等級
        # => 價格高於 max_to_color 的，就不允許兌換
        return prices[to_c] <= prices[max_to_color]

    # 反覆兌換直到沒有規則可套用
    changed = True
    while changed:
        changed = False
        for from_c, ratio, to_c in exchanges:
            if not rule_allowed(to_c):
                continue
            have = result.get(from_c, 0)
            if have >= ratio and ratio > 0:
                n = have // ratio  # 可兌換幾次
                if n > 0:
                    result[from_c] = have - n * ratio
                    result[to_c] = result.get(to_c, 0) + n
                    changed = True
    return result


def meets_stop_on_any(drawn_counts: Dict[str, int], stop_on_any: List[str]) -> bool:
    """
    停止條件：只要抽到任一指定顏色（至少 1 顆）即停
    """
    if not stop_on_any:
        return False
    for c in stop_on_any:
        if drawn_counts.get(c, 0) > 0:
            return True
    return False


def meets_stop_on_all(drawn_counts: Dict[str, int], stop_on_all: Dict[str, int]) -> bool:
    """
    停止條件：同時滿足多顏色數量下限（AND 條件）
    例如 {"White": 2, "Green": 1} 代表要至少 2 白且至少 1 綠
    """
    if not stop_on_all:
        return False
    for c, k in stop_on_all.items():
        if drawn_counts.get(c, 0) < k:
            return False
    return True


def compute_value_after_exchange(
    counts_after_exchange: Dict[str, int],
    prices: Dict[str, int]
) -> int:
    """
    以兌換後的最終球數乘以其單價計價（成本）
    """
    total_value = 0
    for c, q in counts_after_exchange.items():
        total_value += prices.get(c, 0) * q
    return total_value


# =========================
# 模擬器主流程
# =========================

def simulate_round(
    base_inventory: Dict[str, int],
    prices: Dict[str, int],
    draw_price: int,
    rng: random.Random,
    *,
    exchanges: List[Tuple[str, int, str]] = None,
    max_exchange_to_color: str = None,
    stop_immediately: bool = True,
    per_round_draws: int = 10,
    stop_on_any: List[str] = None,
    stop_on_all: Dict[str, int] = None,
) -> Dict[str, any]:
    """
    進行單一輪的抽獎模擬
    - base_inventory: 初始庫存（每輪開始會 reset）
    - prices: 各顏色單價（成本）
    - draw_price: 單抽售價（收入）
    - exchanges: 兌換規則 [(from_color, ratio, to_color), ...]
    - max_exchange_to_color: 最多兌換到此等級（例如 "Silver"）
    - stop_immediately: True=逐抽命中即停；False=批次連抽 per_round_draws
    - per_round_draws: 批次模式下每輪的連抽數量
    - stop_on_any: 只要抽到其中任一色（≥1）就停，例如 ["Gold", "Silver"]
    - stop_on_all: 必須同時達到多色下限，例如 {"White":2, "Green":1}
    回傳：包含該輪各項結果的字典（提供英文欄位）
    """
    inv = deepcopy(base_inventory)
    drawn = Counter()
    draws = 0

    # 逐抽模式：每抽完一次就檢查是否滿足停止條件
    if stop_immediately:
        while sum(inv.values()) > 0:
            c = draw_one(inv, rng)
            if c == "":
                break
            inv[c] -= 1
            drawn[c] += 1
            draws += 1

            # 抽到任一指定顏色即停（如金/銀）
            if meets_stop_on_any(drawn, stop_on_any or []):
                break

            # 同時滿足多色門檻即停
            if meets_stop_on_all(drawn, stop_on_all or {}):
                break

    # 批次模式：一次抽 per_round_draws 顆，不提前中止
    else:
        k = min(per_round_draws, sum(inv.values()))
        for _ in range(k):
            c = draw_one(inv, rng)
            if c == "":
                break
            inv[c] -= 1
            drawn[c] += 1
            draws += 1
        # 批次模式下，抽完後才檢查「抽到任一指定顏色」與「多色門檻」
        # 同時也會在兌換後檢查（例如兌換得到銀，也視為該輪可終止）
        # —— 但此處的「是否終止」只影響回報資訊，不影響本輪抽數（因為已抽完）
        # 若你希望「抽到就立刻不再抽」，請用 stop_immediately=True

    # 兌換（可鏈式）
    counts_raw = dict(drawn)
    if exchanges:
        counts_after = apply_exchanges(counts_raw, exchanges, prices, max_to_color=max_exchange_to_color)
    else:
        counts_after = counts_raw.copy()

    # 若金或銀是在兌換後出現，也視為本輪達到終止條件（回報用）
    hit_any_after = meets_stop_on_any(counts_after, stop_on_any or [])
    hit_all_after = meets_stop_on_all(counts_after, stop_on_all or {})

    # 計價（收入與成本）
    revenue = draws * draw_price
    cost = compute_value_after_exchange(counts_after, prices)
    profit = revenue - cost

    # 回補庫存（供下一輪使用）—— 此函式外層會每輪都用 base_inventory 重新開始，所以無需在此修改 base_inventory

    # 建立回傳（英文明細）
    return {
        "draws": draws,
        "result_raw": counts_raw,
        "result_converted": counts_after,
        "revenue": revenue,
        "cost": cost,
        "profit": profit,
        "stopped_on_any_before_exchange": meets_stop_on_any(drawn, stop_on_any or []),
        "stopped_on_all_before_exchange": meets_stop_on_all(drawn, stop_on_all or {}),
        "stopped_on_any_after_exchange": hit_any_after,
        "stopped_on_all_after_exchange": hit_all_after,
    }


def simulate_many(
    num_runs: int,
    base_inventory: Dict[str, int],
    prices: Dict[str, int],
    draw_price: int,
    *,
    seed: int = 42,
    exchanges: List[Tuple[str, int, str]] = None,
    max_exchange_to_color: str = None,
    stop_immediately: bool = True,
    per_round_draws: int = 10,
    stop_on_any: List[str] = None,
    stop_on_all: Dict[str, int] = None,
) -> None:
    """
    進行多輪模擬並以『中文』列印摘要
    """
    rng = random.Random(seed)
    summary_draws = 0
    summary_profit = 0
    cumulative_raw = Counter()
    cumulative_conv = Counter()

    def _yn(b: bool) -> str:
        return "是" if b else "否"

    for i in range(1, num_runs + 1):
        res = simulate_round(
            base_inventory=base_inventory,
            prices=prices,
            draw_price=draw_price,
            rng=rng,
            exchanges=exchanges,
            max_exchange_to_color=max_exchange_to_color,
            stop_immediately=stop_immediately,
            per_round_draws=per_round_draws,
            stop_on_any=stop_on_any,
            stop_on_all=stop_on_all,
        )

        # —— 每輪明細（中文）——
        print(f"—— 第 {i} 輪模擬 ——")
        print(f"本輪抽數：{res['draws']}")
        print(f"抽中結果（未兌換）：{dict(res['result_raw'])}")
        print(f"抽中結果（兌換後）：{dict(res['result_converted'])}")
        print(f"營收（抽數 × 單抽價）：{res['revenue']}")
        print(f"成本（兌換後球數 × 單價）：{res['cost']}")
        pl = res["profit"]
        sign = "+" if pl >= 0 else "-"
        print(f"損益：{sign}{abs(pl)}")
        print(f"抽取過程中是否因『任一目標』達成而停止：{_yn(res['stopped_on_any_before_exchange'])}")
        print(f"抽取過程中是否因『全部達成（AND）』而停止：{_yn(res['stopped_on_all_before_exchange'])}")
        print(f"兌換後是否符合『任一目標』：{_yn(res['stopped_on_any_after_exchange'])}（此為事後檢查，不會影響本輪抽數）")
        print(f"兌換後是否符合『全部達成（AND）』：{_yn(res['stopped_on_all_after_exchange'])}（此為事後檢查，不會影響本輪抽數）")
        print()

        # 累計統計
        summary_draws += res["draws"]
        summary_profit += res["profit"]
        cumulative_raw.update(res["result_raw"])
        cumulative_conv.update(res["result_converted"])

    # —— 總結（中文）——
    avg_draws = summary_draws / num_runs if num_runs > 0 else 0.0
    avg_profit = summary_profit / num_runs if num_runs > 0 else 0.0

    print("—— 模擬總結 ——")
    print(f"總輪數：{num_runs}")
    print(f"平均每輪抽數：{avg_draws:.2f}")
    print(f"平均每輪損益：{avg_profit:.2f}")
    print(f"平均抽中（未兌換）：{ {k: v/num_runs for k, v in cumulative_raw.items()} }")
    print(f"平均抽中（兌換後）：{ {k: v/num_runs for k, v in cumulative_conv.items()} }")

# =========================
# 範例執行（可依需求修改）
# =========================
if __name__ == "__main__":
    # ---- 基本設定（顏色、數量、單價）----
    base_inventory = {
        "White": 10,
        "Yellow": 5,
        "Silver": 1,
        "Gold": 1,
    }
    prices = {
        "White": 100,   # 白球單價（成本）
        "Yellow": 300,  # 黃球單價（成本）
        "Silver": 1000, # 銀球單價（成本）
        "Gold": 5000,   # 金球單價（成本）
    }
    draw_price = 800  # 單抽售價（收入）

    # ---- 兌換規則 ----
    # 例：2 White -> 1 Yellow；5 Yellow -> 1 Silver；最多換到 Silver（不可換到 Gold）
    exchanges = [
        ("White", 2, "Yellow"),
        ("Yellow", 5, "Silver"),
        # 若希望允許 Silver -> Gold，可再加：("Silver", 2, "Gold")
    ]
    max_exchange_to_color = "Silver"  # 最多換到銀

    # ---- 停止條件 ----
    # 1) 抽到任一指定色即停（如金或銀）
    stop_on_any = ["Gold", "Silver"]

    # 2) 或者使用 AND 條件（例如至少 2 白且 1 綠）：這裡先示範不使用
    stop_on_all = {}  # 例如 {"White": 2, "Yellow": 1}

    # ---- 模式選擇：逐抽或連抽 ----
    stop_immediately = True   # True=逐抽命中即停；False=批次固定連抽
    per_round_draws = 10      # 批次模式下每輪連抽數

    # ---- 跑多輪模擬 ----
    simulate_many(
        num_runs=10,                        # 模擬 10 輪
        base_inventory=base_inventory,      # 每輪開始皆回到此庫存
        prices=prices,
        draw_price=draw_price,
        seed=123,                           # 隨機種子
        exchanges=exchanges,
        max_exchange_to_color=max_exchange_to_color,
        stop_immediately=stop_immediately,
        per_round_draws=per_round_draws,
        stop_on_any=stop_on_any,
        stop_on_all=stop_on_all,
    )
