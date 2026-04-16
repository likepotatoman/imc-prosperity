from datamodel import OrderDepth, TradingState, Order, Trade
from typing import Dict, List, Tuple, Any
import json
import math


class Trader:
    """
    Round-1 microstructure strategy with:
    - Fair value estimation (anchor + microprice / EMA + alpha signals)
    - Alpha extraction (book imbalance, trade-flow, momentum, z-score mean reversion)
    - Dynamic strategy selection (aggressive taking + market making)
    - Inventory-aware skewing and hard limit enforcement
    """

    PRODUCT_LIMITS = {
        # Tuned for tutorial round-0 style liquidity/limits.
        "EMERALDS": 80,
        "TOMATOES": 50,
        "PEARLS": 20,
        "BANANAS": 20,
        "STARFRUIT": 20,
    }

    DEFAULT_PARAMS: Dict[str, Dict[str, float]] = {
        # Calibrated from bundled round-0 CSVs (fallback values).
        "EMERALDS": {
            "anchor": 10000.0,
            "vol": 1.03,
            "base_spread": 21.823,
            "ema_alpha": 0.052,
            "inv_lambda": 0.125,
            "signal_w_imb": -2.068,
            "signal_w_flow": 0.068,
            "signal_w_mom": -0.07,
            "signal_w_z": -1.156,
            "take_edge": 0.867,
            "max_clip": 24,
            "take_clip": 10,
            "is_stable": 1.0,
        },
        "TOMATOES": {
            "anchor": 5000.0,
            "vol": 33.7,
            "base_spread": 17.8,
            "ema_alpha": 0.112,
            "inv_lambda": 0.155,
            "signal_w_imb": -0.311,
            "signal_w_flow": 0.154,
            "signal_w_mom": -0.194,
            "signal_w_z": -0.869,
            "take_edge": 2.6,
            "max_clip": 18,
            "take_clip": 8,
            "is_stable": 0.0,
        },
        # Compatibility aliases for standard IMC tutorials.
        "PEARLS": {
            "anchor": 10000.0,
            "vol": 1.0,
            "base_spread": 4.0,
            "ema_alpha": 0.03,
            "inv_lambda": 0.10,
            "signal_w_imb": -1.2,
            "signal_w_flow": 0.2,
            "signal_w_mom": -0.1,
            "signal_w_z": -1.0,
            "take_edge": 1.2,
            "max_clip": 7,
            "is_stable": 1.0,
        },
        "BANANAS": {
            "anchor": 5000.0,
            "vol": 30.0,
            "base_spread": 6.0,
            "ema_alpha": 0.20,
            "inv_lambda": 0.20,
            "signal_w_imb": -0.3,
            "signal_w_flow": 0.7,
            "signal_w_mom": -0.3,
            "signal_w_z": -0.9,
            "take_edge": 2.0,
            "max_clip": 6,
            "is_stable": 0.0,
        },
        "STARFRUIT": {
            "anchor": 5000.0,
            "vol": 30.0,
            "base_spread": 6.0,
            "ema_alpha": 0.20,
            "inv_lambda": 0.20,
            "signal_w_imb": -0.3,
            "signal_w_flow": 0.7,
            "signal_w_mom": -0.3,
            "signal_w_z": -0.9,
            "take_edge": 2.0,
            "max_clip": 6,
            "is_stable": 0.0,
        },
    }

    HISTORY_LEN = 60
    TRADE_LEN = 80

    def __init__(self) -> None:
        self.calibration = self._bootstrap_calibration()

    def _bootstrap_calibration(self) -> Dict[str, Dict[str, float]]:
        """Self-contained calibration: no file or external data dependencies."""
        return {k: v.copy() for k, v in self.DEFAULT_PARAMS.items()}

    def _load_state(self, trader_data: str) -> Dict[str, Any]:
        if not trader_data:
            return {"products": {}, "last_ts": -1}
        try:
            state = json.loads(trader_data)
            if "products" not in state:
                state["products"] = {}
            if "last_ts" not in state:
                state["last_ts"] = -1
            return state
        except Exception:
            return {"products": {}, "last_ts": -1}

    def _product_state(self, mem: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        products = mem.setdefault("products", {})
        if symbol not in products:
            products[symbol] = {
                "mids": [],
                "imbalances": [],
                "trade_px": [],
                "trade_signed_vol": [],
                "ema_mid": None,
                "last_trade_ts": -1,
            }
        return products[symbol]

    @staticmethod
    def _best_prices(depth: OrderDepth) -> Tuple[int, int]:
        if not depth.buy_orders or not depth.sell_orders:
            return None, None
        best_bid = max(depth.buy_orders)
        best_ask = min(depth.sell_orders)
        return best_bid, best_ask

    @staticmethod
    def _book_volume(depth: OrderDepth, levels: int = 3) -> Tuple[float, float]:
        bids = sorted(depth.buy_orders.items(), reverse=True)[:levels]
        asks = sorted(depth.sell_orders.items())[:levels]
        bid_vol = sum(max(0, v) for _, v in bids)
        ask_vol = sum(max(0, -v) for _, v in asks)
        return float(bid_vol), float(ask_vol)

    @staticmethod
    def _microprice(best_bid: int, best_ask: int, bid_vol: float, ask_vol: float) -> float:
        denom = bid_vol + ask_vol
        if denom <= 0:
            return (best_bid + best_ask) / 2.0
        return (best_bid * ask_vol + best_ask * bid_vol) / denom

    def _apply_position_limit(self, product: str, position: int, desired_qty: int) -> int:
        limit = self.PRODUCT_LIMITS.get(product, 20)
        if desired_qty > 0:
            return max(0, min(desired_qty, limit - position))
        if desired_qty < 0:
            return min(0, max(desired_qty, -limit - position))
        return 0

    def _pseudo_noise(self, timestamp: int, symbol: str) -> float:
        key = (timestamp // 100 + sum(ord(c) for c in symbol) * 13) % 97
        return (key / 48.0) - 1.0

    def compute_signals(
        self,
        symbol: str,
        state: TradingState,
        depth: OrderDepth,
        pstate: Dict[str, Any],
    ) -> Dict[str, float]:
        best_bid, best_ask = self._best_prices(depth)
        if best_bid is None:
            return {
                "mid": 0.0,
                "imbalance": 0.0,
                "flow": 0.0,
                "momentum": 0.0,
                "zscore": 0.0,
                "micro": 0.0,
                "signal_strength": 0.0,
            }

        mid = 0.5 * (best_bid + best_ask)
        bid_vol, ask_vol = self._book_volume(depth, levels=3)
        imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-9)
        micro = self._microprice(best_bid, best_ask, bid_vol, ask_vol)

        mids = pstate["mids"]
        mids.append(mid)
        if len(mids) > self.HISTORY_LEN:
            del mids[:-self.HISTORY_LEN]

        imbs = pstate["imbalances"]
        imbs.append(imbalance)
        if len(imbs) > self.HISTORY_LEN:
            del imbs[:-self.HISTORY_LEN]

        prev_ema = pstate.get("ema_mid")
        alpha = self.calibration.get(symbol, self.DEFAULT_PARAMS["BANANAS"])["ema_alpha"]
        ema_mid = mid if prev_ema is None else (alpha * mid + (1.0 - alpha) * prev_ema)
        pstate["ema_mid"] = ema_mid

        # Trade flow via uptick-rule signing for market trades.
        trades = state.market_trades.get(symbol, [])
        trade_px = pstate["trade_px"]
        trade_sv = pstate["trade_signed_vol"]
        last_trade_ts = pstate.get("last_trade_ts", -1)

        for tr in trades:
            if tr.timestamp < last_trade_ts:
                continue
            px = float(tr.price)
            qty = float(tr.quantity)
            prev_px = trade_px[-1] if trade_px else px
            sign = 1.0 if px > prev_px else (-1.0 if px < prev_px else 0.0)
            if sign == 0.0 and imbalance != 0.0:
                sign = 1.0 if imbalance > 0 else -1.0
            trade_px.append(px)
            trade_sv.append(sign * qty)
            last_trade_ts = max(last_trade_ts, tr.timestamp)

        if len(trade_px) > self.TRADE_LEN:
            del trade_px[:-self.TRADE_LEN]
        if len(trade_sv) > self.TRADE_LEN:
            del trade_sv[:-self.TRADE_LEN]
        pstate["last_trade_ts"] = last_trade_ts

        # Momentum from both mid trajectory and trade flow.
        mom = 0.0
        if len(mids) >= 8:
            mom = (mids[-1] - mids[-8]) / 7.0
        flow = 0.0
        if trade_sv:
            window = trade_sv[-12:]
            denom = sum(abs(x) for x in window) + 1e-9
            flow = sum(window) / denom

        zscore = 0.0
        if len(mids) >= 12:
            tail = mids[-20:]
            mu = sum(tail) / len(tail)
            var = sum((x - mu) ** 2 for x in tail) / len(tail)
            sigma = math.sqrt(max(var, 1e-9))
            zscore = (mid - mu) / sigma

        signal_strength = abs(imbalance) + 0.8 * abs(flow) + 0.35 * abs(mom) + 0.25 * abs(zscore)

        return {
            "mid": mid,
            "imbalance": imbalance,
            "flow": flow,
            "momentum": mom,
            "zscore": zscore,
            "micro": micro,
            "ema_mid": ema_mid,
            "signal_strength": signal_strength,
        }

    def risk_adjustment(self, fair_value: float, position: int, product: str) -> float:
        params = self.calibration.get(product, self.DEFAULT_PARAMS["BANANAS"])
        inv_lambda = params["inv_lambda"]
        return fair_value - inv_lambda * position

    def compute_fair_value(
        self,
        product: str,
        position: int,
        signals: Dict[str, float],
    ) -> float:
        params = self.calibration.get(product, self.DEFAULT_PARAMS["BANANAS"])

        if params["is_stable"] > 0.5:
            fair = 0.65 * params["anchor"] + 0.35 * signals["micro"]
            fair += 0.5 * params["signal_w_imb"] * signals["imbalance"]
        else:
            fair = 0.55 * signals.get("ema_mid", signals["mid"]) + 0.20 * signals["micro"] + 0.25 * signals["mid"]
            fair += params["signal_w_imb"] * signals["imbalance"]
            fair += params["signal_w_flow"] * signals["flow"]
            fair += params["signal_w_mom"] * signals["momentum"]
            fair += params["signal_w_z"] * signals["zscore"]

        return self.risk_adjustment(fair, position, product)

    def generate_quotes(
        self,
        product: str,
        depth: OrderDepth,
        fair_value: float,
        signals: Dict[str, float],
        position: int,
        timestamp: int,
    ) -> Tuple[int, int, int]:
        params = self.calibration.get(product, self.DEFAULT_PARAMS["BANANAS"])
        limit = self.PRODUCT_LIMITS.get(product, 20)
        pos_ratio = position / max(1.0, float(limit))

        best_bid, best_ask = self._best_prices(depth)
        base_spread = params["base_spread"]
        vol_term = 0.08 * params["vol"]
        signal_term = 0.55 * signals["signal_strength"]
        spread = max(2.0, base_spread * 0.35 + vol_term + signal_term)

        lean = 0.45 * signals["imbalance"] + 0.35 * signals["flow"]
        inv_skew = pos_ratio * (0.45 * spread)

        center = fair_value + lean - inv_skew
        center += self._pseudo_noise(timestamp, product) * 0.2

        bid_px = int(math.floor(center - spread / 2.0))
        ask_px = int(math.ceil(center + spread / 2.0))

        if best_bid is not None and best_ask is not None:
            # Queue-positioning: modestly improve without crossing.
            bid_px = min(max(bid_px, best_bid + 1), best_ask - 1)
            ask_px = max(min(ask_px, best_ask - 1), bid_px + 1)

        # Scale passive quote size with product-specific max clip and inventory usage.
        max_clip = int(params["max_clip"])
        utilization = max(0.2, 1.0 - 0.85 * abs(pos_ratio))
        clip = max(1, int(round(max_clip * utilization)))

        return bid_px, ask_px, clip

    def _aggressive_buy(
        self,
        product: str,
        depth: OrderDepth,
        position: int,
        max_qty: int,
        max_price: int,
    ) -> Tuple[List[Order], int]:
        orders: List[Order] = []
        qty_left = max_qty
        for ask in sorted(depth.sell_orders):
            if qty_left <= 0 or ask > max_price:
                break
            avail = max(0, -depth.sell_orders[ask])
            take = min(qty_left, avail)
            take = self._apply_position_limit(product, position, take)
            if take <= 0:
                continue
            orders.append(Order(product, ask, take))
            position += take
            qty_left -= take
        return orders, position

    def _aggressive_sell(
        self,
        product: str,
        depth: OrderDepth,
        position: int,
        max_qty: int,
        min_price: int,
    ) -> Tuple[List[Order], int]:
        orders: List[Order] = []
        qty_left = max_qty
        for bid in sorted(depth.buy_orders, reverse=True):
            if qty_left <= 0 or bid < min_price:
                break
            avail = max(0, depth.buy_orders[bid])
            take = min(qty_left, avail)
            qty = self._apply_position_limit(product, position, -take)
            if qty >= 0:
                continue
            orders.append(Order(product, bid, qty))
            position += qty
            qty_left -= abs(qty)
        return orders, position

    def execution_engine(
        self,
        product: str,
        state: TradingState,
        depth: OrderDepth,
        fair_value: float,
        signals: Dict[str, float],
        position: int,
    ) -> List[Order]:
        params = self.calibration.get(product, self.DEFAULT_PARAMS["BANANAS"])
        orders: List[Order] = []
        best_bid, best_ask = self._best_prices(depth)
        if best_bid is None:
            return orders

        edge = params["take_edge"] + 0.25 * abs(position) / max(1.0, self.PRODUCT_LIMITS.get(product, 20))
        bullish = signals["imbalance"] > 0.38 or signals["flow"] > 0.32
        bearish = signals["imbalance"] < -0.38 or signals["flow"] < -0.32

        # Aggressive taking only when alpha signal and price dislocation align.
        if best_ask <= int(math.floor(fair_value - edge)) or (bullish and best_ask < fair_value - 0.5):
            qty = max(1, int(params.get("take_clip", params["max_clip"])))
            buy_orders, position = self._aggressive_buy(product, depth, position, qty, int(math.floor(fair_value)))
            orders.extend(buy_orders)

        if best_bid >= int(math.ceil(fair_value + edge)) or (bearish and best_bid > fair_value + 0.5):
            qty = max(1, int(params.get("take_clip", params["max_clip"])))
            sell_orders, position = self._aggressive_sell(product, depth, position, qty, int(math.ceil(fair_value)))
            orders.extend(sell_orders)

        bid_px, ask_px, clip = self.generate_quotes(
            product=product,
            depth=depth,
            fair_value=fair_value,
            signals=signals,
            position=position,
            timestamp=state.timestamp,
        )

        # Adverse-selection guard: suppress quotes against strong opposing pressure.
        avoid_bid = signals["imbalance"] < -0.65 and signals["flow"] < -0.45
        avoid_ask = signals["imbalance"] > 0.65 and signals["flow"] > 0.45

        buy_qty = self._apply_position_limit(product, position, clip)
        if buy_qty > 0 and not avoid_bid:
            orders.append(Order(product, bid_px, buy_qty))

        sell_qty = self._apply_position_limit(product, position, -clip)
        if sell_qty < 0 and not avoid_ask:
            orders.append(Order(product, ask_px, sell_qty))

        return orders

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}
        mem = self._load_state(state.traderData)

        for product, depth in state.order_depths.items():
            pstate = self._product_state(mem, product)
            position = state.position.get(product, 0)

            signals = self.compute_signals(product, state, depth, pstate)
            if signals["mid"] == 0.0:
                result[product] = []
                continue

            fair_value = self.compute_fair_value(product, position, signals)
            result[product] = self.execution_engine(
                product=product,
                state=state,
                depth=depth,
                fair_value=fair_value,
                signals=signals,
                position=position,
            )

        mem["last_ts"] = state.timestamp
        trader_data = json.dumps(mem, separators=(",", ":"))
        conversions = 0
        return result, conversions, trader_data
