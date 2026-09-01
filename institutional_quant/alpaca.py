from __future__ import annotations

from datetime import datetime, timedelta, timezone

import httpx

from .config import Settings
from .schemas import PaperOrderPreview, PaperTarget


class AlpacaPaperClient:
    """Paper-only broker adapter. It cannot be configured with a live trading host."""

    def __init__(self, settings: Settings, transport: httpx.AsyncBaseTransport | None = None):
        if settings.alpaca_paper_base_url != "https://paper-api.alpaca.markets":
            raise ValueError("Only Alpaca paper trading is supported")
        self.settings = settings
        self.transport = transport
        self._previews: dict[str, PaperOrderPreview] = {}

    def _headers(self) -> dict[str, str]:
        if not self.settings.alpaca_paper_key or not self.settings.alpaca_paper_secret:
            raise RuntimeError("ALPACA_PAPER_KEY and ALPACA_PAPER_SECRET are required")
        return {
            "APCA-API-KEY-ID": self.settings.alpaca_paper_key,
            "APCA-API-SECRET-KEY": self.settings.alpaca_paper_secret,
        }

    async def preview(self, targets: list[PaperTarget]) -> list[PaperOrderPreview]:
        headers = self._headers()
        async with httpx.AsyncClient(
            base_url=self.settings.alpaca_paper_base_url,
            headers=headers,
            timeout=30,
            transport=self.transport,
        ) as trading:
            account_response, positions_response = (
                await trading.get("/v2/account"),
                await trading.get("/v2/positions"),
            )
            account_response.raise_for_status()
            positions_response.raise_for_status()
            equity = float(account_response.json()["equity"])
            positions = {row["symbol"]: float(row["qty"]) for row in positions_response.json()}
        symbols = sorted({target.symbol.upper() for target in targets})
        async with httpx.AsyncClient(
            base_url=self.settings.alpaca_data_base_url,
            headers=headers,
            timeout=30,
            transport=self.transport,
        ) as market:
            response = await market.get(
                "/v2/stocks/trades/latest", params={"symbols": ",".join(symbols)}
            )
            response.raise_for_status()
            trades = response.json().get("trades", {})
        output: list[PaperOrderPreview] = []
        expiration = datetime.now(timezone.utc) + timedelta(minutes=5)
        for target in targets:
            symbol = target.symbol.upper()
            price = float(trades[symbol]["p"])
            target_qty = equity * target.target_weight / price
            delta = target_qty - positions.get(symbol, 0.0)
            if abs(delta) * price < 1.0:
                continue
            preview = PaperOrderPreview(
                symbol=symbol,
                side="buy" if delta > 0 else "sell",
                qty=round(abs(delta), 6),
                estimated_price=price,
                estimated_notional=round(abs(delta) * price, 2),
                expires_at=expiration,
            )
            self._previews[preview.preview_id] = preview
            output.append(preview)
        return output

    async def submit(self, orders: list[PaperOrderPreview], approved: bool) -> list[dict]:
        if not approved:
            raise ValueError("Explicit approval is required")
        now = datetime.now(timezone.utc)
        for order in orders:
            stored = self._previews.get(order.preview_id)
            if stored is None or stored.model_dump() != order.model_dump():
                raise ValueError(f"Unknown or modified preview: {order.preview_id}")
            expiration = stored.expires_at
            if expiration and expiration.tzinfo is None:
                expiration = expiration.replace(tzinfo=timezone.utc)
            if expiration and expiration < now:
                raise ValueError(f"Expired preview: {order.preview_id}")
        results = []
        async with httpx.AsyncClient(
            base_url=self.settings.alpaca_paper_base_url,
            headers=self._headers(),
            timeout=30,
            transport=self.transport,
        ) as client:
            for order in orders:
                response = await client.post(
                    "/v2/orders",
                    json={
                        "symbol": order.symbol,
                        "qty": str(order.qty),
                        "side": order.side,
                        "type": "market",
                        "time_in_force": "day",
                        "client_order_id": f"iq-{order.preview_id[:24]}",
                    },
                )
                response.raise_for_status()
                results.append(response.json())
                self._previews.pop(order.preview_id, None)
        return results
