import yfinance as yf
import pandas as pd
from datetime import datetime, timezone, timedelta
import urllib.request
import json
import os

# 從 GitHub 設定的 Secrets 中讀取 LINE_TOKEN
TOKEN = os.environ.get("LINE_TOKEN")

def get_stock_data(stock_id):
    """
    功能：抓取股價、財報與技術指標
    """
    try:
        # 使用 yfinance 取得股票數據
        ticker = yf.Ticker(f"{stock_id}.TW")
        hist = ticker.history(period="120d").ffill()
        info = ticker.info
        
        # 提取股票資訊
        price = info.get('currentPrice', 'N/A')
        eps = info.get('trailingEps', 'N/A')
        pe = info.get('trailingPE', 'N/A')
        pb = info.get('priceToBook', 'N/A')
        roe = info.get('returnOnEquity', 0) or 0
        g_m = info.get('grossMargins', 0) or 0
        o_m = info.get('operatingMargins', 0) or 0
        n_m = info.get('profitMargins', 0) or 0
        
        # 計算技術指標
        close = hist['Close']
        ma20 = close.rolling(20).mean().iloc[-1]
        std20 = close.rolling(20).std().iloc[-1]
        b_mid = float(ma20)
        b_top = float(ma20 + (2 * std20))
        b_bot = float(ma20 - (2 * std20))
        
        ema12 = close.ewm(span=12, adjust=False).mean().iloc[-1]
        ema26 = close.ewm(span=26, adjust=False).mean().iloc[-1]
        macd = float(ema12 - ema26)
        
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(window=14).mean().iloc[-1]
        loss = -delta.clip(upper=0).rolling(window=14).mean().iloc[-1]
        rsi = float(100 - (100 / (1 + (gain / loss)))) if loss != 0 else 100
        
        # 組裝成文字報告
        return (f"💰現價:{price}\n"
                f"【獲利】毛利:{g_m*100:.1f}%|營益:{o_m*100:.1f}%|稅後:{n_m*100:.1f}%\n"
                f"【技術】布林底:{b_bot:.0f}|頂:{b_top:.0f}|均:{b_mid:.0f}|MACD:{macd:.2f}|RSI:{rsi:.1f}\n"
                f"【估值】EPS:{eps}|P/E:{pe}|P/B:{pb}|ROE:{roe*100:.1f}%")
                
    except Exception as e:
        return f"資料抓取異常: {e}"

def main():
    # 檢查 TOKEN 是否設定
    if not TOKEN:
        print("❌ 錯誤：找不到 LINE_TOKEN，請檢查 GitHub Secrets 設定")
        return

    # 設定時間戳記
    now = datetime.now(timezone(timedelta(hours=8)))
    report = [f"📊 股市收盤報告 ({now.strftime('%m/%d %H:%M')})", "-"*15]
    
    # 定義要追蹤的股票
    stocks = {
        '2330': '台積電', 
        '2454': '聯發科', 
        '2395': '研華', 
        '2327': '國巨',
        '3231': '緯創',
        '2382': '廣達'
    }
    
    # 遍歷股票清單並獲取資料
    for sid, sname in stocks.items():
        data = get_stock_data(sid)
        report.append(f"【{sname}】\n{data}\n")
    
    # 將所有報告合併為單一字串
    report_text = "\n".join(report)
    
    # 設定廣播訊息的 payload 格式
    payload = {"messages": [{"type": "text", "text": report_text}]}
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {TOKEN}',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    
    # 發送廣播 API 請求
    req = urllib.request.Request("https://api.line.me/v2/bot/message/broadcast", 
                                 data=json.dumps(payload).encode('utf-8'), 
                                 headers=headers)
    
    try:
        # 執行請求並接收回應
        with urllib.request.urlopen(req, timeout=15) as response:
            print(f"✅ 廣播成功！LINE 回應代碼: {response.status}")
    except Exception as e:
        # 捕捉並列印錯誤
        print(f"❌ 廣播失敗: {e}")

if __name__ == "__main__":
    main()
