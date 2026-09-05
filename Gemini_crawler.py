import os
import sys
import time
from datetime import datetime, timedelta
import requests
import pandas as pd
import pandas_ta as ta
import twstock
import yfinance as yf
from google import genai
from google.genai import types
 
# 設定環境變數
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
LINE_TOKEN = os.environ.get("LINE_TOKEN")
USER_ID = os.environ.get("LINE_USER_ID")
 
def fetch_stock_df(ticker_symbol):
    """
    根據標的自動切換抓取來源：
    - 台股 (.TW / .TWO) 使用 twstock (抓取近 60 天資料以足夠計算技術指標)
    - 美股 使用 yfinance
    """
    try:
        if ".TW" in ticker_symbol or ".TWO" in ticker_symbol:
            code = ticker_symbol.replace(".TW", "").replace(".TWO", "")
            stock = twstock.Stock(code)
 
            # 計算約兩個月前（60 天）的年份與月份，確保有足夠 K 線計算 MACD/KD
            start_date = datetime.now() - timedelta(days=60)
            raw_data = stock.fetch_from(start_date.year, start_date.month)
            if not raw_data:
                return None, None
 
            df = pd.DataFrame(raw_data)
            df.rename(columns={
                'date': 'Date', 'open': 'Open', 'high': 'High',
                'low': 'Low', 'close': 'Close', 'capacity': 'Volume'
            }, inplace=True)
            df.set_index('Date', inplace=True)
            latest_date = df.index[-1].strftime('%Y-%m-%d')
            ref = f"資料來源: 臺灣證券交易所/櫃買中心 (截止: {latest_date})"
            return df, ref
        else:
            stock = yf.Ticker(ticker_symbol)
            df = stock.history(period="1y")
            if df.empty or df['Close'].isnull().all():
                return None, None
            latest_date = df.index[-1].strftime('%Y-%m-%d')
            ref = f"資料來源: Yahoo Finance (截止: {latest_date})"
            return df, ref
    except Exception as e:
        print(f"抓取 {ticker_symbol} 失敗: {e}")
        return None, None
 
def get_fundamentals(ticker_symbol):
    """
    透過 yfinance 取得基本面數據（台股/美股皆可用，因 Yahoo Finance 涵蓋 .TW / .TWO 標的）：
    - 本益比 (P/E)、股價淨值比 (P/B)
    - 毛利率、稅後淨利率、稅後淨利
    - 營收年增率（作為市場需求的量化參考指標之一）
    """
    def pct(x):
        return f"{x * 100:.1f}%" if isinstance(x, (int, float)) else "N/A"
 
    def big_num(x):
        if isinstance(x, (int, float)):
            if abs(x) >= 1e9:
                return f"{x / 1e9:.2f}B"
            elif abs(x) >= 1e6:
                return f"{x / 1e6:.2f}M"
            return f"{x:,.0f}"
        return "N/A"
 
    try:
        info = yf.Ticker(ticker_symbol).info
        pe = info.get('trailingPE', 'N/A')
        pb = info.get('priceToBook', 'N/A')
        gross_margin = info.get('grossMargins')
        net_margin = info.get('profitMargins')
        net_income = info.get('netIncomeToCommon')
        revenue_growth = info.get('revenueGrowth')
        currency = info.get('currency', '')
 
        fund_summary = (
            f"P/E: {pe}, P/B: {pb}, "
            f"毛利率: {pct(gross_margin)}, 稅後淨利率: {pct(net_margin)}, "
            f"稅後淨利: {big_num(net_income)} {currency}, 營收年增率: {pct(revenue_growth)}"
        )
        return fund_summary
    except Exception as e:
        print(f"{ticker_symbol} 基本面資料取得失敗: {e}")
        return "基本面資料: N/A（可能為未上市 ETF 或資料源無提供財報數據）"
 
def get_full_analysis(ticker_symbol):
    """完整指標計算與基本面整理"""
    df, ref = fetch_stock_df(ticker_symbol)
 
    if df is None or df.empty or len(df) < 15:
        return None, None, None, None
 
    latest = df.iloc[-1]
    close_val = latest['Close']
    high_val = latest['High']
    low_val = latest['Low']
 
    price_info = f"收盤: {close_val:.2f}, 最高: {high_val:.2f}, 最低: {low_val:.2f}"
 
    # 計算技術指標
    try:
        df.ta.macd(append=True)
        df.ta.rsi(append=True)
        df.ta.bbands(append=True)
        df.ta.stoch(append=True)
 
        bbl = [c for c in df.columns if 'BBL' in c][0]
        bbu = [c for c in df.columns if 'BBU' in c][0]
        macd = [c for c in df.columns if 'MACD_' in c and 'MACDh' not in c and 'MACDs' not in c][0]
        sk = [c for c in df.columns if 'STOCHk' in c][0]
        sd = [c for c in df.columns if 'STOCHd' in c][0]
        rsi = [c for c in df.columns if 'RSI' in c][0]
 
        df['PCT_B'] = (df['Close'] - df[bbl]) / (df[bbu] - df[bbl])
        latest_vals = df.iloc[-1]
 
        tech_summary = f"RSI: {latest_vals.get(rsi, 0):.2f}, MACD: {latest_vals.get(macd, 0):.2f}, KD: {latest_vals.get(sk, 0):.2f}/{latest_vals.get(sd, 0):.2f}, %B: {latest_vals.get('PCT_B', 0):.2f}"
    except Exception as e:
        print(f"{ticker_symbol} 技術指標計算錯誤: {e}")
        tech_summary = "指標計算中"
 
    # 基本面數據（毛利率、稅後淨利率、稅後淨利、營收年增率等）
    fund_summary = get_fundamentals(ticker_symbol)
 
    return price_info, tech_summary, fund_summary, ref
 
def build_prompt(ticker_symbol, ref, price_info, tech_summary, fund_summary):
    """組出要求 AI 依六大投資判斷問題回答的 prompt"""
    return f"""
分析標的：{ticker_symbol}
參考資料：{ref}
今日行情：{price_info}
技術面：{tech_summary}
基本面：{fund_summary}
 
你是一位客觀、嚴謹、不預設立場的資深投資分析師。請先實際搜尋並交叉查證公司最新的產品線、營收結構、客戶結構與產業新聞，確認上述基本面數據與你搜尋到的資訊是否一致，再依據以下 6 個問題逐項給出精煉結論（每項 1-2 句話，禁止空泛套話，需有具體依據）：
 
1. 受惠產品：公司真正受惠的是哪一項產品或業務線？
2. 真實需求 vs 市場想像：這項產品的需求是有實際訂單/出貨佐證，還是主要來自市場預期與想像？
3. 需求週期：目前需求屬於短期拉貨（例如庫存回補、單一大單），還是長期結構性趨勢？
4. 供需狀況：目前供需是否失衡？有無缺貨、漲價、產能滿載或擴產跡象？
5. 漲價動能：若有漲價，主要是終端需求推動（賣方市場），還是原物料/成本上漲後的轉嫁？
6. 客戶與能見度：主要客戶是誰（可指產業別，不確定則說明）？訂單能見度大約多長（例如：1季/2季/半年以上）？
 
【格式要求】
- 依上述 1~6 點條列輸出，每點前標註題號，不加開場白、問候語或免責聲明。
- 最後加一行【結論】：綜合以上，給出中性、不預設立場的觀察重點（非投資建議用語，例如避免使用「買進/賣出」等字眼，改用「值得留意」「風險在於」等中性描述）。
- 全文字數控制在 500 字以內。
"""
 
def generate_ai_analysis(client, prompt):
    """
    使用 Gemini 3.5 系列模型，並啟用 Google Search grounding，
    讓模型在回答前能實際搜尋最新資訊進行交叉查證：
    1. 主力模型：gemini-3.5-flash
    2. 備用模型：gemini-3.5-flash-lite
    """
    PRIMARY_MODEL = "gemini-3.5-flash"
    BACKUP_MODEL = "gemini-3.5-flash-lite"
 
    search_config = types.GenerateContentConfig(
        tools=[types.Tool(google_search=types.GoogleSearch())]
    )
 
    models_queue = [PRIMARY_MODEL, BACKUP_MODEL]
    last_error = ""
 
    for model_name in models_queue:
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=search_config
                )
                if response.text:
                    is_backup = (model_name != PRIMARY_MODEL)
                    version_tag = f"{model_name} [備用模型]" if is_backup else model_name
                    return response.text.strip(), version_tag
            except Exception as e:
                err_msg = str(e)
                last_error = err_msg
 
                # 若遇到 404/NOT_FOUND，代表該 model ID 不存在，直接跳出換下一個 model
                if "404" in err_msg or "NOT_FOUND" in err_msg:
                    break
 
                # 若遇到 503/429 流量限流，退避等待後重試
                if "503" in err_msg or "429" in err_msg or "UNAVAILABLE" in err_msg:
                    time.sleep((attempt + 1) * 6)
                else:
                    break
 
    return f"AI 分析失敗 (原因: {last_error[:100]})", "失敗"
 
def main():
    if not GEMINI_API_KEY:
        print("未設定 GEMINI_API_KEY")
        sys.exit(1)
 
    client = genai.Client(api_key=GEMINI_API_KEY)
 
    tickers = ["2330.TW", "0050.TW", "NVDA", "AMD", "MU"]
    report = f"📈 【Gemini AI 財務技術報告 Version 1.1.0】\n報告時間: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
 
    for t in tickers:
        try:
            price, tech, fund, ref = get_full_analysis(t)
            if price:
                prompt = build_prompt(t, ref, price, tech, fund)
                ai_advice, used_version = generate_ai_analysis(client, prompt)
                report += f"\n--- {t} ---\n【行情】{price}\n【指標】{tech}\n【基本】{fund}\n【AI建議 ({used_version})】\n{ai_advice}\n"
 
                # 每個標的隔 6 秒，降低觸發 API 流量上限的機率
                time.sleep(6)
            else:
                report += f"\n--- {t} ---\n⚠️ 無法取得該標的之有效行情資料\n"
        except Exception as e:
            report += f"\n{t} 分析失敗: {e}\n"
 
    # 安全裁切，確保最終 LINE 訊息不超過官方上限 4,000 字
    final_report = report[:3500] + "\n...(報告已截斷)" if len(report) > 3500 else report
 
    # LINE 發送邏輯
    if LINE_TOKEN and USER_ID:
        payload = {"to": USER_ID, "messages": [{"type": "text", "text": final_report}]}
        headers = {"Authorization": f"Bearer {LINE_TOKEN}", "Content-Type": "application/json"}
        res = requests.post("https://api.line.me/v2/bot/message/push", headers=headers, json=payload)
        print(f"LINE 發送狀態: {res.status_code}")
 
    print(final_report)
 
if __name__ == "__main__":
    main()
 
