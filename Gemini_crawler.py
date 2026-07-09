import os
import yfinance as yf
import pandas_ta as ta
import google.generativeai as genai
import requests

# 設定
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
LINE_TOKEN = os.environ.get("LINE_TOKEN")
USER_ID = os.environ.get("LINE_USER_ID")
MODEL_VERSION = 'gemini-3.5-flash'

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(MODEL_VERSION)

def get_technical_analysis(ticker_symbol):
    stock = yf.Ticker(ticker_symbol)
    # 抓取 1 年數據
    df = stock.history(period="1y")
    
    # 計算指標
    df.ta.macd(append=True)     # MACD
    df.ta.rsi(append=True)      # RSI
    df.ta.bbands(append=True)   # 布林通道 (BBU, BBM, BBL)
    df.ta.stoch(append=True)    # KD 指標 (STOCHk, STOCHd)
    
    # 計算 %B: (Close - Lower Band) / (Upper Band - Lower Band)
    # BBL_5_2.0 為布林下緣, BBU_5_2.0 為布林上緣
    df['PCT_B'] = (df['Close'] - df['BBL_5_2.0']) / (df['BBU_5_2.0'] - df['BBL_5_2.0'])
    
    # 取得最新的一筆數據
    latest = df.iloc[-1]
    
    tech_summary = (
        f"RSI: {latest['RSI_14']:.2f}, "
        f"MACD: {latest['MACD_12_26_9']:.2f}, "
        f"KD(K/D): {latest['STOCHk_14_3_3']:.2f}/{latest['STOCHd_14_3_3']:.2f}, "
        f"%B: {latest['PCT_B']:.2f}"
    )
    
    # 取得基本面 (info)
    info = stock.info
    fundamentals = f"P/E: {info.get('trailingPE', 'N/A')}, P/B: {info.get('priceToBook', 'N/A')}, ROE: {info.get('returnOnEquity', 'N/A')}"
    
    return tech_summary, fundamentals

def main():
    tickers = ["2330.TW", "0050.TW", "NVDA", "AMD", "MU"]
    report = "📈 【AI 深度技術面波段報告】\n"
    
    for t in tickers:
        try:
            tech, fund = get_technical_analysis(t)
            prompt = f"""
            分析標的：{t} (時間週期：過去一年)
            基本面：{fund}
            技術面指標：{tech}
            
            請根據以上 1 年期技術指標與基本面資料，分析目前的波段趨勢：
            1. MACD 與 KD 是否出現黃金/死亡交叉？
            2. 目前 RSI 與 %B 是否顯示超買或超賣？
            3. 提供專業的波段買賣建議。
            
            格式要求：請使用條列式，內容精簡適合手機閱讀。
            """
            response = model.generate_content(prompt)
            report += f"\n--- {t} ---\n{response.text}\n"
        except Exception as e:
            report += f"\n{t} 分析失敗: {e}\n"

    # 發送 LINE
    if LINE_TOKEN and USER_ID:
        payload = {"to": USER_ID, "messages": [{"type": "text", "text": report}]}
        headers = {"Authorization": f"Bearer {LINE_TOKEN}", "Content-Type": "application/json"}
        requests.post("https://api.line.me/v2/bot/message/push", headers=headers, json=payload)
    print(report)

if __name__ == "__main__":
    main()
