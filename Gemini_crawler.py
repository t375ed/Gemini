import os
import yfinance as yf
import pandas_ta as ta
import google.generativeai as genai
import requests
import sys

# 設定環境變數
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
LINE_TOKEN = os.environ.get("LINE_TOKEN")
USER_ID = os.environ.get("LINE_USER_ID")

# 改用穩定版本名稱，解決 404 Not Found 問題
MODEL_VERSION = 'gemini-pro-latest'

def get_technical_analysis(ticker_symbol):
    stock = yf.Ticker(ticker_symbol)
    df = stock.history(period="1y")
    
    if df.empty:
        return None, None

    # 計算技術指標
    df.ta.macd(append=True)
    df.ta.rsi(append=True)
    df.ta.bbands(append=True)
    df.ta.stoch(append=True)
    
    # 動態識別欄位
    cols = df.columns
    try:
        bbl_col = [c for c in cols if 'BBL' in c][0]
        bbu_col = [c for c in cols if 'BBU' in c][0]
        macd_col = [c for c in cols if 'MACD_' in c and 'MACDh' not in c and 'MACDs' not in c][0]
        stoch_k = [c for c in cols if 'STOCHk' in c][0]
        stoch_d = [c for c in cols if 'STOCHd' in c][0]
        rsi_col = [c for c in cols if 'RSI' in c][0]
        
        # 計算 %B
        df['PCT_B'] = (df['Close'] - df[bbl_col]) / (df[bbu_col] - df[bbl_col])
        latest = df.iloc[-1]
        
        tech_summary = (
            f"RSI: {latest[rsi_col]:.2f}, "
            f"MACD: {latest[macd_col]:.2f}, "
            f"KD(K/D): {latest[stoch_k]:.2f}/{latest[stoch_d]:.2f}, "
            f"%B: {latest['PCT_B']:.2f}"
        )
    except Exception as e:
        return f"指標計算錯誤: {e}", "N/A"
    
    info = stock.info
    fundamentals = f"P/E: {info.get('trailingPE', 'N/A')}, P/B: {info.get('priceToBook', 'N/A')}, ROE: {info.get('returnOnEquity', 'N/A')}"
    
    return tech_summary, fundamentals

def main():
    if not GEMINI_API_KEY:
        sys.exit(1)
        
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(MODEL_VERSION)
    
    tickers = ["2330.TW", "0050.TW", "NVDA", "AMD", "MU"]
    report = "📈 【AI 深度技術面波段報告】\n"
    
    for t in tickers:
        try:
            tech, fund = get_technical_analysis(t)
            if tech:
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
        url = "https://api.line.me/v2/bot/message/push"
        headers = {"Authorization": f"Bearer {LINE_TOKEN}", "Content-Type": "application/json"}
        payload = {"to": USER_ID, "messages": [{"type": "text", "text": report}]}
        requests.post(url, headers=headers, json=payload)
    print(report)

if __name__ == "__main__":
    main()
