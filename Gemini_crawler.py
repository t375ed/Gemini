import os
import yfinance as yf
import pandas_ta as ta
import google.generativeai as genai
import requests
import sys
import time

# 設定環境變數
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
LINE_TOKEN = os.environ.get("LINE_TOKEN")
USER_ID = os.environ.get("LINE_USER_ID")

def get_best_model():
    genai.configure(api_key=GEMINI_API_KEY)
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            if 'flash' in m.name:
                return m.name
    return genai.list_models()[0].name

def get_technical_analysis(ticker_symbol):
    stock = yf.Ticker(ticker_symbol)
    df = stock.history(period="1y")
    if df.empty: return None, None

    df.ta.macd(append=True); df.ta.rsi(append=True); df.ta.bbands(append=True); df.ta.stoch(append=True)
    
    cols = df.columns
    try:
        bbl = [c for c in cols if 'BBL' in c][0]
        bbu = [c for c in cols if 'BBU' in c][0]
        macd = [c for c in cols if 'MACD_' in c and 'MACDh' not in c and 'MACDs' not in c][0]
        sk = [c for c in cols if 'STOCHk' in c][0]
        sd = [c for c in cols if 'STOCHd' in c][0]
        rsi = [c for c in cols if 'RSI' in c][0]
        df['PCT_B'] = (df['Close'] - df[bbl]) / (df[bbu] - df[bbl])
        latest = df.iloc[-1]
        tech_summary = f"RSI: {latest[rsi]:.2f}, MACD: {latest[macd]:.2f}, KD: {latest[sk]:.2f}/{latest[sd]:.2f}, %B: {latest['PCT_B']:.2f}"
    except Exception as e:
        return f"指標錯誤: {e}", "N/A"
    
    info = stock.info
    fund = f"P/E: {info.get('trailingPE', 'N/A')}, ROE: {info.get('returnOnEquity', 'N/A')}"
    return tech_summary, fund

def main():
    if not GEMINI_API_KEY: sys.exit(1)
    
    model = genai.GenerativeModel(get_best_model())
    tickers = ["2330.TW", "0050.TW", "NVDA", "AMD", "MU"]
    report = "📈 【AI 深度技術面波段報告】\n"
    
    for t in tickers:
        try:
            tech, fund = get_technical_analysis(t)
            if tech:
                response = model.generate_content(f"分析標的：{t}\n基本面：{fund}\n技術面：{tech}\n請提供建議。")
                report += f"\n--- {t} ---\n{response.text}\n"
                time.sleep(15) 
        except Exception as e:
            report += f"\n{t} 分析失敗: {e}\n"

    # 發送邏輯整合（包含錯誤調試與長度限制）
    if LINE_TOKEN and USER_ID:
        # LINE 訊息單次限制約 5000 字，這裡截斷到 4000 確保安全
        final_report = report[:4000]
        url = "https://api.line.me/v2/bot/message/push"
        headers = {"Authorization": f"Bearer {LINE_TOKEN}", "Content-Type": "application/json"}
        payload = {"to": USER_ID, "messages": [{"type": "text", "text": final_report}]}
        
        response = requests.post(url, headers=headers, json=payload)
        
        if response.status_code == 200:
            print("✅ LINE 訊息發送成功！")
        else:
            print(f"❌ LINE 發送失敗！狀態碼: {response.status_code}")
            print(f"回應內容: {response.text}") # 這裡會印出為何失敗的詳細原因
    else:
        print("⚠️ 未設定 LINE_TOKEN 或 LINE_USER_ID")

    print(report)

if __name__ == "__main__":
    main()
