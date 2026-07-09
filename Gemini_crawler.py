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
    """自動偵測目前 API 金鑰支援的正確模型名稱"""
    genai.configure(api_key=GEMINI_API_KEY)
    
    # 搜尋所有支援 generateContent 的模型
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            # 優先使用 flash 版本，通常支援度最廣且符合免費額度限制
            if 'flash' in m.name:
                return m.name
    
    # 若找不到 flash，回傳第一個找到的可用模型
    return genai.list_models()[0].name

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
    
    cols = df.columns
    try:
        # 動態識別欄位名稱
        bbl_col = [c for c in cols if 'BBL' in c][0]
        bbu_col = [c for c in cols if 'BBU' in c][0]
        macd_col = [c for c in cols if 'MACD_' in c and 'MACDh' not in c and 'MACDs' not in c][0]
        stoch_k = [c for c in cols if 'STOCHk' in c][0]
        stoch_d = [c for c in cols if 'STOCHd' in c][0]
        rsi_col = [c for c in cols if 'RSI' in c][0]
        
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
        print("未設定 GEMINI_API_KEY")
        sys.exit(1)
        
    model_name = get_best_model()
    print(f"自動偵測並使用模型: {model_name}")
    model = genai.GenerativeModel(model_name)
    
    tickers = ["2330.TW", "0050.TW", "NVDA", "AMD", "MU"]
    report = "📈 【AI 深度技術面波段報告】\n"
    
    for t in tickers:
        try:
            tech, fund = get_technical_analysis(t)
            if tech:
                prompt = f"分析標的：{t}\n基本面：{fund}\n技術面指標：{tech}\n請提供波段買賣建議。"
                response = model.generate_content(prompt)
                report += f"\n--- {t} ---\n{response.text}\n"
                
                # 強制暫停 15 秒，避免 429 配額錯誤
                time.sleep(15)
        except Exception as e:
            report += f"\n{t} 分析失敗: {e}\n"
            time.sleep(5)

    if LINE_TOKEN and USER_ID:
        url = "https://api.line.me/v2/bot/message/push"
        headers = {"Authorization": f"Bearer {LINE_TOKEN}", "Content-Type": "application/json"}
        payload = {"to": USER_ID, "messages": [{"type": "text", "text": report}]}
        requests.post(url, headers=headers, json=payload)
    
    print(report)

if __name__ == "__main__":
    main()
