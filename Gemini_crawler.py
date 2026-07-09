import os
import yfinance as yf
import pandas_ta as ta
import google.generativeai as genai
import requests
import sys
import time
from datetime import datetime

# 設定環境變數
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
LINE_TOKEN = os.environ.get("LINE_TOKEN")
USER_ID = os.environ.get("LINE_USER_ID")

def get_best_model():
    """【保留功能】自動偵測支援的模型，避免 404 錯誤"""
    genai.configure(api_key=GEMINI_API_KEY)
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods and 'flash' in m.name:
            return m.name
    return genai.list_models()[0].name

def get_full_analysis(ticker_symbol):
    """【保留功能】完整指標計算 (一年期)"""
    stock = yf.Ticker(ticker_symbol)
    df = stock.history(period="1y")
    if df.empty: return None, None, None, None, None

    latest_date = df.index[-1].strftime('%Y-%m-%d')
    latest = df.iloc[-1]
    price_info = f"收盤: {latest['Close']:.2f}, 最高: {latest['High']:.2f}, 最低: {latest['Low']:.2f}"
    
    df.ta.macd(append=True); df.ta.rsi(append=True); df.ta.bbands(append=True); df.ta.stoch(append=True)
    try:
        bbl = [c for c in df.columns if 'BBL' in c][0]
        bbu = [c for c in df.columns if 'BBU' in c][0]
        macd = [c for c in df.columns if 'MACD_' in c and 'MACDh' not in c and 'MACDs' not in c][0]
        sk = [c for c in df.columns if 'STOCHk' in c][0]
        sd = [c for c in df.columns if 'STOCHd' in c][0]
        rsi = [c for c in df.columns if 'RSI' in c][0]
        df['PCT_B'] = (df['Close'] - df[bbl]) / (df[bbu] - df[bbl])
        latest_vals = df.iloc[-1]
        tech_summary = f"RSI: {latest_vals[rsi]:.2f}, MACD: {latest_vals[macd]:.2f}, KD: {latest_vals[sk]:.2f}/{latest_vals[sd]:.2f}, %B: {latest_vals['PCT_B']:.2f}"
    except:
        tech_summary = "指標計算中"

    info = stock.info
    fund_summary = f"P/E: {info.get('trailingPE', 'N/A')}, P/B: {info.get('priceToBook', 'N/A')}"
    ref = f"資料來源: Yahoo Finance (截止: {latest_date})"
    
    return price_info, tech_summary, fund_summary, ref, latest_date

def main():
    if not GEMINI_API_KEY: sys.exit(1)
    
    # 【保留功能】自適應模型偵測
    model_name = get_best_model()
    model = genai.GenerativeModel(model_name)
    
    tickers = ["2330.TW", "0050.TW", "NVDA", "AMD", "MU"]
    report = f"📈 【Gemini AI 財務技術報告 Vesrsion 1.0.0】\n報告時間: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n引用模型: {model_name}\n"
    
    for t in tickers:
        try:
            price, tech, fund, ref, _ = get_full_analysis(t)
            if price:
                prompt = f"分析標的：{t}\n參考資料：{ref}\n今日行情：{price}\n技術面：{tech}\n基本面：{fund}\n請根據以上資訊給出建議。"
                
                # 【新增】防 429 錯誤的重試機制
                success = False
                for attempt in range(3):
                    try:
                        response = model.generate_content(prompt)
                        report += f"\n--- {t} ---\n【{ref}】\n【行情】{price}\n【指標】{tech}\n【基本】{fund}\n【AI建議】{response.text}\n"
                        success = True
                        break
                    except Exception as e:
                        if "429" in str(e):
                            time.sleep(60) # 等待 60 秒後重試
                        else:
                            raise e
                if not success: report += f"\n{t} 分析因頻率限制失敗\n"
                time.sleep(20) # 每個標的間隔 20 秒
        except Exception as e:
            report += f"\n{t} 分析失敗: {e}\n"

    # 發送邏輯
    if LINE_TOKEN and USER_ID:
        payload = {"to": USER_ID, "messages": [{"type": "text", "text": report[:4000]}]}
        headers = {"Authorization": f"Bearer {LINE_TOKEN}", "Content-Type": "application/json"}
        res = requests.post("https://api.line.me/v2/bot/message/push", headers=headers, json=payload)
        print(f"LINE 發送狀態: {res.status_code}")
    
    print(report)

if __name__ == "__main__":
    main()
