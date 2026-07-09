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
    genai.configure(api_key=GEMINI_API_KEY)
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods and 'flash' in m.name:
            return m.name
    return genai.list_models()[0].name

def get_technical_analysis(ticker_symbol):
    stock = yf.Ticker(ticker_symbol)
    df = stock.history(period="5d") # 獲取近期數據
    if df.empty: return "資料讀取失敗", "N/A", "N/A"

    # 獲取最新交易日 (這就是你的確認手段)
    latest_date = df.index[-1].strftime('%Y-%m-%d')
    latest = df.iloc[-1]
    
    # 資料確認資訊
    ref_info = f"數據來源: Yahoo Finance | 統計日期: {latest_date}"
    price_info = f"收盤: {latest['Close']:.2f}, 最高: {latest['High']:.2f}, 最低: {latest['Low']:.2f}"

    # 計算技術指標
    df.ta.macd(append=True); df.ta.rsi(append=True)
    try:
        rsi = [c for c in df.columns if 'RSI' in c][0]
        tech_summary = f"RSI: {latest[rsi]:.2f}"
    except:
        tech_summary = "指標計算中"
    
    return ref_info, price_info, tech_summary

def main():
    if not GEMINI_API_KEY: sys.exit(1)
    
    model = genai.GenerativeModel(get_best_model())
    tickers = ["2330.TW", "0050.TW", "NVDA", "AMD", "MU"]
    # 加入程式執行時間確認
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M')
    report = f"📈 【AI 深度技術面報告】\n報告產生時間: {current_time}\n"
    
    for t in tickers:
        try:
            ref_info, price_info, tech = get_technical_analysis(t)
            # 將資料來源與即時股價放入 Prompt
            prompt = f"分析標的：{t}\n參考資料：{ref_info}\n今日行情：{price_info}\n技術面：{tech}\n請根據這些最新數據提供買賣建議。"
            response = model.generate_content(prompt)
            report += f"\n--- {t} ---\n{ref_info}\n今日行情: {price_info}\n建議：{response.text}\n"
            time.sleep(15) 
        except Exception as e:
            report += f"\n{t} 分析失敗: {e}\n"

    # 發送邏輯
    if LINE_TOKEN and USER_ID:
        final_report = report[:4000]
        url = "https://api.line.me/v2/bot/message/push"
        headers = {"Authorization": f"Bearer {LINE_TOKEN}", "Content-Type": "application/json"}
        payload = {"to": USER_ID, "messages": [{"type": "text", "text": final_report}]}
        
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            print("✅ LINE 訊息發送成功！")
        else:
            print(f"❌ LINE 發送失敗！狀態碼: {response.status_code}\n內容: {response.text}")
    
    print(report)

if __name__ == "__main__":
    main()
