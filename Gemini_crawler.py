import os
import yfinance as yf
import google.generativeai as genai
import requests
import sys

# 讀取環境變數
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
LINE_TOKEN = os.environ.get("LINE_TOKEN")
USER_ID = os.environ.get("LINE_USER_ID")

# 檢查關鍵金鑰是否存在
if not GEMINI_API_KEY:
    print("錯誤: 未設定 GEMINI_API_KEY")
    sys.exit(1)

# 初始化 Gemini
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

def get_stock_data(ticker_symbol):
    stock = yf.Ticker(ticker_symbol)
    # 確保抓取到數據，若無數據則拋出異常
    hist = stock.history(period="1mo")
    if hist.empty:
        raise ValueError(f"無法取得 {ticker_symbol} 的股價數據")
    return hist.tail(5).to_string()

def analyze_with_gemini(ticker, data):
    prompt = f"你是一位財經專家。以下是 {ticker} 近期的股價數據：\n{data}\n請針對這些數據判斷目前的波段趨勢與技術面重點，以簡短專業的口吻提供建議。"
    response = model.generate_content(prompt)
    return response.text

def send_line_message(message):
    if not LINE_TOKEN or not USER_ID:
        print("警告: LINE_TOKEN 或 USER_ID 未設定，無法推播。")
        print(f"分析結果: {message}")
        return

    url = "https://api.line.me/v2/bot/message/push"
    headers = {
        "Authorization": f"Bearer {LINE_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "to": USER_ID,
        "messages": [{"type": "text", "text": message}]
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        print("LINE 推播成功！")
    else:
        print(f"LINE 推播失敗，狀態碼: {response.status_code}, 內容: {response.text}")

if __name__ == "__main__":
    ticker = "2330.TW" 
    try:
        print(f"正在開始分析 {ticker}...")
        raw_data = get_stock_data(ticker)
        analysis = analyze_with_gemini(ticker, raw_data)
        send_line_message(f"【AI 財經分析報告 - {ticker}】\n{analysis}")
    except Exception as e:
        print(f"程式執行發生錯誤: {e}")
        sys.exit(1)
