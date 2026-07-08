import os
import yfinance as yf
import google.generativeai as genai
import requests

# 1. 改為從環境變數讀取 (這能確保 GitHub Actions 與本地電腦都能共用此邏輯)
# os.environ.get 會優先尋找 GitHub Secrets 中設定的變數
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
LINE_TOKEN = os.environ.get("LINE_TOKEN")
USER_ID = os.environ.get("LINE_USER_ID")

# 初始化 Gemini
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

def get_stock_data(ticker_symbol):
    stock = yf.Ticker(ticker_symbol)
    hist = stock.history(period="1mo")
    data_summary = hist.tail(5).to_string()
    return data_summary

def analyze_with_gemini(data):
    # 設定專業指令 (System Instruction 概念)
    prompt = f"你是一位財經專家。以下是 {ticker} 近期的股價數據：\n{data}\n請針對這些數據判斷目前的波段趨勢與基本面重點，以簡短專業的口吻提供建議。"
    response = model.generate_content(prompt)
    return response.text

def send_line_message(message):
    url = "https://api.line.me/v2/bot/message/push"
    headers = {
        "Authorization": f"Bearer {LINE_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "to": USER_ID,
        "messages": [{"type": "text", "text": message}]
    }
    # 若環境變數未設定 (例如在測試時)，則不發送避免錯誤
    if LINE_TOKEN and USER_ID:
        requests.post(url, headers=headers, json=payload)
    else:
        print("LINE TOKEN 或 USER_ID 未設定，跳過發送：", message)

if __name__ == "__main__":
    ticker = "2330.TW" # 你可以自行修改目標股票
    try:
        raw_data = get_stock_data(ticker)
        analysis = analyze_with_gemini(raw_data)
        send_line_message(f"【AI 財經分析報告 - {ticker}】\n{analysis}")
    except Exception as e:
        print(f"程式執行錯誤: {e}")
