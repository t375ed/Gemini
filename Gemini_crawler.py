import os
import yfinance as yf
import google.generativeai as genai
import requests
import sys

def main():
    # 讀取環境變數並進行偵測
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
    LINE_TOKEN = os.environ.get("LINE_TOKEN")
    USER_ID = os.environ.get("LINE_USER_ID")

    print(f"DEBUG: GEMINI_API_KEY 是否存在: {bool(GEMINI_API_KEY)}")
    print(f"DEBUG: LINE_TOKEN 是否存在: {bool(LINE_TOKEN)}")

    if not GEMINI_API_KEY:
        print("錯誤: 未設定 GEMINI_API_KEY")
        sys.exit(1)

    # 初始化 Gemini
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-3.1-pro-preview')

    ticker = "2330.TW"
    
    # 抓取資料
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1mo")
        if hist.empty:
            print(f"錯誤: 無法取得 {ticker} 資料")
            sys.exit(1)
        data_summary = hist.tail(5).to_string()
        
        # AI 分析
        prompt = f"你是一位財經專家。以下是 {ticker} 近期股價：\n{data_summary}\n請提供簡短專業的波段趨勢建議。"
        response = model.generate_content(prompt)
        analysis = response.text
        
        # LINE 推播
        if LINE_TOKEN and USER_ID:
            url = "https://api.line.me/v2/bot/message/push"
            headers = {"Authorization": f"Bearer {LINE_TOKEN}", "Content-Type": "application/json"}
            payload = {"to": USER_ID, "messages": [{"type": "text", "text": f"【AI 分析】\n{analysis}"}]}
            requests.post(url, headers=headers, json=payload)
            print("訊息已發送")
        else:
            print("未設定 LINE 變數，跳過發送")
            print(f"分析內容: {analysis}")
            
    except Exception as e:
        print(f"執行錯誤: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
