import os
import yfinance as yf
import google.generativeai as genai
import requests
import sys

def main():
    # 讀取環境變數
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
    LINE_TOKEN = os.environ.get("LINE_TOKEN")
    USER_ID = os.environ.get("LINE_USER_ID")
    
    # 【新增】明確標示使用的模型版本
    MODEL_VERSION = 'gemini-3.5-flash'

    if not GEMINI_API_KEY:
        print("錯誤: 未設定 GEMINI_API_KEY")
        sys.exit(1)

    # 初始化 Gemini
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(MODEL_VERSION)

    ticker = "2330.TW"
    
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1mo")
        if hist.empty:
            sys.exit(1)
        data_summary = hist.tail(5).to_string()
        
        # 【修改】提示詞，要求 AI 輸出適合手機閱讀的格式
        prompt = f"""
        你是一位財經專家。以下是 {ticker} 近期股價數據：
        {data_summary}
        請提供簡短專業的波段趨勢建議。
        要求：
        1. 使用條列式重點。
        2. 重要數據請加粗。
        3. 內容保持精簡，適合手機小螢幕閱讀。
        """
        response = model.generate_content(prompt)
        analysis = response.text
        
        # 【修改】排版訊息，加入模型版本資訊
        final_message = (
            f"📈 【台股分析報告】\n"
            f"目標標的：{ticker}\n"
            f"AI 模型：{MODEL_VERSION}\n"
            f"--------------------------\n"
            f"{analysis}\n"
            f"--------------------------\n"
            f"⚠️ 投資請審慎評估風險"
        )
        
        # LINE 推播
        if LINE_TOKEN and USER_ID:
            url = "https://api.line.me/v2/bot/message/push"
            headers = {"Authorization": f"Bearer {LINE_TOKEN}", "Content-Type": "application/json"}
            payload = {"to": USER_ID, "messages": [{"type": "text", "text": final_message}]}
            requests.post(url, headers=headers, json=payload)
            print("訊息已發送")
        else:
            print(final_message)
            
    except Exception as e:
        print(f"執行錯誤: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
