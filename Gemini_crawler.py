import os
import yfinance as yf
import google.generativeai as genai
import requests

# 設定參數
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
LINE_TOKEN = os.environ.get("LINE_TOKEN")
USER_ID = os.environ.get("LINE_USER_ID")
MODEL_VERSION = 'gemini-3.5-flash'

# 初始化 Gemini
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(MODEL_VERSION)

def get_stock_data(tickers):
    """取得多個股票的近期歷史數據"""
    data_list = []
    # 使用 yf.Tickers 批次處理
    tickers_data = yf.Tickers(" ".join(tickers))
    for ticker in tickers:
        hist = tickers_data.tickers[ticker].history(period="1mo")
        if not hist.empty:
            summary = hist.tail(3).to_string()
            data_list.append(f"【{ticker}】\n{summary}")
    return "\n\n".join(data_list)

def main():
    tw_tickers = ["2330.TW", "0050.TW"]
    us_tickers = ["NVDA", "AMD", "MU"]
    
    # 獲取數據
    tw_data = get_stock_data(tw_tickers)
    us_data = get_stock_data(us_tickers)
    
    # AI 分析
    prompt = f"""
    你是一位專業財經分析師。請分別針對以下兩組市場數據提供簡短、專業的波段趨勢建議：
    
    【台股】
    {tw_data}
    
    【美股】
    {us_data}
    
    請注意：
    1. 使用條列式重點排版。
    2. 重要數據請加粗。
    3. 內容需精簡，適合手機小螢幕閱讀。
    """
    
    response = model.generate_content(prompt)
    analysis = response.text
    
    # 訊息排版
    final_message = (
        f"📈 【市場投資分析報告】\n"
        f"AI 模型：{MODEL_VERSION}\n"
        f"--------------------------\n"
        f"{analysis}\n"
        f"--------------------------\n"
        f"⚠️ 投資有風險，請審慎評估。"
    )
    
    # 發送至 LINE
    url = "https://api.line.me/v2/bot/message/push"
    headers = {"Authorization": f"Bearer {LINE_TOKEN}", "Content-Type": "application/json"}
    payload = {"to": USER_ID, "messages": [{"type": "text", "text": final_message}]}
    requests.post(url, headers=headers, json=payload)

if __name__ == "__main__":
    main()
