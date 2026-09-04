import os
import sys
import time
from datetime import datetime, timedelta
import requests
import pandas as pd
import pandas_ta as ta
import twstock
import yfinance as yf
from google import genai

# 設定環境變數
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
LINE_TOKEN = os.environ.get("LINE_TOKEN")
USER_ID = os.environ.get("LINE_USER_ID")

def fetch_stock_df(ticker_symbol):
    """
    根據標的自動切換抓取來源：
    - 台股 (.TW / .TWO) 使用 twstock (抓取近 60 天資料以足夠計算技術指標)
    - 美股 使用 yfinance
    """
    try:
        if ".TW" in ticker_symbol or ".TWO" in ticker_symbol:
            code = ticker_symbol.replace(".TW", "").replace(".TWO", "")
            stock = twstock.Stock(code)
            
            # 計算約兩個月前（60 天）的年份與月份，確保有足夠 K 線計算 MACD/KD
            start_date = datetime.now() - timedelta(days=60)
            raw_data = stock.fetch_from(start_date.year, start_date.month)
            if not raw_data:
                return None, None
            
            df = pd.DataFrame(raw_data)
            df.rename(columns={
                'date': 'Date', 'open': 'Open', 'high': 'High',
                'low': 'Low', 'close': 'Close', 'capacity': 'Volume'
            }, inplace=True)
            df.set_index('Date', inplace=True)
            latest_date = df.index[-1].strftime('%Y-%m-%d')
            ref = f"資料來源: 臺灣證券交易所/櫃買中心 (截止: {latest_date})"
            return df, ref
        else:
            stock = yf.Ticker(ticker_symbol)
            df = stock.history(period="1y")
            if df.empty or df['Close'].isnull().all():
                return None, None
            latest_date = df.index[-1].strftime('%Y-%m-%d')
            ref = f"資料來源: Yahoo Finance (截止: {latest_date})"
            return df, ref
    except Exception as e:
        print(f"抓取 {ticker_symbol} 失敗: {e}")
        return None, None

def get_full_analysis(ticker_symbol):
    """完整指標計算與基本面整理"""
    df, ref = fetch_stock_df(ticker_symbol)
    
    if df is None or df.empty or len(df) < 15: 
        return None, None, None, None

    latest = df.iloc[-1]
    close_val = latest['Close']
    high_val = latest['High']
    low_val = latest['Low']
    
    price_info = f"收盤: {close_val:.2f}, 最高: {high_val:.2f}, 最低: {low_val:.2f}"
    
    # 計算技術指標
    try:
        df.ta.macd(append=True)
        df.ta.rsi(append=True)
        df.ta.bbands(append=True)
        df.ta.stoch(append=True)
        
        bbl = [c for c in df.columns if 'BBL' in c][0]
        bbu = [c for c in df.columns if 'BBU' in c][0]
        macd = [c for c in df.columns if 'MACD_' in c and 'MACDh' not in c and 'MACDs' not in c][0]
        sk = [c for c in df.columns if 'STOCHk' in c][0]
        sd = [c for c in df.columns if 'STOCHd' in c][0]
        rsi = [c for c in df.columns if 'RSI' in c][0]
        
        df['PCT_B'] = (df['Close'] - df[bbl]) / (df[bbu] - df[bbl])
        latest_vals = df.iloc[-1]
        
        tech_summary = f"RSI: {latest_vals.get(rsi, 0):.2f}, MACD: {latest_vals.get(macd, 0):.2f}, KD: {latest_vals.get(sk, 0):.2f}/{latest_vals.get(sd, 0):.2f}, %B: {latest_vals.get('PCT_B', 0):.2f}"
    except Exception as e:
        print(f"{ticker_symbol} 技術指標計算錯誤: {e}")
        tech_summary = "指標計算中"

    # 基本面數據
    if ".TW" in ticker_symbol or ".TWO" in ticker_symbol:
        fund_summary = "P/E: 參考公開資訊觀測站"
    else:
        try:
            info = yf.Ticker(ticker_symbol).info
            fund_summary = f"P/E: {info.get('trailingPE', 'N/A')}, P/B: {info.get('priceToBook', 'N/A')}"
        except Exception:
            fund_summary = "P/E: N/A, P/B: N/A"
            
    return price_info, tech_summary, fund_summary, ref

def generate_ai_analysis(client, prompt):
    """
    具備退避重試機制與備用模型降級的 Gemini 呼叫函式
    使用官方相容性最佳的模型名稱清單
    """
    models_to_try = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]
    
    for model_name in models_to_try:
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )
                return response.text.strip(), model_name
            except Exception as e:
                err_msg = str(e)
                # 遇到 503 (UNAVAILABLE) 或 429 (限流) 進行等待重試
                if "503" in err_msg or "UNAVAILABLE" in err_msg or "429" in err_msg:
                    wait_time = (attempt + 1) * 5
                    print(f"[{model_name}] 伺服器繁忙 (嘗試 {attempt + 1}/3)，等待 {wait_time} 秒後重試...")
                    time.sleep(wait_time)
                elif "404" in err_msg or "NOT_FOUND" in err_msg:
                    print(f"[{model_name}] 模型不可用，自動切換至下一備用模型...")
                    break  # 切換到下一個 model_name
                else:
                    print(f"[{model_name}] 發生未預期錯誤: {e}")
                    break
                    
    return "AI 分析因伺服器繁忙或模型無法調用而失敗", "失敗"

def main():
    if not GEMINI_API_KEY: 
        print("未設定 GEMINI_API_KEY")
        sys.exit(1)
    
    client = genai.Client(api_key=GEMINI_API_KEY)
    
    tickers = ["2330.TW", "0050.TW", "NVDA", "AMD", "MU"]
    report = f"📈 【Gemini AI 財務技術報告 Version 1.0.6】\n報告時間: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
    
    for t in tickers:
        try:
            price, tech, fund, ref = get_full_analysis(t)
            if price:
                prompt = f"""
分析標的：{t}
參考資料：{ref}
今日行情：{price}
技術面：{tech}
基本面：{fund}

請扮演專業投資分析師，針對上述數據給出分析建議。
【硬性要求】：
1. 分析內容請維持重點精煉，字數嚴格限制在 300 字以內。
2. 切勿包含任何開場白或問候語，直接輸出重點結論與操作建議。
"""
                ai_advice, used_model = generate_ai_analysis(client, prompt)
                report += f"\n--- {t} ---\n【行情】{price}\n【指標】{tech}\n【基本】{fund}\n【AI建議 ({used_model})】\n{ai_advice}\n"
                
                time.sleep(5)
            else:
                report += f"\n--- {t} ---\n⚠️ 無法取得該標的之有效行情資料\n"
        except Exception as e:
            report += f"\n{t} 分析失敗: {e}\n"

    # 安全裁切，確保最終 LINE 訊息不超過官方上限 4,000 字
    final_report = report[:3500] + "\n...(報告已截斷)" if len(report) > 3500 else report

    # 發送邏輯
    if LINE_TOKEN and USER_ID:
        payload = {"to": USER_ID, "messages": [{"type": "text", "text": final_report}]}
        headers = {"Authorization": f"Bearer {LINE_TOKEN}", "Content-Type": "application/json"}
        res = requests.post("https://api.line.me/v2/bot/message/push", headers=headers, json=payload)
        print(f"LINE 發送狀態: {res.status_code}")
    
    print(final_report)

if __name__ == "__main__":
    main()
