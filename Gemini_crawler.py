import os
import requests
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import google.generativeai as genai
import sys
import time
from datetime import datetime, timedelta
import twstock

# 設定環境變數
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
LINE_TOKEN = os.environ.get("LINE_TOKEN")
USER_ID = os.environ.get("LINE_USER_ID")

def get_best_model():
    """自動偵測支援的模型，避免 404 錯誤"""
    genai.configure(api_key=GEMINI_API_KEY)
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods and 'flash' in m.name:
            return m.name
    return genai.list_models()[0].name

def fetch_stock_df(ticker_symbol):
    """
    根據標的自動切換抓取來源：
    - 台股 (.TW / .TWO) 使用 twstock
    - 美股 使用 yfinance
    """
    try:
        if ".TW" in ticker_symbol or ".TWO" in ticker_symbol:
            code = ticker_symbol.replace(".TW", "").replace(".TWO", "")
            stock = twstock.Stock(code)
            # 抓取最近的歷史資料
            raw_data = stock.fetch_31()
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
    
    if df is None or df.empty or len(df) < 10: 
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
        tech_summary = "指標計算中"

    # 基本面數據 (美股用 yfinance，台股顯示基礎資訊)
    if ".TW" in ticker_symbol or ".TWO" in ticker_symbol:
        fund_summary = "P/E: 參考公開資訊觀測站"
    else:
        try:
            info = yf.Ticker(ticker_symbol).info
            fund_summary = f"P/E: {info.get('trailingPE', 'N/A')}, P/B: {info.get('priceToBook', 'N/A')}"
        except Exception:
            fund_summary = "P/E: N/A, P/B: N/A"
            
    return price_info, tech_summary, fund_summary, ref

def main():
    if not GEMINI_API_KEY: 
        print("未設定 GEMINI_API_KEY")
        sys.exit(1)
    
    model_name = get_best_model()
    model = genai.GenerativeModel(model_name)
    
    tickers = ["2330.TW", "0050.TW", "NVDA", "AMD", "MU"]
    report = f"📈 【Gemini AI 財務技術報告 Version 1.0.2】\n報告時間: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n引用模型: {model_name}\n"
    
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

請扮演專業投資分析師，針對上述數據給出分析建議並針對所有資訊都再次搜尋交叉驗證其真實性，每次準備買進一家公司前，都會先問自己以下幾個問題：
1. 公司真正受惠的是哪項產品？2. 這項產品有真實需求，還是市場想像？
3. 需求是短期拉貨還是長期趨勢？
4. 供需是否失衡，有無缺貨或產能滿載？
5. 漲價是需求推動還是成本轉嫁？
6. 主要客戶是誰？訂單能見度有多高？請客觀、不預設立場、交叉查證。
【硬性要求】：
1. 分析內容請維持重點精煉，字數嚴格限制在 300 字以內。
2. 切勿包含任何開場白或問候語，直接輸出重點結論與操作建議。
"""
                
                success = False
                for attempt in range(3):
                    try:
                        response = model.generate_content(prompt)
                        report += f"\n--- {t} ---\n【行情】{price}\n【指標】{tech}\n【基本】{fund}\n【AI建議】\n{response.text.strip()}\n"
                        success = True
                        break
                    except Exception as e:
                        if "429" in str(e):
                            time.sleep(60)
                        else:
                            raise e
                if not success: report += f"\n{t} 分析因頻率限制失敗\n"
                time.sleep(15)
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
