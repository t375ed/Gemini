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
    完善的備用模型降級與退避重試機制
    優先使用 gemini-2.5-flash，繁忙時重試，失敗則自動切換至備用模型 gemini-2.5-pro
    回傳值：(分析文字結果, 實際使用的模型 VERSION)
    """
    # 定義主模型與備用模型清單 (均為新版 SDK 相容之名稱)
    PRIMARY_MODEL = "gemini-2.5-flash"
    BACKUP_MODEL = "gemini-2.5-pro"
    
    models_queue = [PRIMARY_MODEL, BACKUP_MODEL]
    last_error = ""

    for model_name in models_queue:
        # 對每個模型最多嘗試 3 次 (因應 429 / 503 流量管制)
        for attempt in range(3):
            try:
                print(f"嘗試使用模型 [{model_name}] 進行分析 (第 {attempt + 1} 次)...")
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )
                if response.text:
                    # 成功產出，回傳結果與使用的模型名稱
                    is_backup = (model_name != PRIMARY_MODEL)
                    version_tag = f"{model_name} [備用模型]" if is_backup else model_name
                    return response.text.strip(), version_tag
            except Exception as e:
                err_msg = str(e)
                last_error = err_msg
                print(f"[{model_name}] 失敗: {err_msg[:100]}")
                
                # 如果是 404 (模型不存在)，直接跳出重試，切換下一個模型
                if "404" in err_msg or "NOT_FOUND" in err_msg:
                    print(f"[{model_name}] 無法找到此模型，準備切換至備用模型...")
                    break
                
                # 如果是 503 / 429 流量問題，等待後重試
                if "503" in err_msg or "429" in err_msg or "UNAVAILABLE" in err_msg:
                    sleep_sec = (attempt + 1) * 4
                    print(f"[{model_name}] 伺服器繁忙，等待 {sleep_sec} 秒後重試...")
                    time.sleep(sleep_sec)
                else:
                    # 其他未預期錯誤，直接切換備用模型
                    break

    return f"AI 分析失敗 (原因: {last_error[:100]})", "失敗"

def main():
    if not GEMINI_API_KEY: 
        print("未設定 GEMINI_API_KEY")
        sys.exit(1)
    
    client = genai.Client(api_key=GEMINI_API_KEY)
    
    tickers = ["2330.TW", "0050.TW", "NVDA", "AMD", "MU"]
    report = f"📈 【Gemini AI 財務技術報告 Version 1.0.8】\n報告時間: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
    
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

請扮演客觀且嚴謹的資深投資分析師，結合技術面指標與基本面資料給出投資建議。
【硬性要求】：
1. 分析內容請維持重點精煉，字數嚴格限制在 300 字以內。
2. 切勿包含任何開場白或問候語，直接輸出重點結論與操作建議。
"""
                ai_advice, used_version = generate_ai_analysis(client, prompt)
                report += f"\n--- {t} ---\n【行情】{price}\n【指標】{tech}\n【基本】{fund}\n【AI建議 ({used_version})】\n{ai_advice}\n"
                
                # 每個標的隔 6 秒，降低觸發 API 流量上限的機率
                time.sleep(6)
            else:
                report += f"\n--- {t} ---\n⚠️ 無法取得該標的之有效行情資料\n"
        except Exception as e:
            report += f"\n{t} 分析失敗: {e}\n"

    # 安全裁切，確保最終 LINE 訊息不超過官方上限 4,000 字
    final_report = report[:3500] + "\n...(報告已截斷)" if len(report) > 3500 else report

    # LINE 發送邏輯
    if LINE_TOKEN and USER_ID:
        payload = {"to": USER_ID, "messages": [{"type": "text", "text": final_report}]}
        headers = {"Authorization": f"Bearer {LINE_TOKEN}", "Content-Type": "application/json"}
        res = requests.post("https://api.line.me/v2/bot/message/push", headers=headers, json=payload)
        print(f"LINE 發送狀態: {res.status_code}")
    
    print(final_report)

if __name__ == "__main__":
    main()
if __name__ == "__main__":
    main()
