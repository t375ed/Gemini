import os
import json
from urllib.parse import quote_plus
import feedparser
import pandas as pd
from bs4 import BeautifulSoup
from google import genai
from google.genai import types
from linebot.v3.messaging import Configuration, ApiClient, MessagingApi, PushMessageRequest, TextMessage

# 取得環境變數（支援 LINE_TOKEN 與 LINE_CHANNEL_ACCESS_TOKEN 相容性）
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
LINE_TOKEN = os.getenv("LINE_TOKEN") or os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_USER_ID = os.getenv("LINE_USER_ID")

client = genai.Client(api_key=GEMINI_API_KEY)

def fetch_google_news(query: str, max_items: int = 10) -> pd.DataFrame:
    """從 Google News RSS 抓取新聞並清除 HTML 標籤。"""
    q = quote_plus(query)
    url = f"https://news.google.com/rss/search?q={q}&hl=zh-Hant&gl=TW&ceid=TW:zh-Hant"
    feed = feedparser.parse(url)
    
    rows = []
    for entry in feed.entries[:max_items]:
        raw_summary = entry.get("summary", "")
        clean_summary = BeautifulSoup(raw_summary, "html.parser").get_text() if raw_summary else ""
        rows.append({
            "title": entry.get("title", ""),
            "summary": clean_summary,
            "link": entry.get("link", "")
        })
    return pd.DataFrame(rows)

def analyze_investment_news(headlines: list[str], summaries: list[str]) -> list[dict]:
    """評估閱讀等級與情緒，並針對 4 星以上新聞進行簡短精準的 6 大投資問題解析。"""
    prompt = "請閱讀以下新聞，進行評分與 6 大投資問題解析：\n\n"
    for i, h in enumerate(headlines):
        prompt += f"新聞{i+1}:\n標題: {h}\n摘要: {summaries[i]}\n\n"

    prompt += """
【評分與回答規則】：
1. reading_grade: 1~5 數值
   - 1-3: 廢文/即時短訊/普通新聞
   - 4-5: 深度產業分析/法說會重大動態
2. sentiment_label: positive / neutral / negative
3. sentiment_score: 0.0~1.0
4. rationale: 1 句極簡理由。

5. investment_qa (6 大投資問題解析)：
   若 reading_grade 達 4 或 5 分，請「精簡條列、精準不誇大」回答下列 6 題，每題控制在 15 字以內。未提及則填「內文未提及」：
   - Q1_benefited_product: 受惠產品/製程
   - Q2_real_demand: 真實需求或想像
   - Q3_trend: 短期拉貨或長期趨勢
   - Q4_supply_demand: 供需狀況（缺貨/滿載）
   - Q5_price_hike: 漲價原因（需求/成本）
   - Q6_client_visibility: 客戶與訂單能見度

若 reading_grade 未達 4 分，investment_qa 各項填「未達標」。
    """

    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema={
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "reading_grade": {"type": "INTEGER"},
                    "sentiment_label": {"type": "STRING", "enum": ["positive", "neutral", "negative"]},
                    "sentiment_score": {"type": "NUMBER"},
                    "rationale": {"type": "STRING"},
                    "investment_qa": {
                        "type": "OBJECT",
                        "properties": {
                            "Q1_benefited_product": {"type": "STRING"},
                            "Q2_real_demand": {"type": "STRING"},
                            "Q3_trend": {"type": "STRING"},
                            "Q4_supply_demand": {"type": "STRING"},
                            "Q5_price_hike": {"type": "STRING"},
                            "Q6_client_visibility": {"type": "STRING"}
                        },
                        "required": [
                            "Q1_benefited_product", "Q2_real_demand", "Q3_trend",
                            "Q4_supply_demand", "Q5_price_hike", "Q6_client_visibility"
                        ]
                    }
                },
                "required": ["reading_grade", "sentiment_label", "sentiment_score", "rationale", "investment_qa"]
            }
        }
    )

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=config
    )
    return json.loads(response.text)

def send_line_push(message_text: str):
    """發送 LINE 推播訊息。"""
    configuration = Configuration(access_token=LINE_TOKEN)
    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)
        push_message_request = PushMessageRequest(
            to=LINE_USER_ID,
            messages=[TextMessage(text=message_text)]
        )
        line_bot_api.push_message(push_message_request)

def main():
    keyword = "台積電"
    df = fetch_google_news(keyword, max_items=10)
    
    if df.empty:
        print("未抓取到新聞")
        return

    results = analyze_investment_news(df["title"].tolist(), df["summary"].tolist())
    df_res = pd.DataFrame(results)
    df_final = pd.concat([df, df_res], axis=1)

    # 僅篩選閱讀等級 >= 4 星的高價值深度新聞
    high_grade_news = df_final[df_final["reading_grade"] >= 4]

    if not high_grade_news.empty:
        for _, row in high_grade_news.iterrows():
            sentiment_emoji = "🟢" if row["sentiment_label"] == "positive" else ("🔴" if row["sentiment_label"] == "negative" else "⚪")
            stars = "⭐" * int(row["reading_grade"])
            qa = row["investment_qa"]

            # 極簡條列式排版
            msg = f"📌 【{keyword} 深度報告】\n"
            msg += f"📰 {row['title']}\n\n"
            msg += f"評分：{stars} ({row['reading_grade']}/5)\n"
            msg += f"情緒：{sentiment_emoji} {row['sentiment_label']} ({row['sentiment_score']})\n"
            msg += f"摘要：{row['rationale']}\n"
            msg += "-----------------------------------\n"
            msg += "🎯 6 大基本面檢視：\n"
            msg += f"• 受惠產品：{qa['Q1_benefited_product']}\n"
            msg += f"• 真實需求：{qa['Q2_real_demand']}\n"
            msg += f"• 趨勢性質：{qa['Q3_trend']}\n"
            msg += f"• 供需狀況：{qa['Q4_supply_demand']}\n"
            msg += f"• 漲價動因：{qa['Q5_price_hike']}\n"
            msg += f"• 能 見 度：{qa['Q6_client_visibility']}\n"
            msg += "-----------------------------------\n"
            msg += f"🔗 原文網址：\n{row['link']}"

            # LINE 單則推播上限為 2000 字
            send_line_push(msg[:2000])
        print(f"已成功推送 {len(high_grade_news)} 則簡短條列式投資檢視報告！")
    else:
        print("今日新聞未達到 4 星以上深度分析標準，無須推送。")

if __name__ == "__main__":
    main()
