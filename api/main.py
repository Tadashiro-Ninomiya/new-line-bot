import logging
import os

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Request
from linebot.v3 import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.messaging import (
    ApiClient,
    Configuration,
    MessagingApi,
    ReplyMessageRequest,
    ShowLoadingAnimationRequest,
    TextMessage,
)
from linebot.v3.webhooks import MessageEvent, TextMessageContent, StickerMessageContent

from utils.config import logger
from utils.chat import generate_chat_response
from common.generation import chat
from utils.cosmos import save_chat_message, fetch_recent_chat_messages

load_dotenv()

# アプリの設定
configuration = Configuration(access_token=os.environ.get("LINE_CHANNEL_ACCESS_TOKEN"))
handler = WebhookHandler(os.environ.get("LINE_CHANNEL_SECRET"))
chatId = os.environ.get("LINE_USER_ID")

app = FastAPI(
    title="LINEBOT-AI-AGENT",
    description="LINEBOT-AI-AGENT by FastAPI.",
)


@app.post("/callback")
async def callback(
    request: Request,
    background_tasks: BackgroundTasks,
    x_line_signature=Header(None),
):
    body = await request.body()
    logger.info(f"受信したリクエストボディ: {body.decode('utf-8')}")  # loggerを使用してログ出力

    try:
        background_tasks.add_task(handler.handle, body.decode("utf-8"), x_line_signature)
        logger.info("バックグラウンドタスクにハンドラを追加しました。")  # loggerを使用してログ出力
    except InvalidSignatureError:
        logger.error("無効な署名が検出されました。")  # loggerを使用してログ出力
        raise HTTPException(status_code=400, detail="Invalid signature")

    logger.info("リクエスト処理が正常に完了しました。")  # loggerを使用してログ出力
    return "ok"


@handler.add(MessageEvent, message=TextMessageContent)
def handle_message(event):
    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)

        user_id = event.source.user_id
        user_input = event.message.text

        # ローディングアニメーションを表示
        line_bot_api.show_loading_animation(ShowLoadingAnimationRequest(chatId=chatId, loadingSeconds=60))
        logger.info("ローディングアニメーションを表示しました。")

        # OpenAIでレスポンスメッセージを作成
        recent_messages = fetch_recent_chat_messages(limit=10)
        session_data = {"messages": [{"role": "user", "content": msg[1]} for msg in recent_messages]}
        # response = generate_chat_response(event.message.text)
        response = chat(user_input, session_data)
        logger.info(f"生成されたレスポンス: {response}")

        # DBにメッセージを保存
        save_chat_message(user_id, user_input)
        save_chat_message("ai", response)

        # メッセージを返信
        line_bot_api.reply_message_with_http_info(
            ReplyMessageRequest(reply_token=event.reply_token, messages=[TextMessage(text=response)])
        )
        logger.info("メッセージをユーザーに返信しました。")

@handler.add(MessageEvent, message=StickerMessageContent)
def handle_sticker_message(event):
    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)

        user_id = event.source.user_id
        user_stamp_message = "今、このようなスタンプを送信しました。\n" + str(event.message.keywords)

        # ローディングアニメーションを表示
        line_bot_api.show_loading_animation(ShowLoadingAnimationRequest(chatId=chatId, loadingSeconds=60))
        logger.info("ローディングアニメーションを表示しました。")

        # OpenAIでレスポンスメッセージを作成
        recent_messages = fetch_recent_chat_messages(limit=10)
        session_data = {"messages": [{"role": "user", "content": msg[1]} for msg in recent_messages]}
        # response = generate_chat_response(event.message.text)
        response = chat(user_stamp_message, session_data)
        logger.info(f"生成されたレスポンス: {response}")

        # DBにメッセージを保存
        save_chat_message(user_id, user_stamp_message)
        save_chat_message("ai", response)

        # メッセージを返信
        line_bot_api.reply_message_with_http_info(
            ReplyMessageRequest(reply_token=event.reply_token, messages=[TextMessage(text=response)])
        )
        logger.info("スタンプの返信メッセージをユーザーに返信しました。")


@app.get("/hello")
async def hello():
    return {"message": "hello world!"}


if __name__ == "__main__":
    app.run()
