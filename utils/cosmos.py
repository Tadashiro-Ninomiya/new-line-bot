import os
import uuid
from datetime import datetime, timedelta

import pytz
from azure.cosmos import CosmosClient, PartitionKey, exceptions
from dotenv import load_dotenv
from fastapi import HTTPException

from utils.config import logger

# .envファイルを読み込む
load_dotenv()


def get_database_config():
    return {
        "url": os.getenv("COSMOS_DB_ACCOUNT_URL"),
        "key": os.getenv("COSMOS_DB_ACCOUNT_KEY"),
        "database_name": os.getenv("COSMOS_DB_DATABASE_NAME"),
        "container_name": "CHAT",
    }


def initialize_cosmos_db():
    config = get_database_config()
    client = CosmosClient(url=config["url"], credential=config["key"])
    try:
        database = client.create_database_if_not_exists(id=config["database_name"])
        container = database.create_container_if_not_exists(
            id=config["container_name"], partition_key=PartitionKey(path="/id")
        )
        logger.info("データベースとコンテナの初期化が成功しました。")
        return container
    except exceptions.CosmosHttpResponseError as e:
        logger.error(f"データベースまたはコンテナの作成に失敗しました: {e}")
        raise HTTPException(status_code=500, detail="データベースの操作に失敗しました")


def save_chat_message(user, message):
    try:
        container = initialize_cosmos_db()
        # 現在の日時を取得
        now = datetime.now(pytz.timezone("Asia/Tokyo"))
        # 保存するデータを作成
        data = {"id": uuid.uuid4().hex, "date": now.isoformat(), "user": user, "message": message}
        # CosmosDBにデータを保存
        container.create_item(data)
        logger.info("チャットメッセージが正常に保存されました。")
    except exceptions.CosmosHttpResponseError as e:
        logger.error(f"CosmosDBへのデータ保存に失敗しました: {e}")
        raise HTTPException(status_code=500, detail="メッセージの保存に失敗しました")


def fetch_recent_chat_messages(limit=10):
    try:
        container = initialize_cosmos_db()
        # CosmosDBから最新のチャットメッセージを取得
        # 最新{limit}件のitemを取得するためにここではDESCを指定
        query = "SELECT * FROM c ORDER BY c.date DESC OFFSET 0 LIMIT @limit"
        items = list(
            container.query_items(
                query=query, parameters=[{"name": "@limit", "value": limit}], enable_cross_partition_query=True
            )
        )
        # 現在の日時を取得
        now = datetime.now(pytz.timezone("Asia/Tokyo"))
        # 取得したitemの中で最新のものが日本時間の現在時刻と比べて1時間以内かを確認
        recent_items = [item for item in items if datetime.fromisoformat(item["date"]) > now - timedelta(hours=1)]
        # ユーザー名とメッセージのタプルのリストに整形
        formatted_items = [(item["user"], item["message"]) for item in reversed(recent_items)]
        logger.info("最新のチャットメッセージが正常に取得されました。")
        return formatted_items
    except exceptions.CosmosHttpResponseError as e:
        logger.error(f"CosmosDBからのデータ取得に失敗しました: {e}")
        raise HTTPException(status_code=500, detail="チャットメッセージの取得に失敗しました")
