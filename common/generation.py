import logging
import os
import pickle
import time
from pathlib import Path

import nltk
from openai import AzureOpenAI
from dotenv import load_dotenv
from pinecone import Pinecone
import copy

# 必要な NLTK データのダウンロード
nltk.download("punkt_tab")

# 環境変数の読み込み
load_dotenv()

# ログ設定
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# 既存のハンドラを削除
if logger.hasHandlers():
    logger.handlers.clear()

# ハンドラを追加
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)

# 定数の定義
OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
OPENAI_API_BASE = os.environ.get("AZURE_END_POINT")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = "gcp-starter"
INDEX_NAME = "hybrid-search"
MODELS_DIR = Path(__file__).resolve().parent / "models"
PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
API_VERSION = "2024-03-01-preview"

# エンベディング用のクライアント
client = AzureOpenAI(
    api_version=API_VERSION,
    api_key=OPENAI_API_KEY,  # Key
    azure_endpoint=OPENAI_API_BASE,  # EndPoint
)

# Pinecone の設定
pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
index = pc.Index(INDEX_NAME)


# BM25 モデルのロード
def load_bm25_model(filename):
    model_path = MODELS_DIR / filename
    with open(model_path, "rb") as f:
        return pickle.load(f)


# プロンプトの読み込み
def load_prompt(filename):
    with open(PROMPTS_DIR / filename, "r", encoding="utf-8") as file:
        return file.read()


def generate_embeddings(text):
    try:
        return client.embeddings.create(input=[text], model="mkj-aj-ada").data[0].embedding
    except Exception as e:
        logger.exception("エンベディング生成中にエラーが発生しました: %s", e)
        return None


# 相談事か否かを判定しプロンプトの選択
def select_prompt(message):
    try:
        system_prompt_for_judge = load_prompt("system_prompt_for_judge.txt")
        user_prompt_for_judge_template = load_prompt("user_prompt_for_judge.txt")
        user_prompt_for_judge = user_prompt_for_judge_template.format(
            message=message,
        )
        response = client.chat.completions.create(
            model="mkj-aj-gpt4o",
            messages=[
                {"role": "system", "content": system_prompt_for_judge},
                {"role": "user", "content": user_prompt_for_judge},
            ],
        )
        judge = response.choices[0].message.content
        logger.info(f"judge:{judge}")
        return judge
    except Exception as e:
        logger.exception("チャット処理中にエラーが発生しました: %s", e)
        return "申し訳ありませんが、リクエストの処理中に問題が発生しました。"


bm25_recipe = load_bm25_model("bm25_model_1.pkl")
bm25_symptom = load_bm25_model("bm25_model_2.pkl")


# 100文字以上の場合はクエリを生成
def generate_query(user_message, messages):
    system_prompt_for_query = load_prompt("system_prompt_for_query")
    user_history = copy.deepcopy(messages)

    # 会話履歴はそのまま、システムプロンプトを置き換え
    if not user_history:
        user_history.append({"role": "system", "content": system_prompt_for_query})
    else:
        user_history[0]["content"] = system_prompt_for_query

    user_history.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model="mkj-aj-gpt4o",
        messages=user_history,
    )
    query = response.choices[0].message.content

    return query


def chat(user_message, session_data):
    try:
        # セッションデータの初期化
        messages = session_data.get("messages", [])

        # 相談事か否かでプロンプトを選択
        judge = select_prompt(user_message)

        if judge == "True":
            # 100文字以上の場合は検索クエリを新規作成
            if len(user_message) > 100:
                query = generate_query(user_message, session_data["messages"])
            else:
                query = user_message

            logger.info(f"クエリ：{query}")

            # エンベディングの生成
            dense = generate_embeddings(query)
            if dense is None:
                return "申し訳ありませんが、現在エンベディングを生成できません。"

            # レシピ検索
            recipe_results = index.query(
                namespace="recipe",
                top_k=4,
                vector=dense,
                sparse_vector=bm25_recipe.encode_queries([query])[0],
                include_metadata=True,
            )
            recipe_list = [doc["metadata"]["content"] for doc in recipe_results["matches"]]

            # 症状検索
            symptom_results = index.query(
                namespace="symptom",
                top_k=4,
                vector=dense,
                sparse_vector=bm25_symptom.encode_queries([query])[0],
                include_metadata=True,
            )
            symptom_list = [doc["metadata"]["content"] for doc in symptom_results["matches"]]

            system_prompt_content = load_prompt("system_prompt_for_consultation.txt")
        else:
            system_prompt_content = load_prompt("system_prompt_simple.txt")

        logger.info(f"メッセージ：{messages}")

        if not messages:
            messages.append({"role": "system", "content": system_prompt_content})
        else:
            messages[0]["content"] = system_prompt_content

        logger.info(f"メッセージ：{messages}")

        if judge == "True":
            # ユーザープロンプトの生成
            user_prompt_template = load_prompt("user_prompt_for_consultation.txt")

            user_prompt = user_prompt_template.format(
                user_message=user_message,
                chat_history=messages[1:],  # 最初のメッセージはシステムプロンプトなので除外
                recipe_list="\n".join(recipe_list),
                symptom_list="\n\n".join(symptom_list),
            )
        else:
            user_prompt_template = load_prompt("user_prompt_simple.txt")

            user_prompt = user_prompt_template.format(
                user_message=user_message,
                chat_history=messages[1:],  # 最初のメッセージはシステムプロンプトなので除外
            )
        logger.info(user_prompt)

        # ユーザーのメッセージをセッションデータに追加
        messages.append({"role": "user", "content": user_prompt})
        session_data["messages"] = messages

        # OpenAI API で応答を生成
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model="mkj-aj-gpt4o",
                    messages=messages,
                    temperature=1.0,
                )

                # 会話履歴を入力プロンプトからユーザーの発言のみに変更
                messages[-1] = {"role": "user", "content": user_message}

                # アシスタントの応答を取得
                response_message = response.choices[0].message.content

                # アシスタントのメッセージをセッションデータに追加
                messages.append({"role": "assistant", "content": response_message})
                session_data["messages"] = messages

                return response_message
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"OpenAI API error after {max_retries} attempts: {e}")
                    return (
                        "申し訳ありませんが、リクエストの処理中に問題が発生しました。しばらくしてからもう一度お試しください。"
                    )
                else:
                    logger.warning(f"OpenAI API error (attempt {attempt + 1}): {e}")
                    time.sleep(2**attempt)  # 指数バックオフ

    except Exception as e:
        logger.exception("チャット処理中にエラーが発生しました: %s", e)
        return "申し訳ありませんが、リクエストの処理中に問題が発生しました。"
