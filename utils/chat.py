import os

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langsmith import Client

from utils.common import read_markdown_file
from utils.config import logger
from utils.cosmos import fetch_recent_chat_messages, save_chat_message

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

client = Client()


def generate_chat_response(user_prompt):
    logger.info("チャットレスポンス生成を開始します。")

    parser = StrOutputParser()
    # プロンプトを設定
    system_prompt = read_markdown_file("prompts/system_prompt.txt")
    # system_prompt,過去の会話履歴,user_promptを組み合わせてプロンプト作成
    messages_list = fetch_recent_chat_messages()
    messages_list = [("system", system_prompt)] + messages_list + [("user", user_prompt)]
    prompt = ChatPromptTemplate.from_messages(messages_list)

    # chatモデルを設定
    # chat = ChatOpenAI(model_name="gpt-4-turbo", temperature=1, max_tokens=256)
    # chat = ChatAnthropic(model="claude-3-haiku-20240307", max_tokens=256, temperature=0.7)
    chat = ChatAnthropic(model="claude-3-opus-20240229", max_tokens=256, temperature=0.7)

    # chainを設定
    chain = prompt | chat | parser
    try:
        result = chain.invoke({"user_input": user_prompt})
        logger.info("チャットレスポンスが生成されました。")
    except Exception as e:
        logger.error(f"チャットレスポンスの生成に失敗しました: {e}")
        result = "エラーが発生しました。"
    # チャットレスポンスをデータベースに保存
    save_chat_message("human", user_prompt)
    save_chat_message("ai", result)

    return result


if __name__ == "__main__":
    generate_chat_response("こんにちは")
