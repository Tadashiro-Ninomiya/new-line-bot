from utils.config import logger


def read_markdown_file(file_path):
    """
    指定されたパスのMarkdownファイルを読み込み、その内容を返します。

    Args:
    file_path (str): Markdownファイルのパス

    Returns:
    str: ファイルの内容
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        return "ファイルが見つかりませんでした。"
    except Exception as e:
        return f"ファイルの読み込み中にエラーが発生しました: {str(e)}"
