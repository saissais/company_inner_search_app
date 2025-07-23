"""
このファイルは、最初の画面読み込み時にのみ実行される初期化処理が記述されたファイルです。
"""

############################################################
# ライブラリの読み込み
############################################################
import os
import logging
from logging.handlers import TimedRotatingFileHandler
from uuid import uuid4
import sys
import unicodedata
from dotenv import load_dotenv
import streamlit as st
from docx import Document
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter  
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import constants as ct

# デバッグ用インポート
try:
    from utils import split_documents_with_metadata, custom_csv_loader  
    print("DEBUG: utils.pyからのインポート成功")
except Exception as e:
    print(f"DEBUG: utils.pyからのインポートエラー: {e}")
    raise e


############################################################
# 設定関連
############################################################
# 「.env」ファイルで定義した環境変数の読み込み
load_dotenv()


############################################################
# 関数定義
############################################################

def initialize():
    """
    画面読み込み時に実行する初期化処理
    """
    try:
        print("DEBUG: 初期化開始")
        
        # 初期化データの用意
        print("DEBUG: セッション状態初期化")
        initialize_session_state()
        
        # ログ出力用にセッションIDを生成
        print("DEBUG: セッションID生成")
        initialize_session_id()
        
        # ログ出力の設定
        print("DEBUG: ログ設定")
        initialize_logger()
        
        # RAGのRetrieverを作成
        print("DEBUG: Retriever作成開始")
        initialize_retriever()
        
        print("DEBUG: 初期化完了")
    except Exception as e:
        print(f"DEBUG: 初期化エラー: {e}")
        import traceback
        traceback.print_exc()
        raise e


def initialize_logger():
    """
    ログ出力の設定
    """
    os.makedirs(ct.LOG_DIR_PATH, exist_ok=True)
    logger = logging.getLogger(ct.LOGGER_NAME)

    if logger.hasHandlers():
        return

    log_handler = TimedRotatingFileHandler(
        os.path.join(ct.LOG_DIR_PATH, ct.LOG_FILE),
        when="D",
        encoding="utf8"
    )
    formatter = logging.Formatter(
        f"[%(levelname)s] %(asctime)s line %(lineno)s, in %(funcName)s, session_id={st.session_state.session_id}: %(message)s"
    )
    log_handler.setFormatter(formatter)
    logger.setLevel(logging.INFO)
    logger.addHandler(log_handler)


def initialize_session_id():
    """
    セッションIDの作成
    """
    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid4().hex


def initialize_retriever():
    """
    画面読み込み時にRAGのRetriever（ベクターストアから検索するオブジェクト）を作成（ページ情報保持対応版）
    """
    logger = logging.getLogger(ct.LOGGER_NAME)

    if "retriever" in st.session_state:
        print("DEBUG: Retriever既に存在")
        return
    
    try:
        print("DEBUG: データソース読み込み開始")
        docs_all = load_data_sources()
        print(f"DEBUG: 読み込み完了、ドキュメント数: {len(docs_all)}")

        for doc in docs_all:
            doc.page_content = adjust_string(doc.page_content)
            for key in doc.metadata:
                doc.metadata[key] = adjust_string(doc.metadata[key])
        
        print("DEBUG: Embeddings作成")
        embeddings = OpenAIEmbeddings()
        
        print("DEBUG: ドキュメント分割開始")
        # ページ情報を保持したドキュメント分割を使用
        splitted_docs = split_documents_with_metadata(docs_all)
        print(f"DEBUG: 分割完了、チャンク数: {len(splitted_docs)}")

        print("DEBUG: ベクターストア作成")
        db = FAISS.from_documents(splitted_docs, embeddings)

        print("DEBUG: Retriever作成")
        st.session_state.retriever = db.as_retriever(
            search_kwargs={"k": ct.RETRIEVER_TOP_K}
        )
        print("DEBUG: Retriever作成完了")
        
    except Exception as e:
        print(f"DEBUG: Retriever作成エラー: {e}")
        import traceback
        traceback.print_exc()
        raise e


def initialize_session_state():
    """
    初期化データの用意
    """
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.chat_history = []


def load_data_sources():
    """
    RAGの参照先となるデータソースの読み込み

    Returns:
        読み込んだ通常データソース
    """
    try:
        docs_all = []
        print(f"DEBUG: フォルダチェック開始: {ct.RAG_TOP_FOLDER_PATH}")
        recursive_file_check(ct.RAG_TOP_FOLDER_PATH, docs_all)
        print(f"DEBUG: ファイル読み込み完了、ドキュメント数: {len(docs_all)}")

        web_docs_all = []
        for web_url in ct.WEB_URL_LOAD_TARGETS:
            print(f"DEBUG: Web読み込み: {web_url}")
            loader = WebBaseLoader(web_url)
            web_docs = loader.load()
            web_docs_all.extend(web_docs)
        docs_all.extend(web_docs_all)
        print(f"DEBUG: 総ドキュメント数: {len(docs_all)}")

        return docs_all
    except Exception as e:
        print(f"DEBUG: データソース読み込みエラー: {e}")
        import traceback
        traceback.print_exc()
        raise e


def recursive_file_check(path, docs_all):
    """
    RAGの参照先となるデータソースの読み込み

    Args:
        path: 読み込み対象のファイル/フォルダのパス
        docs_all: データソースを格納する用のリスト
    """
    try:
        if os.path.isdir(path):
            files = os.listdir(path)
            for file in files:
                full_path = os.path.join(path, file)
                recursive_file_check(full_path, docs_all)
        else:
            print(f"DEBUG: ファイル処理: {path}")
            file_load(path, docs_all)
    except Exception as e:
        print(f"DEBUG: ファイルチェックエラー ({path}): {e}")
        raise e


def file_load(path, docs_all):
    """
    ファイル内のデータ読み込み（CSV統合対応版）
    """
    try:
        file_extension = os.path.splitext(path)[1]
        file_name = os.path.basename(path)
        print(f"DEBUG: ファイル読み込み: {file_name} (拡張子: {file_extension})")

        if file_extension in ct.SUPPORTED_EXTENSIONS:
            # CSVファイルの場合は統合ローダーを使用
            if file_extension == ".csv":
                print(f"DEBUG: CSV統合ローダー使用: {path}")
                docs = custom_csv_loader(path)
                print(f"DEBUG: CSV読み込み完了: {len(docs)}ドキュメント")
            else:
                print(f"DEBUG: 標準ローダー使用: {path}")
                # その他のファイルは標準ローダーを使用
                loader_config = ct.SUPPORTED_EXTENSIONS[file_extension]
                loader = loader_config(path)
                docs = loader.load()
                print(f"DEBUG: 標準読み込み完了: {len(docs)}ドキュメント")
            
            docs_all.extend(docs)
        else:
            print(f"DEBUG: サポート外拡張子: {file_extension}")
    except Exception as e:
        print(f"DEBUG: ファイル読み込みエラー ({path}): {e}")
        import traceback
        traceback.print_exc()
        raise e


def adjust_string(s):
    """
    Windows環境でRAGが正常動作するよう調整
    
    Args:
        s: 調整を行う文字列
    
    Returns:
        調整を行った文字列
    """
    if type(s) is not str:
        return s

    if sys.platform.startswith("win"):
        s = unicodedata.normalize('NFC', s)
        s = s.encode("cp932", "ignore").decode("cp932")
        return s

    return s