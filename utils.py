"""
このファイルは、ユーティリティ関数を定義するファイルです。
"""

############################################################
# ライブラリの読み込み
############################################################
import os
from dotenv import load_dotenv
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage
from langchain.docstore.document import Document
from langchain_openai import ChatOpenAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import constants as ct

# pandasは条件付きインポート
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

############################################################
# 設定関連
############################################################
# 「.env」ファイルで定義した環境変数の読み込み
load_dotenv()


############################################################
# 関数定義
############################################################

def get_source_icon(source):
    """
    メッセージと一緒に表示するアイコンの種類を取得

    Args:
        source: 参照元のありか

    Returns:
        メッセージと一緒に表示するアイコンの種類
    """
    # 参照元がWebページの場合とファイルの場合で、取得するアイコンの種類を変える
    if source.startswith("http"):
        icon = ct.LINK_SOURCE_ICON
    else:
        icon = ct.DOC_SOURCE_ICON
    
    return icon


def build_error_message(message):
    """
    エラーメッセージと管理者問い合わせテンプレートの連結

    Args:
        message: 画面上に表示するエラーメッセージ

    Returns:
        エラーメッセージと管理者問い合わせテンプレートの連結テキスト
    """
    return "\n".join([message, ct.COMMON_ERROR_MESSAGE])


def get_llm_response(chat_message):
    """
    LLMからの回答取得

    Args:
        chat_message: ユーザー入力値

    Returns:
        LLMからの回答（context情報も含む）
    """
    # LLMのオブジェクトを用意
    llm = ChatOpenAI(model_name=ct.MODEL, temperature=ct.TEMPERATURE)

    # 会話履歴なしでもLLMに理解してもらえる、独立した入力テキストを取得するためのプロンプトテンプレートを作成
    question_generator_template = ct.SYSTEM_PROMPT_CREATE_INDEPENDENT_TEXT
    question_generator_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", question_generator_template),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )

    # モードによってLLMから回答を取得する用のプロンプトを変更
    if st.session_state.mode == ct.ANSWER_MODE_1:
        # モードが「社内文書検索」の場合のプロンプト
        question_answer_template = ct.SYSTEM_PROMPT_DOC_SEARCH
    else:
        # モードが「社内問い合わせ」の場合のプロンプト
        question_answer_template = ct.SYSTEM_PROMPT_INQUIRY
    # LLMから回答を取得する用のプロンプトテンプレートを作成
    question_answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", question_answer_template),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )

    # 会話履歴なしでもLLMに理解してもらえる、独立した入力テキストを取得するためのRetrieverを作成
    history_aware_retriever = create_history_aware_retriever(
        llm, st.session_state.retriever, question_generator_prompt
    )

    # LLMから回答を取得する用のChainを作成
    question_answer_chain = create_stuff_documents_chain(llm, question_answer_prompt)
    # 「RAG x 会話履歴の記憶機能」を実現するためのChainを作成
    chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # LLMへのリクエストとレスポンス取得
    llm_response = chain.invoke({"input": chat_message, "chat_history": st.session_state.chat_history})
    # LLMレスポンスを会話履歴に追加
    st.session_state.chat_history.extend([HumanMessage(content=chat_message), llm_response["answer"]])

    return llm_response


def split_documents_with_metadata(documents):
    """
    ドキュメントを分割し、ページ情報をメタデータに保持する
    
    Args:
        documents: 分割対象のドキュメントリスト
        
    Returns:
        分割されたドキュメントリスト（ページ情報付き）
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=ct.CHUNK_SIZE,
        chunk_overlap=ct.CHUNK_OVERLAP
    )
    
    splits = []
    for doc in documents:
        # ページ情報がメタデータに含まれているか確認
        if 'page' in doc.metadata:
            page_num = doc.metadata['page']
        else:
            page_num = 0  # PyMuPDFは0ベースなのでデフォルト値は0
        
        # ドキュメントを分割
        doc_splits = text_splitter.split_documents([doc])
        
        # 各分割にページ情報を付与
        for split in doc_splits:
            split.metadata['page'] = page_num
            splits.append(split)
    
    return splits


def display_answer_with_page_info(response, retrieved_docs):
    """
    回答とソースドキュメント（ページ情報付き）を表示
    
    Args:
        response: LLMからの回答テキスト
        retrieved_docs: 検索されたドキュメントリスト
    """
    # 回答を表示
    st.write(response)
    
    # ソースドキュメントを表示
    if retrieved_docs:
        st.markdown("### 参考文書")
        
        for i, doc in enumerate(retrieved_docs, 1):
            # ファイル名を取得
            source = doc.metadata.get('source', 'Unknown')
            filename = source.split('/')[-1] if '/' in source else source
            
            # ページ情報を取得
            page = doc.metadata.get('page', None)
            
            # ページ情報を含めて表示
            if filename.endswith('.pdf') and page is not None:
                st.markdown(f"**{filename}** (ページ {page + 1})")  # ページは0ベースなので+1
            else:
                st.markdown(f"**{filename}**")
            
            # ドキュメントの内容を表示
            st.text(doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content)
            st.markdown("---")


def custom_csv_loader(file_path):
    """
    CSVファイルを正しい部署列で読み込む（修正版）
    
    Args:
        file_path: CSVファイルのパス
        
    Returns:
        正しく人事部を特定したドキュメントのリスト
    """
    # pandasが利用できない場合は従来の方法にフォールバック
    if not PANDAS_AVAILABLE:
        from langchain_community.document_loaders.csv_loader import CSVLoader
        loader = CSVLoader(file_path, encoding="utf-8")
        return loader.load()
    
    try:
        print(f"DEBUG: CSVファイル読み込み開始: {file_path}")
        
        # CSVファイルを読み込み
        df = pd.read_csv(file_path, encoding='utf-8')
        print(f"DEBUG: 読み込み成功。総行数: {len(df)}")
        print(f"DEBUG: カラム一覧: {list(df.columns)}")
        
        # 部署列を正しく特定（部署を最優先）
        dept_columns = ['部署', '部門', 'department', 'dept', '従業員区分']  # 「部署」を最優先
        dept_col = None
        
        print("DEBUG: 部署列の探索（修正版）:")
        for col in dept_columns:
            if col in df.columns:
                dept_col = col
                print(f"DEBUG: 部署列発見: {col}")
                break
        
        if dept_col:
            print(f"DEBUG: 部署列 '{dept_col}' の値一覧:")
            unique_depts = df[dept_col].unique()
            for dept in unique_depts:
                if pd.notna(dept):
                    count = len(df[df[dept_col] == dept])
                    print(f"  - {dept}: {count}人")
        
        # 人事部従業員を抽出（正しい列で検索）
        hr_employees = None
        if dept_col:
            print("DEBUG: 人事部従業員の抽出試行（修正版）:")
            print(f"DEBUG: 検索条件: {dept_col}.str.contains('人事', na=False)")
            
            # 人事部を含む部署を検索
            hr_mask = df[dept_col].astype(str).str.contains('人事', na=False)
            hr_employees = df[hr_mask]
            print(f"DEBUG: 人事部従業員数: {len(hr_employees)}")
            
            if len(hr_employees) > 0:
                print("DEBUG: 人事部従業員発見:")
                for idx, (_, emp) in enumerate(hr_employees.iterrows(), 1):
                    name = emp.get('氏名（フルネーム）', emp.get('氏名', f'名前不明{idx}'))
                    dept = emp.get(dept_col, '部署不明')
                    print(f"  {idx}. {name} - {dept}")
            else:
                print("DEBUG: '人事'で見つからず。別の検索語を試行:")
                # より柔軟な検索
                search_terms = ['人事', 'HR', '人事部', '人事課', '総務', '管理']
                for search_term in search_terms:
                    mask = df[dept_col].astype(str).str.contains(search_term, na=False, case=False)
                    matches = df[mask]
                    print(f"  '{search_term}' 検索結果: {len(matches)}人")
                    if len(matches) > 0:
                        print(f"    該当部署: {matches[dept_col].unique()}")
                        # 最初にヒットした検索語を使用
                        if hr_employees is None or len(hr_employees) == 0:
                            hr_employees = matches
        
        # 超大型統合ドキュメント生成
        content_lines = []
        
        # タイトルセクション
        content_lines.append("=" * 80)
        content_lines.append("社員名簿データベース - 人事部従業員完全一覧")
        content_lines.append("=" * 80)
        content_lines.append("")
        content_lines.append("【重要】人事部に所属している従業員情報を一覧化")
        content_lines.append("人事部 人事部門 HR部 人事課 人事担当 人事部員 HR担当者")
        content_lines.append("人事スタッフ 人事チーム 人事メンバー 人事職員")
        content_lines.append("")
        
        # 人事部従業員の完全統合情報
        if hr_employees is not None and not hr_employees.empty:
            content_lines.append(f"🏢 人事部総従業員数: {len(hr_employees)}人")
            content_lines.append("人事部に所属している従業員の完全な一覧は以下の通りです：")
            content_lines.append("")
            content_lines.append("-" * 100)
            content_lines.append("人事部従業員完全リスト（詳細版）")
            content_lines.append("-" * 100)
            content_lines.append("")
            
            # 各従業員の詳細情報を統合
            for idx, (_, emp) in enumerate(hr_employees.iterrows(), 1):
                content_lines.append(f"🔸 人事部従業員 {idx}番目")
                content_lines.append(f"   氏名: {emp.get('氏名（フルネーム）', '不明')}")
                content_lines.append(f"   社員ID: {emp.get('社員ID', '不明')}")
                content_lines.append(f"   所属: {emp.get(dept_col, '人事部')}")
                
                # 全項目の詳細情報
                for col, val in emp.items():
                    if pd.notna(val) and str(val).strip():
                        content_lines.append(f"   {col}: {val}")
                
                content_lines.append("")
                content_lines.append(f"   ※ {emp.get('氏名（フルネーム）', f'従業員{idx}')}は人事部に所属している従業員です")
                content_lines.append("   " + "=" * 80)
                content_lines.append("")
            
            # サマリーセクション
            content_lines.append("-" * 100)
            content_lines.append("🔍 人事部従業員サマリー")
            content_lines.append("-" * 100)
            content_lines.append("")
            content_lines.append("【人事部従業員名簿】")
            for idx, (_, emp) in enumerate(hr_employees.iterrows(), 1):
                name = emp.get('氏名（フルネーム）', f'従業員{idx}')
                emp_id = emp.get('社員ID', '不明')
                role = emp.get('役職', '職員')
                dept = emp.get(dept_col, '人事部')
                content_lines.append(f"{idx}. {name} (社員ID: {emp_id}) - 役職: {role} - 所属: {dept}")
            
            content_lines.append("")
            content_lines.append("【重要確認】")
            content_lines.append(f"上記リストが人事部に所属している全{len(hr_employees)}人の従業員です。")
            content_lines.append("人事部員 人事担当者 HR部員 人事スタッフの完全な一覧情報。")
            
        else:
            content_lines.append("⚠️ 人事部従業員が見つかりませんでした")
            content_lines.append("デバッグ情報:")
            content_lines.append(f"検索対象列: {dept_col}")
            if dept_col and dept_col in df.columns:
                content_lines.append("部署一覧:")
                for dept in df[dept_col].unique():
                    if pd.notna(dept):
                        content_lines.append(f"  - {dept}")
        
        # 統合されたテキストを作成
        content = "\n".join(content_lines)
        
        # 1つの巨大な統合ドキュメントとして作成
        doc = Document(
            page_content=content,
            metadata={
                "source": file_path,
                "file_type": "csv",
                "document_type": "corrected_hr_database",
                "total_employees": len(df),
                "hr_employees": len(hr_employees) if hr_employees is not None else 0,
                "search_column": dept_col,
                "description": "修正版人事部従業員統合データベース"
            }
        )
        
        return [doc]
    
    except Exception as e:
        print(f"DEBUG: CSVエラー: {e}")
        # エラーの場合は従来の方法にフォールバック
        from langchain_community.document_loaders.csv_loader import CSVLoader
        loader = CSVLoader(file_path, encoding="utf-8")
        return loader.load()


def display_answer_with_sources(llm_response):
    """
    LLMレスポンスから回答とソースドキュメント（ページ情報付き）を表示
    
    Args:
        llm_response: LLMからの回答レスポンス
    """
    answer = llm_response["answer"]
    context_docs = llm_response.get("context", [])
    
    # 回答とソース情報を表示
    display_answer_with_page_info(answer, context_docs)