import streamlit as st
from langchain_core.messages.chat import ChatMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_teddynote.prompts import load_prompt
from langchain_teddynote import logging


import glob
import os

# PDF 기반의 RAG 시스템

# API KEY를 환경변수로 관리하기 위한 설정 파일
from dotenv import load_dotenv

# API KEY 정보로드
load_dotenv()

# Langsmith 프로젝트 이름을 입력합니다.
logging.langsmith("[Pjt] PDF-RAG")

# 캐시 디렉토리 생성
if not os.path.exists(".cache"):
    os.mkdir(".cache")

if not os.path.exists(".cache/files"):
    os.mkdir(".cache/files")

if not os.path.exists(".cache/embeddings"):
    os.mkdir(".cache/embeddings")


st.title("PDF 기반의 RAG QA 챗 GPT⌨️")

# 처음 1번만 실행하기 위한 코드
if "messages" not in st.session_state:
    # 대화기록을 저장하기 위한 용도로 생성
    st.session_state["messages"] = []

if "chain" not in st.session_state:
    # 아무런 파일도 업로드 하지 않을 경우
    st.session_state["chain"] = None

# 사이드바 생성
with st.sidebar:
    clear_btn = st.button("대화 초기화")
    # 파일 업로드
    uploaded_file = st.file_uploader("파일 업로드", type=["pdf"])
    # LLM  모델 선택
    selected_model = st.selectbox(
        "LLM 선택", ["gpt-4o", "gpt-4-turbo", "gpt-4o-mini"], index=0
    )


# 이전대화를 출력
def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)


# 새로운 메시지 추가
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))


# 파일을 캐시 저장(시간이 오래 걸리는 작업 처리 예정)
@st.cache_resource(show_spinner="업로드한 파일을 처리 중입니다.")
def embed_file(file):
    # 업로드한 파일을 캐시 디렉토리에 저장
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    # 단계 1: 문서 로드(Load Documents)
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()

    # 단계 2: 문서 분할(Split Documents)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    split_documents = text_splitter.split_documents(docs)

    # 단계 3: 임베딩(Embedding) 생성
    embeddings = OpenAIEmbeddings()

    # 단계 4: DB 생성(Create DB) 및 저장
    # 벡터스토어를 생성합니다.
    vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)

    # 단계 5: 검색기(Retriever) 생성
    # 문서에 포함되어 있는 정보를 검색하고 생성합니다.
    retriever = vectorstore.as_retriever()
    return retriever


# 체인 생성
def create_chain(retriever, model_name="gpt-4o"):

    # 단계 6: 프롬프트 생성(Create Prompt)
    # 프롬프트를 생성합니다.
    prompt = load_prompt("prompts/pdf-rag.yaml", encoding="utf-8")

    # 단계 7: 언어모델(LLM) 생성
    # 모델(LLM) 을 생성합니다.
    llm = ChatOpenAI(model_name=model_name, temperature=0)

    # 단계 8: 체인(Chain) 생성
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


# 파일이 업로드 되었을 때
if uploaded_file:
    # 파일 업로드 후 retriever 생성(작업시간 오래 걸림)
    retriever = embed_file(uploaded_file)
    chain = create_chain(retriever, model_name=selected_model)
    st.session_state["chain"] = chain

# 초기화 버튼이 눌리면
if clear_btn:
    st.session_state["messages"] = []
# 이전 대화 기록 출력
print_messages()

# 사용자의 입력을 받는다.
user_input = st.chat_input("궁금한 내용 입력 하세요")

# 경고 메시지를 띄우기 위한 빈 영역
warining_msg = st.empty()

if user_input:

    # Chain 생성
    chain = st.session_state["chain"]

    if chain is not None:
        # 사용자 입력
        st.chat_message("user").write(user_input)

        # 스트리밍 호출
        response = chain.stream(user_input)
        with st.chat_message("assistant"):
            # 빈 공간을 만들어서, 여기에 토큰을 스트리밍 출력
            container = st.empty()

            ai_answer = ""
            for token in response:
                ai_answer += token
                container.markdown(ai_answer)

        # 대화 기록 저장
        add_message("user", user_input)
        add_message("assistant", ai_answer)
    else:
        # 파일 업로드 하라는 경고 메세지 출력
        warining_msg.error("파일을 업로드 해주세요")
