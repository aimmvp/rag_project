import streamlit as st
from langchain_core.messages.chat import ChatMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_teddynote.prompts import load_prompt
from langchain import hub


# API KEY를 환경변수로 관리하기 위한 설정 파일
from dotenv import load_dotenv

# API KEY 정보로드
load_dotenv()


st.title("나의 챗GPT s0wnd !!!")

# 처음 1번만 실행하기 위한 코드
if "messages" not in st.session_state:
    # 대화기록을 저장하기 위한 용도로 생성
    st.session_state["messages"] = []

# 사이드바 생성
with st.sidebar:
    clear_btn = st.button("대화 초기화")

    selected_prompt = st.selectbox(
        "프롬프트를 선택해 주세요", ("기본", "SNS 게시글", "요약"), index=0
    )


# 이전대화를 출력
def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)


# 새로운 메시지 추가
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))


# 체인 생성
def create_chain(prompt_type):
    # prompt | llm | output_parser
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "당신은 친절한 AI 어시스턴트입니다. 다음의 질문에 간결하게 답변해 주세요",
            ),
            ("user", "#Question:\n{question}"),
        ]
    )

    if prompt_type == "SNS 게시글":
        # encoding 은 Windows 사용자는 cp949
        prompt = load_prompt("prompts/sns.yaml", encoding="utf-8")

    elif prompt_type == "요약":
        prompt = hub.pull("teddynote/chain-of-density-korean")

    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    return chain


# 초기화 버튼이 눌리면
if clear_btn:
    st.session_state["messages"] = []
# 이전 대화 기록 출력
print_messages()

# 사용자의 입력을 받는다.
user_input = st.chat_input("궁금한 내용 입력 하세요")

if user_input:
    # 사용자 입력
    st.chat_message("user").write(user_input)
    # Chain 생성
    chain = create_chain(selected_prompt)
    response = chain.stream({"question": user_input})
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
