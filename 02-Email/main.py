import streamlit as st
from langchain_core.messages.chat import ChatMessage
from langchain_core.prompts import ChatPromptTemplate

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

from langchain_teddynote.prompts import load_prompt
import glob


from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
import os
from langchain_community.utilities import SerpAPIWrapper


# 이메일 본문에서 주요 엔티티 추출하기
class EmailSummary(BaseModel):
    person: str = Field(description="메일을 보낸 사람")
    company_info: str = Field(description="메일을 보낸 사람의 회사")
    phone_number: str = Field(description="메일 본문에 언급된 전화번호")
    email: str = Field(description="메일을 보낸 사람의 이메일 주소")
    subject: str = Field(description="메일 제목")
    summary: str = Field(description="메일 본문을 요약한 텍스트")
    date: str = Field(description="메일 본문에 언급된 미팅 날짜와 시간")


from langchain_core.prompts import PromptTemplate


# API KEY를 환경변수로 관리하기 위한 설정 파일
from dotenv import load_dotenv

# API KEY 정보로드
load_dotenv()


st.title("Email 요약하기 📧 !!!")

# 처음 1번만 실행하기 위한 코드
if "messages" not in st.session_state:
    # 대화기록을 저장하기 위한 용도로 생성
    st.session_state["messages"] = []

# 사이드바 생성
with st.sidebar:
    clear_btn = st.button("대화 초기화")


# 이전대화를 출력
def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)


# 새로운 메시지 추가
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))


# 체인 생성
def create_email_parsing_chain():

    # PydanticOutputParser 생성
    output_parser = PydanticOutputParser(pydantic_object=EmailSummary)
    prompt = PromptTemplate.from_template(
        """
    You are a helpful assistant. Please answer the following questions in KOREAN.

    #QUESTION:
    다음의 이메일 내용 중에서 주요 내용을 추출해 주세요

    #EMAIL CONVERSATION:
    {email_conversation}

    #FORMAT:
    {format}
    """
    )

    # format 에 PydanticOutputParser의 부분 포맷팅(partial) 추가
    prompt = prompt.partial(format=output_parser.get_format_instructions())

    # 체인생성
    # prompt 를 만들려면 output_parser 의 format 이 필요함
    chain = prompt | ChatOpenAI(model="gpt-4-turbo") | output_parser
    return chain


def create_report_chain():
    # prompt 적용
    report_prompt = load_prompt("prompts/email.yaml", encoding="utf-8")

    llm = ChatOpenAI(model_name="gpt-4-turbo", temperature=0)
    output_parser = StrOutputParser()
    chain = report_prompt | llm | output_parser

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
    # 1) 이메일 파싱 Chain 생성
    email_chain = create_email_parsing_chain()
    answer = email_chain.invoke({"email_conversation": user_input})

    # 2) 보낸 사람의 추가 정보 수집(검색)
    params = {"engine": "google", "gl": "kr", "hl": "ko", "num": "3"}
    search = SerpAPIWrapper(params=params)
    search_query = f"{answer.person} {answer.company_info} {answer.email}"
    search_result = search.run(search_query)
    search_result = eval(search.run(search_query))
    search_result_string = "\n".join(search_result)

    # 3) 이메일 요약 리포트 생성
    report_chain = create_report_chain()
    report_chain_input = {
        "sender": answer.person,
        "additional_information": search_result_string,
        "company": answer.company_info,
        "email": answer.email,
        "subject": answer.subject,
        "summary": answer.summary,
        "date": answer.date,
    }

    # 스트리밍 호출
    response = report_chain.stream(report_chain_input)
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
