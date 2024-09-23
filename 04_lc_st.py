import streamlit as st
from langchain.chat_models import init_chat_model
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
'''
[Langchain source code for this app](https://github.com/langchain-ai/streamlit-agent/blob/main/streamlit_agent/basic_memory.py)
'''

# 页面配置
st.set_page_config(page_title="AI Chatbot", page_icon="🤖")
st.title("🤖 AI Chatbot")

# 侧边栏配置
api_key = st.sidebar.text_input("API Key", type="password", value='sk-51db804f09e541dfacc9cabcf17749bd')
base_url = st.sidebar.text_input("Base URL", value="https://api.deepseek.com")
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.1)
max_turns = st.sidebar.slider("Max Turns", min_value=1, max_value=10, value=5, step=1)

if not api_key:
    st.warning("Please enter your OpenAI API key in the sidebar to start chatting.")
    st.stop()

# 设置聊天历史
msgs = StreamlitChatMessageHistory(key="chat_messages")
if len(msgs.messages) == 0:
    msgs.add_message(SystemMessage(content="You are a helpful assistant. Try to think step by step whenever possible."))
    msgs.add_message(AIMessage(content="How can I help you today?"))

# 设置LangChain
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an AI chatbot having a conversation with a human."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])

llm = init_chat_model(
        model="deepseek-chat",
        model_provider="openai",
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
    )
chain = prompt | llm | StrOutputParser()
chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: msgs,
    input_messages_key="question",
    history_messages_key="history",
)

# 显示聊天消息
for msg in msgs.messages[1:]:  # Skip the system message
    st.chat_message(msg.type).write(msg)

# 处理用户输入
if user_input := st.chat_input("Type your message here..."):
    st.chat_message("human").write(user_input)
    config = {"configurable": {"session_id": "any"}}
    response = chain_with_history.invoke({"question": user_input}, config)
    st.chat_message("ai").write(response)

# 管理聊天历史
def manage_chat_history(history, max_turns):
    messages = history.messages
    if len(messages) > max_turns * 2 + 1:
        messages = [messages[0]] + messages[-(max_turns * 2):]
    return messages

msgs.messages = manage_chat_history(msgs, max_turns)

# 清除聊天按钮
if st.button("Clear Chat"):
    msgs.clear()
    msgs.add_message(SystemMessage(content="You are a helpful assistant. Try to think step by step whenever possible."))
    msgs.add_message(AIMessage(content="How can I help you today?"))
    st.rerun()

# 显示消息历史（可选）
with st.expander("View Message History"):
    st.json(st.session_state.chat_messages)