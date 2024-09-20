from langchain.chat_models import init_chat_model
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
import streamlit as st



# 初始化聊天模型
def initialize_chat_model(api_key, base_url, temperature):
    if not api_key:
        return None
    return init_chat_model(
        model="deepseek-chat",
        model_provider="openai",
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
    )

# 管理聊天历史，保持最大对话轮数
def manage_chat_history(history, max_turns):
    messages = history.messages
    if len(messages) > max_turns * 2 + 1:
        messages = [messages[0]] + messages[-(max_turns * 2):]
    return messages

# 向历史记录添加消息，并管理最大对话轮数
def add_message_to_history(history, message, max_turns):
    history.add_message(message)
    return manage_chat_history(history, max_turns)

# 清除聊天历史并添加系统消息
def clear_chat_history(history):
    history.clear()
    history.add_message(SystemMessage(content='You are a helpful assistant. Try to think step by step whenever possible'))

# 显示聊天消息
def display_chat_messages(history):
    for message in history.messages[1:]:
        if isinstance(message, HumanMessage):
            st.chat_message("user").write(message.content)
        elif isinstance(message, AIMessage):
            st.chat_message("assistant").write(message.content)

# 处理用户输入
def handle_user_input(user_query, history, llm, max_turns):
    if user_query:
        add_message_to_history(history, HumanMessage(content=user_query), max_turns)
        with st.spinner("Thinking..."):
            response = llm.invoke(history.messages)
        add_message_to_history(history, AIMessage(content=response.content), max_turns)
        st.rerun()

def main():
    # 设置页面标题
    st.title("AI Chatbot")

    # 在侧边栏创建输入字段和滑块
    api_key = st.sidebar.text_input("API Key", type="password")
    base_url = st.sidebar.text_input("Base URL", value="https://api.deepseek.com")
    temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.1)
    max_turns = st.sidebar.slider("Max Turns", min_value=1, max_value=10, value=5, step=1)

    # 初始化聊天模型
    llm = initialize_chat_model(api_key, base_url, temperature)
    
    # 如果没有提供API密钥，显示警告信息
    if not api_key:
        st.warning("Please enter your API key in the sidebar to start chatting.")
        return

    # 创建Streamlit聊天消息历史对象
    history = StreamlitChatMessageHistory(key="chat_messages")

    # 如果历史记录为空，清除聊天历史
    if not history.messages:
        clear_chat_history(history)

    # 显示聊天消息
    display_chat_messages(history)

    # 创建用户输入框
    user_query = st.chat_input("Type your message here...")
    
    # 如果聊天模型已初始化，处理用户输入
    if llm:
        handle_user_input(user_query, history, llm, max_turns)
        
    # 创建清除聊天按钮    
    if st.button("Clear Chat"):
        clear_chat_history(history)
        st.rerun()

if __name__ == "__main__":
    main()




