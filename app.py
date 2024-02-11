# !pip install -q ctransformers langchain
import streamlit as st
from langchain.llms import CTransformers
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

st.title("How can I help you today?")
main_placeholder = st.empty()

query = main_placeholder.text_input("", placeholder="Type here")

main_placeholder_again = st.empty()

clicked_button = main_placeholder_again.button("Submit")

if clicked_button:
    main_placeholder_again.text("Processing.........")
    llm = CTransformers(model="TheBloke/Llama-2-7B-Chat-GGML", model_file = 'llama-2-7b-chat.ggmlv3.q2_K.bin', callbacks=[StreamingStdOutCallbackHandler()])

    template = """
    [INST] <<SYS>>
    You are a helpful, respectful and honest assistant. Your answers are always brief.
    <</SYS>>
    {text}[/INST]
    """

    prompt = PromptTemplate(template=template, input_variables=["text"])

    llm_chain = LLMChain(prompt=prompt, llm=llm)
    response = llm_chain.run(query)
    main_placeholder_again.write(response)

