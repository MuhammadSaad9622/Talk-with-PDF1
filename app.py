import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import docx2txt
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, hide_st_style, footer
import json
import os
from datetime import datetime, timedelta

# File to store recent responses
RESPONSES_FILE = 'recent_responses.json'

def get_pdf_text(pdf_docs):
    """Extract text from PDF files."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        num_pages = len(pdf_reader.pages)
        for i, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            if page_text:
                text += page_text
            # Update progress
            progress.progress((i + 1) / num_pages)
    return text

def get_docx_text(docx_docs):
    """Extract text from DOCX files."""
    text = ""
    for docx in docx_docs:
        doc_text = docx2txt.process(docx)
        if doc_text:
            text += doc_text
    return text

def get_text_chunks(text):
    """Split the extracted text into manageable chunks."""
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    """Create a vector store from text chunks using FAISS."""
    embeddings = OpenAIEmbeddings()
    if not text_chunks:  # Check if text_chunks is empty
        return None  # Return None if no text chunks
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    """Initialize the conversation chain with memory."""
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def generate_course(conversation_chain):
    """Generate course content including modules, lessons, and practice questions."""
    course_content = {}

    # Generate course modules and lessons
    modules = ["Introduction", "Key points", " objectives","Conclusion",]
    for module in modules:
        response = conversation_chain({
            'question': f"Generate a detailed {module} for a study course based on the given document and  add also sections where you think appropriate."
        })
        course_content[module] = response['chat_history'][-1].content

    # Generate practice questions and assessments
    response = conversation_chain({
        'question': " provide some links for practise tests and exams or refernece sites as well. "
    })
    course_content["Practice & Assessment"] = response['chat_history'][-1].content

    file_name = "_".join(st.session_state.uploaded_file_names)  # Create a unique file name
    save_recent_responses(course_content, file_name)
    return course_content

def summarize_content(text_chunks):
    """Summarize large volumes of content into concise study notes."""
    summarization_chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(),
        retriever=get_vectorstore(text_chunks).as_retriever(),
        memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    )

    summary = summarization_chain({
        'question': "Summarize the content into concise study notes and key points for revision."
    })
    return summary['chat_history'][-1].content

def display_course(course_content):
    """Display the generated course content on the Streamlit app."""
    st.header("Generated Course from Documents")
    for section, content in course_content.items():
        st.subheader(section)
        st.write(content)

def save_recent_responses(course_content, file_name):
    """Save recent responses to a file with the file name as key."""
    timestamp = datetime.now().isoformat()
    response_data = {
        'timestamp': timestamp,
        'course_content': course_content
    }
    
    if os.path.exists(RESPONSES_FILE):
        with open(RESPONSES_FILE, 'r') as file:
            all_responses = json.load(file)
    else:
        all_responses = {}

    all_responses[file_name] = response_data
    
    with open(RESPONSES_FILE, 'w') as file:
        json.dump(all_responses, file)

def load_recent_responses():
    """Load recent responses from a file if within 24 hours."""
    if os.path.exists(RESPONSES_FILE):
        with open(RESPONSES_FILE, 'r') as file:
            all_responses = json.load(file)
        
        valid_responses = {}
        for file_name, response in all_responses.items():
            try:
                # Ensure 'response' is a dictionary and contains a valid timestamp
                if isinstance(response, dict) and 'timestamp' in response:
                    timestamp = datetime.fromisoformat(response['timestamp'])
                    if datetime.now() - timestamp < timedelta(hours=24):
                        valid_responses[file_name] = response
            except (ValueError, TypeError) as e:
                print(f"Error processing response for {file_name}: {e}")
        
        return valid_responses
    return {}

def delete_recent_response(response_title):
    """Delete a specific recent response."""
    if os.path.exists(RESPONSES_FILE):
        with open(RESPONSES_FILE, 'r') as file:
            all_responses = json.load(file)
        
        if response_title in all_responses:
            del all_responses[response_title]
            with open(RESPONSES_FILE, 'w') as file:
                json.dump(all_responses, file)

def main():
    load_dotenv()
    st.set_page_config(page_title="Generate Course from Documents", page_icon="icon.png", layout="wide")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "uploaded_file_names" not in st.session_state:
        st.session_state.uploaded_file_names = []

    if "selected_response" not in st.session_state:
        st.session_state.selected_response = None

    global progress
    progress = st.sidebar.progress(0)

    with st.sidebar:
        st.subheader("Your Documents")
        doc_files = st.file_uploader(
            "Upload your documents in PDF or DOCX format and click 'Process'",
            accept_multiple_files=True,
            type=['pdf', 'docx']
        )

        if doc_files:
            st.session_state.uploaded_file_names = [doc.name for doc in doc_files]

        st.subheader("Recent Responses")
        recent_responses = load_recent_responses()

        with st.expander("View Recent Responses", expanded=True):
            if recent_responses:
                for response_title in recent_responses.keys():
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        if st.button(response_title, key=f"select_{response_title}"):
                            st.session_state.selected_response = response_title
                    with col2:
                        if st.button("X", key=f"delete_{response_title}"):
                            delete_recent_response(response_title)
                            st.session_state.selected_response = None
                            st.experimental_rerun()  # Rerun to refresh the list

            else:
                st.info("No recent course content available.")

        if st.button("Process"):
            if st.session_state.selected_response:
                st.error("A recent response is selected. Please deselect it to process new documents.")
            elif not doc_files:
                st.error("Please upload at least one document.")
            else:
                with st.spinner("Processing..."):
                    raw_text = ""
                    pdf_docs = [doc for doc in doc_files if doc.type == "application/pdf"]
                    docx_docs = [doc for doc in doc_files if doc.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]

                    if pdf_docs:
                        raw_text += get_pdf_text(pdf_docs)
                    if docx_docs:
                        raw_text += get_docx_text(docx_docs)

                    text_chunks = get_text_chunks(raw_text)
                    
                    if text_chunks:
                        vectorstore = get_vectorstore(text_chunks)
                        if vectorstore is not None:
                            st.session_state.conversation = get_conversation_chain(vectorstore)
                            st.success("Your data has been processed successfully!")
                        else:
                            st.error("No valid text found in the uploaded documents.")
                    else:
                        st.error("No valid text found in the uploaded documents.")
            progress.empty()

        if st.button("Clear Selection"):
            st.session_state.selected_response = None

    if st.session_state.selected_response:
        selected_response = st.session_state.selected_response
        recent_responses = load_recent_responses()
        if selected_response in recent_responses:
            st.header(f"Full Content for {selected_response}")
            course_content = recent_responses[selected_response].get('course_content', {})
            display_course(course_content)
        else:
            st.error("Selected response not available.")
    elif st.session_state.conversation:
        with st.spinner("Generating course content..."):
            file_name = "_".join(st.session_state.uploaded_file_names)
            course_content = generate_course(st.session_state.conversation)
            display_course(course_content)
    
    st.markdown(hide_st_style, unsafe_allow_html=True)
    st.markdown(footer, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
