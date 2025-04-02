import asyncio #used
from typing import Literal, Optional, TypedDict, Union
import streamlit as st
from constants import RESPOND_TO_MESSAGE_SYSTEM_PROMPT
from db import DocumentInformationChunks, db
from peewee import SQL
from anyio import sleep
import os
from google import genai
# Initialize Gemini AI
GENAI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GENAI_API_KEY:
    raise ValueError("⚠️ GEMINI_API_KEY is not set in environment variables.")
client = genai.Client(api_key=GENAI_API_KEY)

st.set_page_config(page_title="PASSPORT SEVA")
st.title("PASSPORT SEVA")

# Define message structure
class Message(TypedDict):
    role: Union[Literal["user"], Literal["assistant"]]
    content: str
    references: Optional[list[str]]

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state["messages"] = []

def push_message(message: Message):
    """Add a message to the session state."""
    st.session_state["messages"].append(message)

def get_embedding(text):
    """Generate embedding for a given text."""
    if not text.strip():
        st.warning("⚠️ Cannot generate embedding for empty content.")
        return None

    try:
        response = client.models.embed_content(
            model="text-embedding-004",
            contents=text,  
        )
        embedding = response.embeddings if hasattr(response, "embeddings") else None
        
        if embedding:
            return embedding[0].values  # Return valid embeddings
        else:
            st.error("⚠️ Failed to generate embeddings. Response missing 'embeddings'.")
            return None
    except Exception as e:
        st.error(f"⚠️ Error generating embedding: {e}")
        return None

async def send_message(input_message: str):
    """Process user message and generate AI response."""
    related_document_information_chunks: list[str] = []

    # Fetch related document chunks from DB
    try:
        with db.atomic() as transaction:
            embedding = get_embedding(input_message)
            if not embedding:
                return
            embedding_str = f"[{', '.join(map(str, embedding))}]"
            result = (
                DocumentInformationChunks.select()
                .order_by(SQL(f"embedding <-> %s", (embedding_str,)))
                .limit(5)
                .execute()
            )

            for row in result:
                related_document_information_chunks.append(row.chunk)
            transaction.commit()
    except Exception as e:
        st.error(f"⚠️ Database error: {e}")
        return

    # Save user message
    push_message({
        "role": "user",
        "content": input_message,
        "references": related_document_information_chunks
    })

    chat_history = st.session_state["messages"][-5:]
    formatted_history = "\n".join(
    [f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history]
    )
    # Try generating AI response
    total_retries = 0
    while total_retries < 5:
        try:
            # model = genai.GenerativeModel("gemini-1.5-pro")
            #prompt = f"{RESPOND_TO_MESSAGE_SYSTEM_PROMPT.replace('{{knowledge}}', '\n'.join([f'{i+1}. {chunk}' for i, chunk in enumerate(related_document_information_chunks)]))}\n\nUser: {input_message}"
            prompt = f"""
            {RESPOND_TO_MESSAGE_SYSTEM_PROMPT.replace('{{knowledge}}', '\n'.join([f'{i+1}. {chunk}' for i, chunk in enumerate(related_document_information_chunks)]))}

            **Chat History:**
            {formatted_history}

            **User:** {input_message}
            """
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[prompt]
            )

            if not response or not response.text.strip():
                st.error("⚠️ No response generated. Try again.")
                break
            push_message({
                "role": "assistant",
                "content": response.text,
                "references": None
            })
            print(f"Generated response: {response.text}")
            break
        except Exception as e:
            total_retries += 1
            await sleep(1)
            print(f"Retry {total_retries}: Error generating response: {e}")

    st.rerun()

# Display chat messages
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message["references"]:
            with st.expander("References"):
                for reference in message["references"]:
                    st.write(reference)

# Handle user input
input_message = st.chat_input("Ask your query regarding passports")
if input_message:
    asyncio.run(send_message(input_message))  # ✅ Proper event loop usage
