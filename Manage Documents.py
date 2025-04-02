import asyncio
import os
from itertools import chain
import streamlit as st
from google import genai
from anyio import sleep
from peewee import SQL, JOIN, NodeList
import json
import re
import fitz

from constants import CREATE_FACT_CHUNKS_SYSTEM_PROMPT
from db import DocumentInformationChunks, db, Documents

# Initialize Gemini AI
GENAI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GENAI_API_KEY:
    st.error("⚠️ Missing Gemini API key! Set `GEMINI_API_KEY` in environment variables.")
    st.stop()

client = genai.Client(api_key=GENAI_API_KEY)

st.set_page_config(page_title="Manage Documents")
st.title("📂 Manage Documents")

IDEAL_CHUNK_LENGTH = 4000

def delete_document(document_id: int):
    """Deletes a document from the database."""
    Documents.delete().where(Documents.id == document_id).execute()
    st.success(f"✅ Document {document_id} deleted!")
    st.rerun()
def get_embedding(text):
    """Generates embeddings using Gemini AI and ensures correct format."""
    try:
        # ✅ Correct API call for embedding generation
        response = client.models.embed_content(
            model="text-embedding-004",  # ✅ Correct model for embeddings
            contents=text  # ✅ `content` instead of `contents`
        )

        # ✅ Extract the embedding safely
        embedding = response.embeddings if hasattr(response, "embeddings") else None
        
        if embedding:
            return embedding[0].values  # Return valid embeddings
        else:
            st.error("⚠️ Failed to generate embeddings. Response missing 'embeddings'.")
            return None

    except Exception as e:
        st.error(f"⚠️ Embedding error: {e}")
        return None



async def generate_chunks(index: int, pdf_text_chunk: str):
    """Generates fact-based chunks from a PDF text segment."""
    #print(f"{index}   index    {pdf_text_chunk}")
    
    prompt = f"{CREATE_FACT_CHUNKS_SYSTEM_PROMPT}\n\n{pdf_text_chunk}"
    # print(prompt)
    for retry in range(5):
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[prompt]
            )
            response_text = response.candidates[0].content.parts[0].text
            json_match = re.search(r'```json\n(.*?)\n```', response_text, re.DOTALL)
            json_data = json_match.group(1)
            facts_dict = json.loads(json_data)
            facts = facts_dict.get("facts", [])
            print(f"Generated {len(facts)} facts for chunk {index}.")
            return facts
        except Exception as e:
            await sleep(1)
            print(f"Retry {retry+1}: Error generating facts: {e}")

    return []

# async def get_matching_tags(pdf_text: str):
    """Finds the most relevant tags for a given document text."""
    tags_result = Tags.select()
    tags = [tag.name.lower() for tag in tags_result]
    
    if not tags:
        return []

    prompt = f"Given these predefined tags: {', '.join(tags)}, find the most relevant ones for the following content:\n\n{pdf_text[:5000]}"
    
    for retry in range(5):
        try:
            response=client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[prompt]
            )
            matching_tags = [tag.strip().lower() for tag in response.tags.split(",")]
            matching_tag_ids = [tag.id for tag in tags_result if tag.name.lower() in matching_tags]
            print(f"Generated matching tags: {matching_tags}")
            return matching_tag_ids
        except Exception as e:
            await sleep(1)
            print(f"Retry {retry+1}: Error generating tags: {e}")

    return []

async def process_document(name: str, pdf_file: bytes):
    """Processes the document by extracting text, generating chunks, and tagging."""
    doc = fitz.open(stream=pdf_file, filetype="pdf")

    pdf_text = ""
    for page in doc:
        pdf_text += page.get_text("text") + "\n\n"  # Extract text page-wise
    
    doc.close() 
    pdf_text_chunks = [pdf_text[i:i + IDEAL_CHUNK_LENGTH] for i in range(0, len(pdf_text), IDEAL_CHUNK_LENGTH)]
    # print(pdf_text_chunks)
    # Run chunk generation & tag extraction concurrently
    document_information_chunks = await asyncio.gather(
        asyncio.gather(*[generate_chunks(index, chunk) for index, chunk in enumerate(pdf_text_chunks)]),
    )

    document_information_chunks = list(chain.from_iterable(document_information_chunks))

    # Insert into database
    with db.atomic() as transaction:
        document_id = Documents.insert(name=name).execute()

        # Insert document chunks
        DocumentInformationChunks.insert_many(
            [{"document_id": document_id, "chunk": chunk, "embedding": get_embedding(chunk)} for chunk in document_information_chunks]
        ).execute()

        

        transaction.commit()

    st.success(f"✅ Document '{name}' uploaded successfully with {len(document_information_chunks)} chunks!")

def upload_document(name: str, pdf_file: bytes):
    """Handles document upload using Streamlit's event model."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(process_document(name, pdf_file))

# UI: Upload Document Button
with st.expander("📤 Upload Document"):
    pdf_file = st.file_uploader("Upload a PDF", type="pdf")
    if pdf_file and st.button("Upload"):
        upload_document(pdf_file.name, pdf_file.getvalue())
        st.rerun()

# UI: Display Documents
# st.subheader("📄 Uploaded Documents")
# documents = Documents.select(
#     Documents.id,
#     Documents.name,
#     NodeList([
#         SQL('array_remove(array_agg('),
#         Tags.name,
#         SQL('), NULL)')
#     ]).alias("tags")
# ).join(DocumentTags, JOIN.LEFT_OUTER).join(Tags, JOIN.LEFT_OUTER).group_by(Documents.id).execute()

st.subheader("📄 Uploaded Documents")

# Fetch documents from the database
documents = Documents.select(Documents.id, Documents.name).execute()

# Display documents in UI
if documents:
    for document in documents:
        with st.container():
            st.write(f"📄 **{document.name}**")
            st.button("🗑 Delete", key=f"delete-{document.id}", on_click=lambda doc_id=document.id: delete_document(doc_id))
else:
    st.info("No documents uploaded yet. Upload one to get started!")

