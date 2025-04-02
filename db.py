import os
from pgvector.peewee import VectorField
from peewee import PostgresqlDatabase, Model, TextField, ForeignKeyField, IntegrityError

# Initialize Database Connection
POSTGRES_DB_NAME = os.getenv("POSTGRES_DB_NAME")
POSTGRES_DB_HOST = os.getenv("POSTGRES_DB_HOST")
POSTGRES_DB_PORT = os.getenv("POSTGRES_DB_PORT")
POSTGRES_DB_USER = os.getenv("POSTGRES_DB_USER")
POSTGRES_DB_PASSWORD = os.getenv("POSTGRES_DB_PASSWORD")

if not all([POSTGRES_DB_NAME, POSTGRES_DB_HOST, POSTGRES_DB_PORT, POSTGRES_DB_USER, POSTGRES_DB_PASSWORD]):
    raise ValueError("⚠️ Missing one or more required PostgreSQL environment variables.")

db = PostgresqlDatabase(
    POSTGRES_DB_NAME,
    host=POSTGRES_DB_HOST,
    port=POSTGRES_DB_PORT,
    user=POSTGRES_DB_USER,
    password=POSTGRES_DB_PASSWORD,
)

# Define Models
class BaseModel(Model):
    """Base model to set the database."""
    class Meta:
        database = db

class Documents(BaseModel):
    name = TextField(unique=True)  # Ensure unique document names

    class Meta:
        db_table = 'documents'



class DocumentInformationChunks(BaseModel):
    document = ForeignKeyField(Documents, backref="document_information_chunks", on_delete='CASCADE')
    chunk = TextField()
    embedding = VectorField(dimensions=768)  # Ensure correct vector dimensions

    class Meta:
        db_table = 'document_information_chunks'

# Connect and Create Tables
db.connect()

try:
    db.create_tables([Documents, DocumentInformationChunks])
except IntegrityError as e:
    print(f"⚠️ Error creating tables: {e}")

# Ensure `pgvector` extension is enabled
db.execute_sql("CREATE EXTENSION IF NOT EXISTS vector;")

# Create HNSW Index for Efficient Searches
# db.execute_sql("""
#     CREATE INDEX IF NOT EXISTS document_chunks_embedding_hnsw
#     ON document_information_chunks
#     USING hnsw (embedding vector_l2_ops);
# """)

# # Function to Set DiskANN Query Rescore (If Needed)
# def set_diskann_query_rescore(query_rescore: int):
#     """Sets the diskann query rescore value for tuning retrieval efficiency."""
#     db.execute_sql("SET diskann.query_rescore = %s", (query_rescore,))
