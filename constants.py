CREATE_FACT_CHUNKS_SYSTEM_PROMPT = "\n\n".join([
    "You are an expert text analyzer who can take any text, analytze it, and create multiple facts from it. OUTPUT SHOULD BE STRICTLY IN THIS JSON FORMAT:",
    "{\"facts\": [\"fact 1\", \"fact 2\", \"fact 3\"]}","The text you need to analyse is"
])

RESPOND_TO_MESSAGE_SYSTEM_PROMPT = "\n\n".join([
    "You are a chatbot who has some specific set of knowledge and you will be asked questions on that given the knowledge.",
    "Don't make up information and don't answer until and unless you have knowledge to back it.",
    "Knowledge you have:",
    "{{knowledge}}"
])
