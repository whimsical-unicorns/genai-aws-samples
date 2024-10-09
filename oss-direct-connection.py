import json
import uuid
import boto3
from requests_aws4auth import AWS4Auth
from langchain.vectorstores import OpenSearchVectorSearch
from langchain.embeddings import BedrockEmbeddings
from opensearchpy import RequestsHttpConnection

# file aws_secrets.py with AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_OPEN_SEARCH_HOST_KEY
import aws_secrets

REGION_NAME = 'us-east-1'

session = boto3.Session(
    aws_access_key_id=aws_secrets.AWS_ACCESS_KEY_ID,
    aws_secret_access_key=aws_secrets.AWS_SECRET_ACCESS_KEY,
    region_name=REGION_NAME
)

service = 'aoss'
credentials = session.get_credentials()
awsauth = AWS4Auth(
    credentials.access_key,
    credentials.secret_key,
    REGION_NAME,
    service,
    session_token=credentials.token
)
opensearch_client = session.client('opensearch')

vectorstore = OpenSearchVectorSearch(
    index_name='test',
    embedding_function=lambda x: [],
    opensearch_url=[
        {
            'host': f"{aws_secrets.AWS_OPEN_SEARCH_HOST_KEY}.us-east-1.aoss.amazonaws.com",
            'port': 443
        }
    ],
    http_auth=awsauth,
    engine='faiss',
    timeout=3600,
    use_ssl=True,
    verify_certs=True,
    connection_class=RequestsHttpConnection,
)

print('Index exists "test":',  vectorstore.index_exists('test'))

# TODO: make bedrock client to use titan embedding model to embed and search
bedrock_client = session.client('bedrock')
bedrock_runtime = session.client('bedrock-runtime')

# available_models = bedrock_client.list_foundation_models()

# for model in available_models['modelSummaries']:
#   if 'amazon' in model['modelId']:
#     print(model)

model_id = 'amazon.titan-embed-text-v2:0'
prompt_data = """I really like eating pizza"""

# body = json.dumps({
#     "inputText": prompt_data,
# })

# response = bedrock_runtime.invoke_model(
#     body=body,
#     modelId=model_id,
#     accept='application/json',
#     contentType='application/json'
# )

# response_body = json.loads(response['body'].read())
# embedding = response_body.get('embedding')
# print('Embedding:', embedding)

bedrock_embeddings = BedrockEmbeddings(
    client=bedrock_runtime,
    model_id=model_id,
    region_name=REGION_NAME,
)

# print('Embedding:', bedrock_embeddings.embed_query(prompt_data))


# Store the embedding in the vector store

vectorstore2 = OpenSearchVectorSearch(
    index_name='test',
    embedding_function=bedrock_embeddings,
    opensearch_url=[
        {
            'host': f"{aws_secrets.AWS_OPEN_SEARCH_HOST_KEY}.us-east-1.aoss.amazonaws.com",
            'port': 443
        }
    ],
    http_auth=awsauth,
    engine='faiss',
    timeout=3600,
    use_ssl=True,
    verify_certs=True,
    connection_class=RequestsHttpConnection,
    kwargs={'vector_field': 'embeddings'} # not sure this is needed
)

run = input('Do you want to store the embedding in the vector store? (y/n): ')
if run == 'y':
    print('Storing embedding in vector store...')
    vectorstore2.add_texts(
        texts=[prompt_data],
        ids=[str(uuid.uuid4())],
        batch_size=1,
        refresh=True,
        vector_field='embeddings',
    )

# Search the vector store
similar_docs = vectorstore2.similarity_search_with_score(
    query=prompt_data,
    top_k=5,
    vector_field='embeddings',
)

print('Similar docs:', similar_docs)
# print details of documents; it's a list of tuples of document and float score; for each document print some more details
for doc, score in similar_docs:
    print(f"Document ID: {doc.id}, metadata: {doc.metadata}, text: {doc.page_content}")
    print('Score:', score)
    print()