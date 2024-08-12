from flask import Flask, request, jsonify, render_template
import requests
import json
import pandas as pd
from datetime import datetime
import os

app = Flask(__name__)



# Load your API keys from environment variables or configuration files
openai_api_key = os.environ.get('OPEN_API_KEY')
pinecone_api_key = os.environ.get('PINECONE_API_KEY')
azure_sas = os.environ.get('AZURE_API_KEY')
index_host = 'pm3-embedding-a569a40.svc.aped-4627-b74a.pinecone.io'

# Function to send data to Azure Blob
def send_to_blob(file_content, file_name):
    url = f"https://shiftdocs.blob.core.windows.net/logs/bestoutcome/{file_name}{azure_sas}"
    headers = {
        'Content-Type': 'application/json',
        'x-ms-blob-type': 'BlockBlob',
    }
    response = requests.put(url, headers=headers, data=file_content)
    print(response.text)

# Function to get embedding from OpenAI
def get_embedding(text):
    url = "https://api.openai.com/v1/embeddings"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_api_key}"
    }
    data = {
        "model": "text-embedding-ada-002",
        "input": text
    }
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    return response.json()['data'][0]['embedding']

# Function to query Pinecone
def query_pinecone(embedding, top_k=10):
    url = f"https://{index_host}/query"
    headers = {
        "Api-Key": pinecone_api_key,
        "Content-Type": "application/json",
        "X-Pinecone-API-Version": "2024-07"
    }
    data = {
        "vector": embedding,
        "topK": top_k,
        "includeValues": True,
        "includeMetadata": True
    }
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    return response.json()['matches']

# Function to generate GPT-4 response
def generate_response(prompt):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_api_key}"
    }
    data = {
        "model": "gpt-4",
        "messages": [
            {
                "role": "system",
                "content": "You are a very capable technical sales executive selling a product called PM3. You do not overstate functionality and you only ever answer based on the evidence you are provided. You write in British English at a GCSE reading level. You are filling in a Request for Proposal questionnaire and the client has asked the following question and is expecting a brief response:"
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    }
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    blob_json = response.json()
    blob_json['question'] = prompt
    send_to_blob(json.dumps(blob_json), datetime.now().strftime("%Y%m%d-%H:%M:%S") + '.json')
    return response.json()['choices'][0]['message']['content']

# Function to format response
def superscript_blue(text):
    import re
    return re.sub(r'\((.*?)\)', r'<sup><font color="blue">\1</font></sup>', text)

# Route to handle queries
@app.route('/query', methods=['POST'])
def handle_query():
    data = request.json
    query = data.get('question')
    if not query:
        return jsonify({"error": "No query provided"}), 400
    
    embedding = get_embedding(query)
    results = query_pinecone(embedding, top_k=10)
    df = pd.DataFrame([result['metadata'] for result in results])

    formatted_results = "\n".join([json.dumps(result['metadata'], indent=2) for result in results])
    prompt = f"""
{query}

Please use the relevant sections of the following document search results to formulate a very brief answer including references to documents if needs be in standard brackets.

{formatted_results}
"""
    response = generate_response(prompt)
    final_response = superscript_blue(response)
    
    return jsonify({"response": final_response, "results": df.to_dict(orient='records')})

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
