# 🤖 CrewAI Multi-Tool Collection

A comprehensive collection of AI-powered tools built with CrewAI framework, featuring document analysis, code review, text processing, API testing, and much more. This repository demonstrates the power of multi-agent AI systems for various automation tasks.

## 📋 Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Environment Setup](#environment-setup)
- [Tool Categories](#tool-categories)
- [Usage Examples](#usage-examples)
- [API Keys Setup](#api-keys-setup)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## ✨ Features

### 🔧 Code Analysis Tools
- **Bug Detection & Fixing**: Automatically detect and fix code bugs
- **Performance Optimization**: Analyze and optimize code performance
- **Security Vulnerability Scanner**: Identify and fix security issues
- **External API Detector**: Find all external API calls in code

### 📄 Document Processing Tools
- **Document Parser**: Extract structured data from PDFs using Google Document AI
- **Text Summarization**: Generate concise summaries of long texts
- **Document Comparison**: Compare two documents and highlight differences
- **Document Classification**: Categorize documents into predefined categories

### 🌐 API & Integration Tools
- **API Invocation Tool**: Validate and execute API requests with LLM validation
- **XML to JSON Converter**: Convert XML data to JSON format
- **Translation Bot**: Translate text between multiple languages

### ✍️ Text Processing Tools
- **Text Correction**: Fix spelling, grammar, and factual errors
- **Text Elaboration**: Expand text with relevant details and context
- **Persona Style Changer**: Rewrite text in different personas/styles

### 🔍 Computer Vision Tools
- **Face Verification**: Compare faces to determine if they belong to the same person

## 🔧 Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.8+**
- **pip** (Python package installer)
- **Git** (for cloning the repository)

### System Dependencies

#### For Face Verification Tool:
```bash
# Ubuntu/Debian
sudo apt-get install python3-opencv

# macOS
brew install opencv

# Windows
# OpenCV will be installed via pip
```

#### For Document Processing:
- Google Cloud Account (for Document AI)
- Google Cloud CLI (optional, for easier setup)

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/GundetiManoj/crewai_bots2tools.git
```

### 2. Create Virtual Environment:(https://docs.crewai.com/installation)

Install uv
On macOS/Linux:
Use curl to download the script and execute it with sh:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
If your system doesn’t have curl, you can use wget:

```bash
wget -qO- https://astral.sh/uv/install.sh | sh
```

On Windows:

Use irm to download the script and iex to execute it:


```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

If you run into any issues, refer to UV’s installation guide for more information.

```bash
Install CrewAI 🚀
```

Run the following command to install crewai CLI:


```bash
uv tool install crewai
```
If you encounter a PATH warning, run this command to update your shell:


```bash
uv tool update-shell
```
If you encounter the chroma-hnswlib==0.7.6 build error (fatal error C1083: Cannot open include file: 'float.h') on Windows, install (Visual Studio Build Tools)[https://visualstudio.microsoft.com/downloads/] with Desktop development with C++.

To verify that crewai is installed, run:


```bash
uv tool list
```
You should see something like:

```bash
crewai v0.102.0
```

### 3. Install Dependencies

```bash
# Install core dependencies
pip install -r requirements.txt

# Or install manually:
pip install crewai requests python-dotenv opencv-python transformers torch
pip install googletrans==4.0.0rc1 xmltodict litellm
pip install google-cloud-documentai google-generativeai
pip install difflib numpy pandas
```

### 4. Create requirements.txt

```txt
crewai>=0.28.0
requests>=2.31.0
python-dotenv>=1.0.0
opencv-python>=4.8.0
transformers>=4.35.0
torch>=2.0.0
googletrans==4.0.0rc1
xmltodict>=0.13.0
litellm>=1.0.0
google-cloud-documentai>=2.20.0
google-generativeai>=0.3.0
numpy>=1.24.0
pandas>=2.0.0
Pillow>=10.0.0
```

## 🔑 Environment Setup

### 1. Create .env File

Create a `.env` file in the root directory:

```bash
touch .env
```

### 2. Add API Keys

```env
# Required API Keys
GROQ_API_KEY=your_groq_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here

# Optional API Keys (for specific tools)
OPENAI_API_KEY=your_openai_api_key_here
HUGGINGFACE_API_KEY=your_huggingface_api_key_here

# Google Cloud Configuration (for Document AI)
GOOGLE_APPLICATION_CREDENTIALS=path/to/your/service-account-key.json
GOOGLE_CLOUD_PROJECT_ID=your_project_id
GOOGLE_CLOUD_LOCATION=us
DOCUMENT_AI_PROCESSOR_ID=your_processor_id
```

## 🔐 API Keys Setup

### 1. Groq API Key (Required)
1. Visit [Groq Console](https://console.groq.com/)
2. Sign up/Login
3. Navigate to API Keys section
4. Create new API key
5. Copy and add to `.env` file

### 2. Google Gemini API Key (For Document Processing)
1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create new API key
3. Copy and add to `.env` file

### 3. Google Cloud Setup (For Document AI)

#### Create Service Account:
```bash
# Install Google Cloud CLI
# Visit: https://cloud.google.com/sdk/docs/install

# Login to Google Cloud
gcloud auth login

# Create service account
gcloud iam service-accounts create crewai-docai \
    --display-name="CrewAI Document AI Service Account"

# Download service account key
gcloud iam service-accounts keys create ./google-cloud-key.json \
    --iam-account=crewai-docai@your-project-id.iam.gserviceaccount.com

# Grant necessary permissions
gcloud projects add-iam-policy-binding your-project-id \
    --member="serviceAccount:crewai-docai@your-project-id.iam.gserviceaccount.com" \
    --role="roles/documentai.apiUser"
```

#### Enable Document AI API:
```bash
gcloud services enable documentai.googleapis.com
```

#### Create Document AI Processor:
```bash
# Create a form parser processor
gcloud ai document-processors create \
    --location=us \
    --display-name="Form Parser" \
    --type=FORM_PARSER_PROCESSOR
```

## 🎯 Tool Categories

### 1. Code Analysis Tools

#### Bug Fixer
**Purpose**: Automatically detect and fix bugs in code

**Example Input**:
```
def divide_numbers(a, b):
    return a / b  # Division by zero bug

numbers = [1, 2, 3]
print(divide_numbers(10, 0))
```

**Example Output**:
```
## Bugs Found:
- Division by zero error when b=0
- No input validation

## Fixed Code:
def divide_numbers(a, b):
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

numbers = [1, 2, 3]
try:
    print(divide_numbers(10, 0))
except ValueError as e:
    print(f"Error: {e}")

## Explanation:
- Added zero division check
- Implemented proper error handling
- Used try-except for safe execution
```

#### Performance Optimizer
**Purpose**: Analyze and optimize code for better performance

**Example Input**:
```
def find_duplicates(arr):
    duplicates = []
    for i in range(len(arr)):
        for j in range(i+1, len(arr)):
            if arr[i] == arr[j] and arr[i] not in duplicates:
                duplicates.append(arr[i])
    return duplicates
```

**Example Output**:
```
## Performance Issues Found:
- O(n³) time complexity due to nested loops and 'in' operation
- Inefficient duplicate checking

## Optimized Version:
def find_duplicates(arr):
    seen = set()
    duplicates = set()
    for item in arr:
        if item in seen:
            duplicates.add(item)
        else:
            seen.add(item)
    return list(duplicates)

## Time Complexity Analysis:
- Original: O(n³)
- Optimized: O(n)
- Space complexity: O(n)

## Performance Improvement: ~1000x faster for large datasets
```

#### Security Scanner
**Purpose**: Identify and fix security vulnerabilities

**Example Input**:
```
import sqlite3
from flask import request

def login():
    username = request.form['username']
    password = request.form['password']
    
    # SQL Injection vulnerability
    query = f"SELECT * FROM users WHERE username='{username}' AND password='{password}'"
    conn = sqlite3.connect('db.sqlite')
    result = conn.execute(query).fetchone()
    return result
```

**Example Output**:
```
## Security Issues Found:
- SQL Injection vulnerability in query construction
- No input validation
- Plain text password handling
- Missing error handling

## Secure Version:
import sqlite3
import hashlib
from flask import request

def login():
    username = request.form.get('username', '').strip()
    password = request.form.get('password', '')
    
    # Input validation
    if not username or not password:
        return None
    
    # Use parameterized queries
    query = "SELECT * FROM users WHERE username = ? AND password_hash = ?"
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    
    try:
        conn = sqlite3.connect('db.sqlite')
        result = conn.execute(query, (username, password_hash)).fetchone()
        conn.close()
        return result
    except sqlite3.Error:
        return None

## Security Fixes:
- Parameterized queries prevent SQL injection
- Password hashing for secure storage
- Input validation and sanitization
- Proper error handling
```

### 2. Document Processing Tools

#### Document Parser
**Purpose**: Extract structured data from PDF documents

**Example Input**: `invoice.pdf` (PDF file)

**Example Output**:
```json
{
  "personal_details": {
    "name": "John Smith",
    "phone": "+1-555-0123",
    "email": "john.smith@email.com",
    "address": "123 Main St, New York, NY 10001"
  },
  "text_with_coords":[],
  "key_value_pairs":[],
  "tables":[]
}
```

#### Text Summarization
**Purpose**: Generate concise summaries of long texts

**Example Input**:
```
Artificial Intelligence has revolutionized the way we approach problem-solving across various industries. From healthcare to finance, AI technologies are enabling unprecedented levels of automation and efficiency. Machine learning algorithms can analyze vast amounts of data, identify patterns, and make predictions with remarkable accuracy. However, the rapid advancement of AI also raises important ethical questions about privacy, job displacement, and algorithmic bias that society must address.
```

**Example Output**:
```
AI revolutionizes problem-solving across industries like healthcare and finance through automation and efficiency. Machine learning analyzes data and makes accurate predictions, but raises ethical concerns about privacy, job displacement, and bias.
```

### 3. Text Processing Tools

#### Translation
**Purpose**: Translate text between different languages

**Example Input**:
```
Text: "Hello, how are you today? I hope you're having a great day!"
Target Language: "es" (Spanish)
```

**Example Output**:
```
¡Hola, ¿cómo estás hoy? ¡Espero que tengas un gran día!
```

#### Text Correction
**Purpose**: Fix spelling, grammar, and factual errors

**Example Input**:
```
The sun rotates arond the earth and it's distence is 93 milions miles. This proces takes 365 days to compleat.
```

**Example Output**:
```
The Earth rotates around the sun and its distance is 93 million miles. This process takes 365 days to complete.
```

#### Text Elaboration
**Purpose**: Expand text with relevant details and context

**Example Input**:
```
Text: "Tell me about Republic Day"
Word Limit: 200
```

**Example Output**:
```
Republic Day is a significant national holiday celebrated in India on January 26th every year. This day commemorates the adoption of the Indian Constitution in 1950, marking India's transition from a British dominion to a sovereign republic. The Constitution came into effect on this date, replacing the Government of India Act 1935. The main celebration takes place in New Delhi at Rajpath, featuring a grand parade showcasing India's military might, cultural diversity, and technological achievements. The President of India hoists the national flag and delivers an address to the nation. Various states present their cultural heritage through colorful tableaux, and the armed forces display their latest equipment and capabilities. The celebration concludes with a spectacular air show by the Indian Air Force, making it one of the most important patriotic events in the country.
```

#### Persona Style Changer
**Purpose**: Rewrite text in different personas or styles

**Example Input**:
```
Text: "Explain the methodology of solving Dijkstra's algorithm"
Persona: "A 7th semester student of IIT Bombay"
```

**Example Output**:
```
Hey! So Dijkstra's algorithm - this is actually pretty cool stuff we covered in our Data Structures and Algorithms course. 

The methodology is quite systematic:

1. **Initialization**: Start with source vertex, set its distance to 0, all others to infinity. Create a priority queue.

2. **Greedy Selection**: Always pick the unvisited vertex with minimum distance (this is the greedy part our prof emphasized).

3. **Relaxation**: For the selected vertex, update distances to all its neighbors if we found a shorter path.

4. **Repeat**: Keep doing this until we've visited all vertices or found our target.

The key insight our TA mentioned is the optimal substructure property - if we have the shortest path to vertex A, and A connects to B, then the shortest path to B through A is just distance[A] + weight(A,B).

Time complexity is O((V+E)logV) with a binary heap, which we derived in class. Pretty elegant algorithm, and the proof of correctness using the cut property is quite beautiful too!

Pro tip from experience: Always remember to handle negative weights - Dijkstra fails there, need Bellman-Ford instead.
```

### 4. API Tools

#### API Invoker
**Purpose**: Validate and execute API requests

**Example Input**:
```
URL: "https://jsonplaceholder.typicode.com/posts/1"
Method: "GET"
```

**Example Output**:
```
Status: 200
Response: {
  "userId": 1,
  "id": 1,
  "title": "sunt aut facere repellat provident occaecati excepturi optio reprehenderit",
  "body": "quia et suscipit\nsuscipit recusandae consequuntur expedita et cum\nreprehenderit molestiae ut ut quas totam\nnostrum rerum est autem sunt rem eveniet architecto"
}
```

#### XML to JSON Converter
**Purpose**: Convert XML data to JSON format

**Example Output**:
```json
{
  "library": {
    "book": [
      {
        "@id": "1",
        "title": "Python Programming",
        "author": "John Doe",
        "price": "29.99",
        "category": "Programming"
      },
      {
        "@id": "2",
        "title": "Data Science Handbook",
        "author": "Jane Smith",
        "price": "39.99",
        "category": "Data Science"
      }
    ]
  }
}
```

#### External API Detector
**Purpose**: Find all external API calls in code

**Example Input**:
```python
import requests

response = requests.get("https://api.example.com/v1/users")
data = response.json()

# Another API Call
fetch_data = requests.post("https://service.example.org/api/upload", json={"file": "data.txt"})
webhook_url = "https://hooks.slack.com/services/webhook"
```

**Example Output**:
```
**LLM Detected URLs:**
- https://api.example.com/v1/users
- https://service.example.org/api/upload  
- https://hooks.slack.com/services/webhook

**Regex Detected URLs:**
- https://api.example.com/v1/users
- https://service.example.org/api/upload
- https://hooks.slack.com/services/webhook

**Analysis:**
3 external API endpoints detected in the code snippet.
```

### 5. Computer Vision Tools

#### Face Verification
**Purpose**: Compare faces to determine if they belong to the same person

**Example Input**:
```
Image 1: "/path/to/person1_photo1.jpg"
Image 2: "/path/to/person1_photo2.jpg"
```

**Example Output**:
```
✅ MATCH: Similarity score: 23.45 - Faces likely belong to the same person.

Analysis Details:
- Face detection: Successful for both images
- Similarity threshold: <50 (indicates match)
- Confidence level: High
- Recommendation: Same person detected
```

**Example Input (Different People)**:
```
Image 1: "/path/to/person1.jpg"
Image 2: "/path/to/person2.jpg"
```

**Example Output**:
```
❌ NO MATCH: Difference score: 78.92 - Faces appear to be different people.

Analysis Details:
- Face detection: Successful for both images
- Similarity threshold: >50 (indicates different people)
- Confidence level: High
- Recommendation: Different people detected
```

### 6. Document Processing Tools (Additional)

#### Document Comparison
**Purpose**: Compare two documents and highlight differences

**Example Input**:
```
Document 1: "The quick brown fox jumps over the lazy dog. This is a sample text for testing purposes."

Document 2: "The quick red fox runs over the sleeping dog. This is a sample document for testing purposes."
```

**Example Output**:
```
Document Comparison Results:
- Similarity: 75.5%
- Total Changes: 3

Detailed Changes:
CHANGED: 'brown' → 'red'
CHANGED: 'jumps' → 'runs'  
CHANGED: 'lazy' → 'sleeping'
CHANGED: 'text' → 'document'
```

#### Document Classification
**Purpose**: Categorize documents into predefined categories

**Example Input**:
```
"The stock market continues to fluctuate as investors await the Federal Reserve's next decision on interest rates. Experts predict that the economy will continue to face challenges, but opportunities still exist for long-term investors seeking portfolio diversification."
```

**Example Output**:
```
Category: Finance (Confidence: 94.2%)

Classification Details:
- Primary Category: Finance
- Secondary Categories: News (87.3%), Technology (12.1%)
- Keywords Detected: stock market, Federal Reserve, interest rates, economy, investors
- Classification Model: BART-Large-MNLI
```

### System Requirements

- **Minimum**: 4GB RAM, 2GB storage
- **Recommended**: 8GB RAM, 5GB storage
- **For Document AI**: Additional 2GB for model cache

## 🙏 Acknowledgments

- **CrewAI Team** for the amazing framework
- **Google Cloud** for Document AI services
- **Groq** for high-performance inference
- **Hugging Face** for transformer models
- **OpenCV** community for computer vision tools


**Built with ❤️ using CrewAI Framework**

Made for developers, by developers. Transform your workflow with AI-powered automation tools.
