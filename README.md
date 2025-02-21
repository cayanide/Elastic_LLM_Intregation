# 🔍 BetterQueryQuest

> The egg-cellent Elasticsearch query generator that combines the power of Elasticsearch with the wit of OpenAI's GPT-4. Developed by Cayanide Labs, this project brings you a quirky, fun, and highly functional tool to generate and execute complex queries against your Elasticsearch indexes.

## 📑 Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Customization](#customization)
- [Development Notes](#development-notes)
- [License](#license)
- [Credits](#credits)

## 🎯 Introduction
BetterQueryQuest is a Flask-based application that leverages OpenAI to generate dynamic Elasticsearch queries based on user input.creative code design, this tool is as entertaining as it is useful. Whether you're debugging or deploying, there's a touch of whimsy in every function!

## ✨ Features
- 🤖 **Dynamic Query Generation**: Utilize GPT-4 to create valid Elasticsearch queries tailored to your dataset
- ✅ **Index Schema Validation**: Automatically retrieve and analyze index schemas to build accurate queries
- 🌐 **Flask API Server**: Simple API endpoints to interact with the query generator
- 🔧 **Customizable**: Easily update index mappings, query samples, and file paths without exposing sensitive data

## 📋 Requirements
- Python 3.7 or higher
- Flask
- Elasticsearch Python Client
- OpenAI Python SDK
- Other dependencies as listed in `requirements.txt`

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/betterqueryquest.git
cd betterqueryquest
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Set up your environment variables:
Create a `.env` file in the project root and add your configuration:
```env
API_KEY=your_openai_api_key
ES_HOST=http://localhost:9200
ES_USERNAME=your_elasticsearch_username
ES_PASSWORD=your_elasticsearch_password
```

## 💻 Usage

### Running the Server
To run the Flask server in development mode:
```bash
python egg.py --serve --port 7001
```
- `--serve`: Runs the application in server mode
- `--port`: Specifies the port number (default is 7001)

Once running, access the main page at `http://localhost:7001`

### Command-Line Mode
You can also use the code in command-line mode for testing or automation purposes by omitting the `--serve` flag.

## 🔌 API Endpoints

### GET /
Renders the main response viewer page.

### GET /api/view-response
- Retrieves the query results from the last executed query
- Response: JSON object with the query results or an error message

### GET /api/query-json
- Retrieves the generated query JSON
- Response: JSON object containing the query

### POST /api/search-by-text
- Accepts a JSON payload with a `text` key representing the user's query
- Processes the query, selects relevant indexes, generates and executes the query
- Response: JSON object with the query details and results or error information

## ⚙️ Customization

### Updating Index Mappings & Query Samples
1. **Mapping Files**: Update the file paths in the `load_all_index_details()` function
2. **Supported Indexes**: Modify the `SUPPORTED_INDEXES` list to reflect your current indexes
3. **Query Examples**: Modify the `generate_query_samples()` function to include new working examples

## 🔧 Development Notes

### Logging
- All errors and major steps are logged in `query_quest.log`
- Real-time monitoring via console output

### OpenAI Integration
- Uses GPT-4 model for query generation
- Requires valid API key in `.env` file

### Elasticsearch Connection
- Validates connection before processing queries
- Ensure Elasticsearch instance is running and accessible

## 📄 License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 👏 Credits
- Developed by Cayanide Labs: Bringing you creative code solutions with a twist
- Special thanks to the open-source community for all the libraries that make this project possible

Enjoy using BetterQueryQuest and remember – even when your queries get scrambled, there's always a yolk of humor to brighten your debugging session! 🥚✨
