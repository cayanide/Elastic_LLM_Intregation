"""
BetterQuery: The Quirky Query Conjurer
========================================
Welcome to BetterQuery – where Elasticsearch meets OpenAI with a side of egg-ceptional humor.
Warning: This code is sprinkled with puns, hidden easter eggs, and clever nods to hard-working devs.
Developed with egg-ceptional passion by Cayanide Labs.

INSTRUCTIONS:
  - To update query examples, edit the function `generate_query_examples()` with your own field mappings.
  - To change index mappings, update the file paths in the search endpoint and the helper function `load_all_index_details()`.
  - Replace all "path/to/generic_*.txt" placeholders with your actual mapping file paths.
  - Enjoy the whimsical comments as you explore this code!
"""

# Importing our essential ingredients for this magical query cauldron.
import flask
from flask import Flask, render_template, jsonify, request  # Serving up views and JSON delights.
from elasticsearch import Elasticsearch, BadRequestError  # Our trusty connector to Elasticsearch.
from openai import OpenAI  # OpenAI: Where queries turn into data masterpieces.
import json
import logging
import argparse
import os
import re
from flask_cors import CORS  # Because boundaries are meant to be crossed!
from dotenv import load_dotenv
import concurrent.futures
from datetime import datetime
from typing import Dict, List, Any, Optional, Set
from difflib import get_close_matches
import numpy as np
import time

# Load our secret sauce (environment variables) without spilling any beans.
load_dotenv()
API_KEY = os.getenv("API_KEY") #Open AI API KEY
ES_HOST = os.getenv("ES_HOST", "http://localhost:9200")
ES_USERNAME = os.getenv("ES_USERNAME", "elastic")
ES_PASSWORD = os.getenv("ES_PASSWORD")

# Supported indexes are now generic – no sensitive info here!
SUPPORTED_INDEXES = ["generic_index_a", "generic_index_b", "generic_index_c"]

# Set up logging to capture our brilliant moments (and errors) in style.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('query_processor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("BetterQueryLogger")

# Initialize Elasticsearch and OpenAI clients – let's crack on!
try:
    es = Elasticsearch(ES_HOST, basic_auth=(ES_USERNAME, ES_PASSWORD), verify_certs=False)
except Exception as e:
    logger.error(f"Elasticsearch initialization failed: {str(e)}")
    es = None

client = OpenAI(api_key=API_KEY)

# Type hints for our magical data recipes.
JsonDict = Dict[str, Any]
IndexSchemas = Dict[str, JsonDict]

# Validate connection – because nothing is more egg-straordinary than a live ping!
def validate_es_connection() -> None:
    if not es or not es.ping():
        raise ConnectionError("Failed to connect to Elasticsearch")

# Create our Flask app – hotter than a sunny-side up egg!
app = Flask(__name__, template_folder='templates')
CORS(app)  # Open your arms – no CORS restrictions here!

@app.route('/')
def index():
    # Render the main response viewer template.
    return render_template('response_viewer.html')

@app.route('/api/view-response')
def get_response_data():
    """Retrieve query results with a dash of humor."""
    try:
        with open('query_results.json', 'r') as f:
            results = json.load(f)
        return jsonify({"success": True, "results": results}), 200
    except FileNotFoundError:
        return jsonify({"success": False, "error": "Results file not found"}), 404
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/query-json', methods=['GET'])
def get_query_json():
    """Retrieve the generated query JSON – no yolk, all skill."""
    try:
        with open('generated_query.json', 'r') as f:
            query_data = json.load(f)
        return jsonify({"success": True, "query": query_data}), 200
    except FileNotFoundError:
        return jsonify({"success": False, "error": "Generated query file not found"}), 404
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/search-by-text', methods=['POST'])
def search_by_text():
    """Search endpoint where query magic happens—scrambled to perfection!"""
    try:
        data = request.get_json()
        if not data:
            logger.error("No JSON data received")
            return jsonify({"error": "No data provided"}), 400

        user_query = data.get('text', '').strip()
        if not user_query:
            logger.error("Empty query received")
            return jsonify({"error": "Empty query"}), 400

        logger.info(f"Processing search request for query: {user_query}")

        # Validate our connection before getting scrambled.
        try:
            validate_es_connection()
        except ConnectionError as e:
            logger.error(f"Connection error: {str(e)}")
            return jsonify({"error": "Database connection error"}), 503

        # Fetch the blueprints for our data mansion.
        try:
            schemas = get_comprehensive_schemas(SUPPORTED_INDEXES)
            if not schemas:
                logger.error("No schemas retrieved")
                return jsonify({"error": "Failed to retrieve index schemas"}), 500
        except Exception as e:
            logger.error(f"Schema retrieval failed: {str(e)}")
            return jsonify({"error": "Schema retrieval failed"}), 500

        # Select target indexes based on the query.
        selected_indexes = select_indexes(user_query, schemas)
        if not selected_indexes:
            logger.error("No relevant indexes found for query")
            return jsonify({"error": "No relevant indexes found for the query"}), 400

        logger.info(f"Selected indexes: {selected_indexes}")

        best_query = None
        query_errors = []

        # Assemble index-specific details – think of it as mixing secret ingredients!
        for index in selected_indexes:
            try:
                logger.info(f"Generating query for index: {index}")
                target_index = index

                # Map generic index to its corresponding mapping file(s).
                if target_index == "generic_index_a":
                    file_a1 = "path/to/generic_mapping_file_a1.txt"  # UPDATE with your own file path
                    file_a2 = "path/to/generic_mapping_file_a2.txt"  # UPDATE with your own file path
                    mapping_file_content = ""
                    try:
                        with open(file_a1, 'r') as f1:
                            mapping_file_content += f1.read()
                        with open(file_a2, 'r') as f2:
                            mapping_file_content += f2.read()
                        logger.info(f"Successfully merged mapping files for {target_index}")
                    except Exception as e:
                        logger.error(f"Failed to merge files for {target_index}: {str(e)}")
                        mapping_file_content = ""
                elif target_index == "generic_index_b":
                    mapping_file_content = "path/to/generic_mapping_file_b.txt"  # UPDATE with your own file path
                elif target_index == "generic_index_c":
                    mapping_file_content = "path/to/generic_mapping_file_c.txt"  # UPDATE with your own file path
                else:
                    mapping_file_content = None

                # Load index details from mapping file if available.
                index_details = ""
                if mapping_file_content:
                    try:
                        with open(mapping_file_content, 'r') as f:
                            index_details = f.read()
                    except Exception as e:
                        logger.error(f"Failed to load mapping details for {target_index}: {str(e)}")
                else:
                    logger.warning(f"No mapping details file mapped for {target_index}")

                logger.info(f"Loaded mapping details for {target_index}")

                query_prompt = generate_query_prompt(user_query, schemas[index], target_index, index_details)
                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": query_prompt}],
                    temperature=0.1
                )

                logger.info(f"Response from OpenAI: {response}")

                raw_query = json.loads(response.choices[0].message.content.strip())
                logger.info(f"Generated raw query for {target_index}")

                clean_query = sanitize_query(raw_query, schemas[index])
                logger.info(f"Sanitized query for {target_index}")

                with open('generated_query.json', 'w') as f:
                    json.dump(clean_query, f, indent=2)
                print("\nGenerated query saved to 'generated_query.json'")

                if test_query(target_index, clean_query):
                    best_query = (target_index, clean_query)
                    logger.info(f"Found valid query for {target_index}")
                    break
                else:
                    error_msg = f"Query returned no results for {target_index}"
                    logger.warning(error_msg)
                    query_errors.append(error_msg)

            except Exception as e:
                error_msg = f"Query failed for {index}: {str(e)}"
                logger.warning(error_msg)
                query_errors.append(error_msg)
                continue

        if not best_query:
            error_details = "\n".join(query_errors) if query_errors else "No valid queries generated"
            logger.error(f"No valid queries generated. Errors:\n{error_details}")
            return jsonify({"error": "No valid queries generated", "details": query_errors}), 400

        # Execute the best query found.
        try:
            target_index, final_query = best_query
            results = execute_query(final_query, [target_index])
            processed_results = results.body
            with open('query_results.json', 'w') as f:
                json.dump(processed_results, f, indent=2)
            print("Query results saved to 'query_results.json'")
            return jsonify({
                "success": True,
                "index": target_index,
                "query": final_query,
                "results": processed_results
            }), 200
        except Exception as e:
            logger.error(f"Query execution failed: {str(e)}")
            return jsonify({"error": "Query execution failed"}), 500

    except Exception as e:
        logger.exception(f"Search endpoint error: {str(e)}")
        return jsonify({"error": "Search failed", "details": str(e)}), 500

def get_comprehensive_schemas(indexes: List[str]) -> IndexSchemas:
    """Fetch and validate schemas for supported indexes – our data cookbook for perfect query soufflés."""
    schemas: IndexSchemas = {}
    for index in indexes:
        try:
            logger.info(f"Fetching schema for index: {index}")
            mapping = es.indices.get_mapping(index=index)
            logger.info(f"Schema fetched for index: {index}")
            properties = mapping.get(index, {}).get('mappings', {}).get('properties', {})

            # Extract field metadata.
            date_fields = set()
            numeric_fields = set()
            keyword_fields = set()
            primary_keys = set()

            def analyze_fields(props: Dict, prefix: str = "") -> None:
                for field, config in props.items():
                    full_path = f"{prefix}{field}" if prefix else field
                    field_type = config.get('type')
                    if field_type == 'date':
                        date_fields.add(full_path)
                    elif field_type in ('long', 'integer', 'double', 'float'):
                        numeric_fields.add(full_path)
                    if field_type == 'text' and 'keyword' in config.get('fields', {}):
                        keyword_fields.add(full_path)
                    if 'properties' in config:
                        analyze_fields(config['properties'], f"{full_path}.")

            analyze_fields(properties)

            schemas[index] = {
                "index": index,
                "properties": properties,
                "field_types": {k: v.get('type', 'object') for k, v in properties.items()},
                "nested_paths": find_nested_paths(properties),
                "date_fields": list(date_fields),
                "numeric_fields": list(numeric_fields),
                "keyword_fields": list(keyword_fields),
                "primary_keys": list(primary_keys),
                "field_metadata": {
                    field: {
                        "type": config.get('type'),
                        "analyzer": config.get('analyzer'),
                        "fields": config.get('fields', {}),
                    }
                    for field, config in properties.items()
                }
            }

            if 'settings' in mapping[index]:
                settings = mapping[index]['settings']
                schemas[index].update({
                    "table_name": settings.get('index.table_name', index),
                    "table_description": settings.get('index.description', ''),
                    "default_analyzer": settings.get('index.analysis.analyzer.default', {})
                })

        except Exception as e:
            logger.error(f"Schema fetch failed for {index}: {str(e)}")
            schemas[index] = {"error": str(e)}

    return schemas

def find_nested_paths(properties: JsonDict, path: str = "") -> List[str]:
    """Recursively find all nested field paths – an egg-stra scavenger hunt."""
    paths = []
    for field, config in properties.items():
        current_path = f"{path}.{field}" if path else field
        if config.get('type') == 'nested' and 'properties' in config:
            paths.append(current_path)
            paths.extend(find_nested_paths(config['properties'], current_path))
    return paths

def select_indexes(query: str, schemas: IndexSchemas) -> List[str]:
    """Select valid indexes based on the user query and schema details – our in-house egg-head magic."""
    try:
        logger.info(f"Processing query: {query}")
        if not schemas:
            logger.error("No valid schemas provided")
            return []
        index_details = load_all_index_details()
        logger.info(f"Index details loaded: {index_details}")
        prompt = generate_index_selection_prompt(query, schemas, index_details)
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        response_text = response.choices[0].message.content.strip()
        logger.info(f"AI response for index selection: {response_text}")
        try:
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if not json_match:
                logger.error("No JSON array found in response")
                raise ValueError("No JSON array found in response")
            ai_selected = json.loads(json_match.group().replace('`', ''))
            logger.info(f"Parsed AI selected indexes: {ai_selected}")
            return ai_selected
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {str(e)}")
            return []

        valid_indexes = []
        query_lower = query.lower()

        def generate_field_variations(field: str) -> Set[str]:
            variations = {
                field.lower(),
                field.lower().replace('_', ' '),
                field.lower().replace('_', ''),
                f"{field.lower()}_id",
                f"{field.lower()} id",
                f"{field.lower()}id",
                field.lower().replace('_code', ''),
                field.lower().replace('_date', ''),
                field.lower().replace('_status', '')
            }
            parts = field.lower().split('_')
            camel_case = parts[0] + ''.join(p.capitalize() for p in parts[1:])
            variations.add(camel_case)
            return variations

        for index in ai_selected:
            if index not in SUPPORTED_INDEXES:
                logger.warning(f"Index {index} not supported")
                continue
            if index not in schemas:
                logger.warning(f"Index {index} not found in schemas")
                continue
            if 'error' in schemas[index]:
                logger.warning(f"Index {index} schema error: {schemas[index]['error']}")
                continue
            if 'field_types' not in schemas[index]:
                logger.warning(f"Index {index} missing field_types")
                continue
            field_variations = set()
            for field in schemas[index]['field_types'].keys():
                field_variations.update(generate_field_variations(field))
            if any(variation in query_lower for variation in field_variations):
                valid_indexes.append(index)
                logger.info(f"Added valid index: {index}")
                continue

            common_terms = {
                'generic detail': ["generic_index_a"],
                'log': ['generic_index_b'],
                'note': ['generic_index_c']
            }
            for term, related_indexes in common_terms.items():
                if term in query_lower and index in related_indexes:
                    valid_indexes.append(index)
                    logger.info(f"Added valid index based on term '{term}': {index}")
                    break

        if not valid_indexes:
            logger.info("No valid indexes from primary validation, trying schema-based fallback")
            for index, schema in schemas.items():
                if 'error' in schema or 'field_types' not in schema:
                    continue
                schema_metadata = [
                    schema.get('table_description', '').lower(),
                    schema.get('table_name', '').lower(),
                    index.lower()
                ]
                if any(term in ' '.join(schema_metadata) for term in query_lower.split()):
                    valid_indexes.append(index)
                    logger.info(f"Added index from schema metadata match: {index}")

        final_indexes = list(dict.fromkeys(valid_indexes))[:3]
        logger.info(f"Final selected indexes: {final_indexes}")
        return final_indexes

    except Exception as e:
        logger.exception(f"Index selection failed: {str(e)}")
        return []

def generate_index_selection_prompt(query: str, schemas: IndexSchemas, index_details) -> str:
    """Generate a structured prompt for index selection using generic index names."""
    schema_summary = {
        index: {
            "fields": list(schema['field_types'].keys()),
            "nested_paths": schema.get('nested_paths', []),
            "field_count": len(schema['field_types'])
        }
        for index, schema in schemas.items() if not schema.get('error')
    }
    return f"""You MUST respond with a JSON array of index names from these options:
    {SUPPORTED_INDEXES}
Analyze the query and select the most relevant indexes.
Query: "{query}"
Index Mapping Details:
{index_details}
Index Schema Summary:
{json.dumps(schema_summary, indent=2)}
Response Format:
["index_name1", "index_name2"]
ONLY include indexes that directly match query terms. Do NOT add any extra explanation.
"""

def load_all_index_details() -> Dict[str, str]:
    """Load all generic index mapping details. UPDATE the file paths accordingly."""
    index_files = {
        "generic_index_a": "path/to/generic_mapping_file_a.txt",  # UPDATE with your file path
        "generic_index_b": "path/to/generic_mapping_file_b.txt",  # UPDATE with your file path
        "generic_index_c": "path/to/generic_mapping_file_c.txt"   # UPDATE with your file path
    }
    index_details = {}
    for index_name, file_path in index_files.items():
        try:
            with open(file_path, 'r') as f:
                index_details[index_name] = f.read()
            logger.info(f"Loaded mapping details for {index_name}")
        except Exception as e:
            logger.error(f"Failed to load mapping details for {index_name}: {str(e)}")
            index_details[index_name] = ""
    return index_details

def generate_query_examples() -> str:
    """Return a string containing generic query examples – your personal cookbook for query recipes.
    UPDATE these examples with your own index-specific mappings as needed.
    """
    return """
    Example 1: Exact Match for a Keyword Field
    {
      "query": {
        "bool": {
          "must": [
            { "term": { "generic_field.keyword": "example_value" } }
          ]
        }
      },
      "_source": ["generic_field.keyword"],
      "size": 10
    }

    Example 2: Full-Text Search for a Text Field
    {
      "query": {
        "bool": {
          "must": [
            { "match": { "generic_text_field": "search term" } }
          ]
        }
      },
      "_source": ["generic_text_field"],
      "size": 10
    }

    Example 3: Date Range Query for a Date Field
    {
      "query": {
        "bool": {
          "must": [
            { "range": { "generic_date_field": { "gte": "2023-01-01", "lte": "2023-12-31", "format": "strict_date_optional_time||epoch_millis" } } }
          ]
        }
      },
      "_source": ["generic_date_field"],
      "size": 10
    }

    Example 4: Nested Query for a Nested Field
    {
      "query": {
        "nested": {
          "path": "generic_nested_field",
          "query": {
            "bool": {
              "must": [
                { "match": { "generic_nested_field.inner_text": "nested search" } }
              ]
            }
          },
          "inner_hits": { "size": 5 }
        }
      },
      "_source": ["generic_nested_field"],
      "size": 10
    }

    Example 5: Aggregation on a Keyword Field
    {
      "size": 0,
      "aggs": {
        "keyword_agg": {
          "terms": { "field": "generic_field.keyword", "size": 10 }
        }
      }
    }
    """

def generate_query_prompt(user_query: str, schema: JsonDict, index_name: str, index_details: str) -> str:
    """
    Generate a structured Elasticsearch query prompt using provided schema and mapping details.
    """
    examples = generate_query_examples()
    logger.info(f'Loaded Query Examples: {examples}')

    mapping = es.indices.get_mapping(index=index_name).body  # Retrieve index mapping
    logger.info(f'Mappings loaded for {index_name}: {mapping}')

    keyword_fields = schema.get('keyword_fields', [])
    text_fields = [f for f, v in schema['field_types'].items() if v == 'text']
    numeric_fields = schema.get('numeric_fields', [])
    date_fields = schema.get('date_fields', [])
    nested_fields = schema.get('nested_paths', [])

    return f"""
    Generate a valid Elasticsearch query JSON for index '{index_name}'
    using the provided schema, mapping details, and user query.

    ### Index Mapping and Fields:
    - Nested Paths: {', '.join(nested_fields) if nested_fields else "None"}
    - Date Fields: {', '.join(date_fields) if date_fields else "None"}
    - Keyword Fields: {', '.join(keyword_fields) if keyword_fields else "None"}
    - Text Fields: {', '.join(text_fields) if text_fields else "None"}
    - Numeric Fields: {', '.join(numeric_fields) if numeric_fields else "None"}

    #### Mapping File Details:
    {index_details}

    #### Complete Mappings:
    {json.dumps(mapping, indent=2)}

    ### Query Construction Rules:
    - For text fields, use "match" for full-text search.
    - For keyword fields, use "term" for exact matches.
    - Use "range" for date and numeric filters.
    - For nested fields, use "nested" queries with inner_hits.
    - Follow the examples provided below.

    ### Query Examples:
    {examples}

    **User Query:**
    "{user_query}"

    **Output:**
    Return ONLY a valid Elasticsearch JSON query without additional explanation.
    """

def sanitize_query(raw_query: JsonDict, schema: JsonDict) -> JsonDict:
    """
    Sanitize the generated query by ensuring:
    - '.keyword' is only used for keyword fields.
    - 'term' is replaced with 'match' for text fields.
    """
    def sanitize_clause(clause):
        if isinstance(clause, dict):
            new_clause = {}
            for key, value in clause.items():
                if isinstance(value, dict):
                    new_clause[key] = sanitize_clause(value)
                elif isinstance(value, list):
                    new_clause[key] = [sanitize_clause(item) for item in value]
                else:
                    new_clause[key] = value

                if key in ["term", "match", "range"] and isinstance(value, dict):
                    field_name = list(value.keys())[0]
                    base_field = field_name.replace(".keyword", "")
                    if field_name.endswith(".keyword") and base_field not in schema["keyword_fields"]:
                        logger.info(f"Removing '.keyword' from {base_field} as it's not a keyword field")
                        new_clause[key] = {base_field: value[field_name]}
                    if key == "term" and base_field in schema["field_types"]:
                        if schema["field_types"][base_field] == "text":
                            logger.info(f"Replacing 'term' with 'match' for text field: {base_field}")
                            new_clause["match"] = {base_field: value[field_name]}
                            del new_clause[key]
            return new_clause
        return clause

    try:
        logger.info("Starting query sanitization")
        sanitized_query = sanitize_clause(raw_query)
        logger.info("Query sanitization complete")
        return sanitized_query
    except Exception as e:
        logger.error(f"Sanitization failed: {str(e)}")
        return raw_query

def test_query(index: str, query: JsonDict) -> bool:
    """Test if a query returns valid results – a quick crack to see if our eggs don't fry."""
    try:
        logger.info(f"Testing query on index {index}: {json.dumps(query, indent=2)}")
        test_query_body = query.copy()
        test_query_body['size'] = 1
        response = es.search(index=index, body=test_query_body)
        total_hits = response['hits']['total']['value']
        logger.info(f"Test query on {index} returned {total_hits} hits")
        return total_hits > 0
    except Exception as e:
        logger.error(f"Test query failed: {str(e)}")
        return False

def execute_query(query: JsonDict, indexes: List[str]) -> JsonDict:
    """Execute the Elasticsearch query and return the raw response."""
    try:
        logger.info(f"Executing query on indexes: {indexes}")
        if not query.get('size'):
            query['size'] = 100  # Default size
        response = es.search(index=indexes, body=query)
        logger.info("Query executed successfully")
        return response
    except BadRequestError as e:
        logger.error(f"Query execution failed: {str(e.info)}")
        return {"error": str(e.info)}
    except Exception as e:
        logger.error(f"Unexpected error during query execution: {str(e)}")
        return {"error": str(e)}

def generate_query_examples() -> str:
    """Return generic query examples – update these with your own examples if needed."""
    return """
    Example 1: Exact Match for a Keyword Field
    {
      "query": {
        "bool": {
          "must": [
            { "term": { "generic_field.keyword": "example_value" } }
          ]
        }
      },
      "_source": ["generic_field.keyword"],
      "size": 10
    }

    Example 2: Full-Text Search for a Text Field
    {
      "query": {
        "bool": {
          "must": [
            { "match": { "generic_text_field": "search term" } }
          ]
        }
      },
      "_source": ["generic_text_field"],
      "size": 10
    }

    Example 3: Date Range Query for a Date Field
    {
      "query": {
        "bool": {
          "must": [
            { "range": { "generic_date_field": { "gte": "2023-01-01", "lte": "2023-12-31", "format": "strict_date_optional_time||epoch_millis" } } }
          ]
        }
      },
      "_source": ["generic_date_field"],
      "size": 10
    }

    Example 4: Nested Query for a Nested Field
    {
      "query": {
        "nested": {
          "path": "generic_nested_field",
          "query": {
            "bool": {
              "must": [
                { "match": { "generic_nested_field.inner_text": "nested search" } }
              ]
            }
          },
          "inner_hits": { "size": 5 }
        }
      },
      "_source": ["generic_nested_field"],
      "size": 10
    }
    """

def main():
    # Command-line argument parsing – let the code know how to strut its stuff.
    parser = argparse.ArgumentParser(description='Elasticsearch Query Generator')
    parser.add_argument('--port', type=int, default=7001, help='Port to run the server')
    parser.add_argument('--serve', action='store_true', help='Run in server mode')
    args = parser.parse_args()
    if args.serve:
        app.run(debug=True, host='0.0.0.0', port=args.port)

if __name__ == '__main__':
    main()
