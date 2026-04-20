# CRS-RAG: Course Recommendation System with Retrieval-Augmented Generation

## Overview

This project implements a course recommendation system using retrieval-augmented generation (RAG) techniques. It provides intelligent course recommendations based on student profiles and learning goals.

## Prerequisites

Before running this project, you need to set up a Google Cloud project to obtain a project ID:

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Enable the necessary APIs (such as Vertex AI, if applicable)
4. Create a service account and download the credentials file
5. Note your **Project ID** - you'll need it to configure the application

## Installation

1. Clone or download this repository
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Configure your Google Cloud credentials:
   - Set the `GOOGLE_CLOUD_PROJECT` environment variable with your project ID
   - Set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable with the path to your credentials JSON file

## Usage

### 1. Generate Resource Descriptions

Use `generate_document.py` to generate descriptions for all resources in your database:

```bash
python generate_document.py
```

This script reads the resource data from `database/resources.json` and generates detailed descriptions for each resource using the LLM. The generated descriptions are stored and used by the recommendation system.

### 2. Run the Six Scenarios

Execute `main.py` to run the six scenarios defined in this project:

```bash
python main.py
```

The six scenarios demonstrate different use cases and capabilities of the course recommendation system:
- Scenario 1: Basic course recommendation
- Scenario 2: Personalized recommendations based on student profile
- Scenario 3: Advanced filtering and ranking
- Scenario 4: Multi-criteria recommendation
- Scenario 5: Dynamic recommendation updates
- Scenario 6: Zapper behavior simulation

Each scenario executes and displays results with detailed information about the recommendations generated.

## Project Structure

- `main.py` - Entry point that runs all six scenarios
- `generate_document.py` - Generates resource descriptions from `database/resources.json`
- `kgrag_agent.py` - Knowledge graph RAG agent implementation
- `nudge_agent.py` - Nudging recommendation agent
- `recommender.py` - Core recommendation engine
- `llm_client.py` - LLM client for API interactions
- `database/` - Contains resource data and knowledge graphs
  - `resources.json` - Resource definitions
  - `content_data.json` - Content information
  - `forward_graph.json` - Forward knowledge graph
  - `remedial_graph.json` - Remedial knowledge graph
- `models/` - Pre-trained models and configurations
  - `course_metadata.json` - Course metadata
  - `ideal_profile.json` - Ideal student profiles
  - `behavior.json` - Behavior models
  - `kmeans.joblib` - K-means clustering model
  - `scaler.joblib` - Data scaler model
- `private/` - Private data (grades, logs)

## Configuration

Before running the scripts, make sure to:
1. Set your Google Cloud Project ID
2. Configure API credentials
3. Verify that all required data files exist in the `database/` directory

## Output

- Generated descriptions are saved in the appropriate database files
- Scenario results are displayed in the console
- Logs are recorded in `private/logs.csv` for analysis

## Contact

alemarti@uji.es
