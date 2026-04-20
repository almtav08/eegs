from doc_generator import DocumentGenerator
import json

# Credentials
PROJECT_ID = ""
REGION = "us-central1"

# 1. Initialize the generator
generator = DocumentGenerator(project_id=PROJECT_ID, location=REGION)

# 2. Generate documents
with open("database/resources.json", "r", encoding="utf-8") as f:
    resources = json.load(f)

generations = {}

for resource in resources:
    title = resource["name"]
    recid = resource["recid"]

    document = generator.generate_from_title(title)
    generations[recid] = document

generator.stop()

with open("database/generated_documents.json", "w", encoding="utf-8") as f:
    json.dump(generations, f, indent=2, ensure_ascii=False)