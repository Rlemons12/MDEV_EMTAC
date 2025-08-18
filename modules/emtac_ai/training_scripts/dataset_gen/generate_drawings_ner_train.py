import pandas as pd
import json
import os
import random
import importlib.util
from modules.configuration.config import (
    ORC_TRAINING_DATA_DIR,
    ORC_QUERY_TEMPLATE_DRAWINGS,
    ORC_DRAWINGS_TRAIN_DATA_DIR
)

# Excel file path - corrected to remove duplicate "datasets"
EXCEL_PATH = os.path.join(ORC_TRAINING_DATA_DIR, "loadsheet", "Active Drawing List.xlsx")

# Template and variations paths - load from existing files without .py extension
QUERY_TEMPLATES_PATH = os.path.join(ORC_QUERY_TEMPLATE_DRAWINGS, "DRAWINGS_ENHANCED_QUERY_TEMPLATES.py")
VARIATIONS_PATH = os.path.join(ORC_QUERY_TEMPLATE_DRAWINGS, "DRAWINGS_NATURAL_LANGUAGE_VARIATIONS.py")

# Output folder and filename
OUTPUT_DIR = os.path.join(ORC_DRAWINGS_TRAIN_DATA_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "intent_train_drawings.jsonl")

# Debug: Print the paths being used
print(f"Looking for templates at: {QUERY_TEMPLATES_PATH}")
print(f"Looking for variations at: {VARIATIONS_PATH}")
print(f"Excel file path: {EXCEL_PATH}")
print(f"Output path: {OUTPUT_PATH}")


def load_module_from_path(file_path, module_name):
    """Load a Python module from a file path"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_query_templates():
    """Load query templates from Python file"""
    try:
        templates_module = load_module_from_path(QUERY_TEMPLATES_PATH, "query_templates")
        return templates_module.DRAWINGS_ENHANCED_QUERY_TEMPLATES
    except Exception as e:
        print(f"Error loading query templates: {e}")
        # Fallback to basic templates
        return [
            "I need the print for equipment [EQUIPMENT_NUMBER]",
            "Show me the drawing for [EQUIPMENT_NAME]",
            "Find drawing [DRAWING_NUMBER]",
            "I need the [DRAWING_NAME]",
            "Show me the print for part # [SPARE_PART_NUMBER]"
        ]


def load_natural_language_variations():
    """Load natural language variations from Python file"""
    try:
        variations_module = load_module_from_path(VARIATIONS_PATH, "variations")
        return variations_module.DRAWING_NATURAL_LANGUAGE_VARIATIONS
    except Exception as e:
        print(f"Error loading variations: {e}")
        # Fallback to basic variations
        return {
            "EQUIPMENT_NUMBER": {"formal": ["equipment {equipment_number}"]},
            "EQUIPMENT_NAME": {"formal": ["{equipment_name}"]},
            "DRAWING_NUMBER": {"formal": ["drawing {drawing_number}"]},
            "DRAWING_NAME": {"formal": ["{drawing_name}"]},
            "SPARE_PART_NUMBER": {"formal": ["part {spare_part_number}"]}
        }


def clean_value(value):
    """Clean and validate field values"""
    if pd.isna(value) or value is None:
        return ""
    return str(value).strip()


def parse_spare_parts(spare_part_string):
    """Parse comma-separated spare part numbers"""
    if not spare_part_string or pd.isna(spare_part_string):
        return []

    # Split by comma and clean each part number
    parts = [part.strip() for part in str(spare_part_string).split(',')]
    # Remove empty strings
    parts = [part for part in parts if part]
    return parts


def get_random_variation(variations_dict, entity_type, value):
    """Get a random variation for an entity type"""
    if entity_type not in variations_dict:
        return value

    entity_variations = variations_dict[entity_type]
    all_variations = []

    # Collect all variations from all categories
    for category in entity_variations.values():
        if isinstance(category, list):
            all_variations.extend(category)

    if not all_variations:
        return value

    # Choose random variation and format it
    variation = random.choice(all_variations)
    placeholder = f"{{{entity_type.lower()}}}"
    return variation.replace(placeholder, value)


def generate_training_examples(row, query_templates, variations):
    """Generate training examples for a single row of data"""
    examples = []

    # Extract and clean field values
    equipment_number = clean_value(row.get("equipment_number", ""))
    equipment_name = clean_value(row.get("equipment_name", ""))
    drawing_number = clean_value(row.get("drawing_number", ""))
    drawing_name = clean_value(row.get("drawing_name", ""))
    spare_parts_raw = row.get("spare_part_number", "")

    # Parse multiple spare part numbers
    spare_part_numbers = parse_spare_parts(spare_parts_raw)

    # Skip rows with no useful data
    if not any([equipment_number, equipment_name, drawing_number, drawing_name, spare_part_numbers]):
        return examples

    # Generate examples using query templates
    for template in query_templates:
        # Skip spare part templates if no spare parts
        if "[SPARE_PART_NUMBER]" in template and not spare_part_numbers:
            continue

        # For spare part templates, create examples for each spare part
        if "[SPARE_PART_NUMBER]" in template and spare_part_numbers:
            for spare_part_number in spare_part_numbers:
                # Create multiple variations for each spare part
                for _ in range(2):
                    query = template
                    entities = []

                    # Replace spare part placeholder
                    variation = get_random_variation(variations, "SPARE_PART_NUMBER", spare_part_number)
                    query = query.replace("[SPARE_PART_NUMBER]", variation)
                    entities.append({
                        "entity": "SPARE_PART_NUMBER",
                        "value": spare_part_number,
                        "start": query.find(variation),
                        "end": query.find(variation) + len(variation)
                    })

                    # Replace other placeholders if they exist in the template
                    if "[EQUIPMENT_NUMBER]" in query and equipment_number:
                        eq_variation = get_random_variation(variations, "EQUIPMENT_NUMBER", equipment_number)
                        query = query.replace("[EQUIPMENT_NUMBER]", eq_variation)
                        entities.append({
                            "entity": "EQUIPMENT_NUMBER",
                            "value": equipment_number,
                            "start": query.find(eq_variation),
                            "end": query.find(eq_variation) + len(eq_variation)
                        })

                    if "[EQUIPMENT_NAME]" in query and equipment_name:
                        name_variation = get_random_variation(variations, "EQUIPMENT_NAME", equipment_name)
                        query = query.replace("[EQUIPMENT_NAME]", name_variation)
                        entities.append({
                            "entity": "EQUIPMENT_NAME",
                            "value": equipment_name,
                            "start": query.find(name_variation),
                            "end": query.find(name_variation) + len(name_variation)
                        })

                    if "[DRAWING_NUMBER]" in query and drawing_number:
                        dwg_variation = get_random_variation(variations, "DRAWING_NUMBER", drawing_number)
                        query = query.replace("[DRAWING_NUMBER]", dwg_variation)
                        entities.append({
                            "entity": "DRAWING_NUMBER",
                            "value": drawing_number,
                            "start": query.find(dwg_variation),
                            "end": query.find(dwg_variation) + len(dwg_variation)
                        })

                    if "[DRAWING_NAME]" in query and drawing_name:
                        name_variation = get_random_variation(variations, "DRAWING_NAME", drawing_name)
                        query = query.replace("[DRAWING_NAME]", name_variation)
                        entities.append({
                            "entity": "DRAWING_NAME",
                            "value": drawing_name,
                            "start": query.find(name_variation),
                            "end": query.find(name_variation) + len(name_variation)
                        })

                    # Only add if we successfully replaced all placeholders
                    if entities and "[" not in query:
                        examples.append({
                            "text": query,
                            "intent": "drawing_search",
                            "entities": entities
                        })

        else:
            # Handle non-spare-part templates
            for _ in range(2):  # Generate 2 variations per template
                query = template
                entities = []

                # Replace placeholders with actual values and variations
                if "[EQUIPMENT_NUMBER]" in query and equipment_number:
                    variation = get_random_variation(variations, "EQUIPMENT_NUMBER", equipment_number)
                    query = query.replace("[EQUIPMENT_NUMBER]", variation)
                    entities.append({
                        "entity": "EQUIPMENT_NUMBER",
                        "value": equipment_number,
                        "start": query.find(variation),
                        "end": query.find(variation) + len(variation)
                    })

                if "[EQUIPMENT_NAME]" in query and equipment_name:
                    variation = get_random_variation(variations, "EQUIPMENT_NAME", equipment_name)
                    query = query.replace("[EQUIPMENT_NAME]", variation)
                    entities.append({
                        "entity": "EQUIPMENT_NAME",
                        "value": equipment_name,
                        "start": query.find(variation),
                        "end": query.find(variation) + len(variation)
                    })

                if "[DRAWING_NUMBER]" in query and drawing_number:
                    variation = get_random_variation(variations, "DRAWING_NUMBER", drawing_number)
                    query = query.replace("[DRAWING_NUMBER]", variation)
                    entities.append({
                        "entity": "DRAWING_NUMBER",
                        "value": drawing_number,
                        "start": query.find(variation),
                        "end": query.find(variation) + len(variation)
                    })

                if "[DRAWING_NAME]" in query and drawing_name:
                    variation = get_random_variation(variations, "DRAWING_NAME", drawing_name)
                    query = query.replace("[DRAWING_NAME]", variation)
                    entities.append({
                        "entity": "DRAWING_NAME",
                        "value": drawing_name,
                        "start": query.find(variation),
                        "end": query.find(variation) + len(variation)
                    })

                # Only add if we successfully replaced at least one placeholder
                if entities and "[" not in query:
                    examples.append({
                        "text": query,
                        "intent": "drawing_search",
                        "entities": entities
                    })

    return examples


def generate_combination_examples(df, variations, num_examples=100):
    """Generate examples using combination patterns"""
    examples = []
    combination_templates = [
        "I need the {drawing_name} for {equipment_name}",
        "Show me {drawing_name} for equipment {equipment_number}",
        "Find {drawing_name} drawing {drawing_number}",
        "Looking for part {spare_part_number} on {equipment_name}",
        "Where is part {spare_part_number} on equipment {equipment_number}?",
        "I need drawing {drawing_number} for the {equipment_name}",
        "Can you find the {equipment_name} print with part {spare_part_number}?",
        "Show the {drawing_name} for {equipment_name} number {equipment_number}"
    ]

    for _ in range(num_examples):
        # Random row from dataframe
        row = df.sample(1).iloc[0]

        equipment_number = clean_value(row.get("equipment_number", ""))
        equipment_name = clean_value(row.get("equipment_name", ""))
        drawing_number = clean_value(row.get("drawing_number", ""))
        drawing_name = clean_value(row.get("drawing_name", ""))
        spare_parts_raw = row.get("spare_part_number", "")
        spare_part_numbers = parse_spare_parts(spare_parts_raw)

        template = random.choice(combination_templates)

        # If template needs spare part, pick one randomly from the list
        if "{spare_part_number}" in template and spare_part_numbers:
            spare_part_number = random.choice(spare_part_numbers)
        elif "{spare_part_number}" in template and not spare_part_numbers:
            # Skip this template if no spare parts available
            continue
        else:
            spare_part_number = ""

        query = template
        entities = []

        # Replace placeholders that exist in both template and data
        replacements = {
            "{equipment_number}": equipment_number,
            "{equipment_name}": equipment_name,
            "{drawing_number}": drawing_number,
            "{drawing_name}": drawing_name,
            "{spare_part_number}": spare_part_number
        }

        for placeholder, value in replacements.items():
            if placeholder in query and value:
                # Apply natural language variation
                entity_type = placeholder.strip("{}").upper()
                variation = get_random_variation(variations, entity_type, value)
                query = query.replace(placeholder, variation)

                entities.append({
                    "entity": entity_type,
                    "value": value,
                    "start": query.find(variation),
                    "end": query.find(variation) + len(variation)
                })

        # Only add if we have entities and no remaining placeholders
        if entities and "{" not in query:
            examples.append({
                "text": query,
                "intent": "drawing_search",
                "entities": entities
            })

    return examples


def main():
    print("Loading drawing data...")
    df = pd.read_excel(EXCEL_PATH)
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

    print("Loading query templates and variations...")
    query_templates = load_query_templates()
    variations = load_natural_language_variations()

    print(f"Loaded {len(query_templates)} query templates")
    print(f"Loaded variations for {len(variations)} entity types")

    all_examples = []

    # Generate examples from each row
    print("Generating training examples from data rows...")
    for idx, row in df.iterrows():
        examples = generate_training_examples(row, query_templates, variations)
        all_examples.extend(examples)

        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1} rows, generated {len(all_examples)} examples so far")

    # Generate combination examples
    print("Generating combination examples...")
    combination_examples = generate_combination_examples(df, variations, num_examples=200)
    all_examples.extend(combination_examples)

    # Remove duplicates
    unique_examples = []
    seen_texts = set()
    for example in all_examples:
        if example["text"] not in seen_texts:
            unique_examples.append(example)
            seen_texts.add(example["text"])

    print(f"Generated {len(all_examples)} total examples")
    print(f"After deduplication: {len(unique_examples)} unique examples")

    # Save to JSONL
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f_out:
        for example in unique_examples:
            f_out.write(json.dumps(example, ensure_ascii=False) + "\n")

    print(f"Training data saved to {OUTPUT_PATH}")


    # Print some sample examples
    print("\nSample training examples:")
    for i, example in enumerate(unique_examples[:5]):
        print(f"{i + 1}. Text: {example['text']}")
        print(f"   Entities: {example['entities']}")
        print()

    # Print statistics about spare parts
    spare_part_examples = [ex for ex in unique_examples if
                           any(ent['entity'] == 'SPARE_PART_NUMBER' for ent in ex['entities'])]
    print(f"Examples with spare parts: {len(spare_part_examples)}")

    # Count unique spare parts
    unique_spare_parts = set()
    for example in spare_part_examples:
        for entity in example['entities']:
            if entity['entity'] == 'SPARE_PART_NUMBER':
                unique_spare_parts.add(entity['value'])
    print(f"Unique spare part numbers: {len(unique_spare_parts)}")

    if unique_spare_parts:
        print(f"Sample spare parts: {list(unique_spare_parts)[:10]}")  # Show first 10


if __name__ == "__main__":
    main()