import os
import pandas as pd
import time
import re
from typing import Dict, List, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from dataclasses import dataclass

# Import your performance tracker
from modules.emtac_ai.training_scripts.performance_tst_model.performance_tracker import (
    PerformanceTracker, QueryResult, EntityMatch
)
from modules.emtac_ai.config import ORC_PARTS_MODEL_DIR, ORC_TRAINING_DATA_LOADSHEET

# Enhanced query templates with natural language variations
ENHANCED_QUERY_TEMPLATES = [
    # Single entity queries (15)
    "I need part number {itemnum}",
    "Do you have {description}?",
    "I'm looking for something from {manufacturer}",
    "Can I get model {model}?",
    "Do you stock part {itemnum}?",
    "I need some {description}",
    "Looking for {manufacturer} parts",
    "Can you find model {model}?",
    "Do you carry the {itemnum} part?",
    "I'm searching for {description}",
    "Need anything from {manufacturer}",
    "Any model {model} available?",
    "I require part number {itemnum}",
    "Show me {description}",
    "Find {manufacturer} items",

    # Two entity combinations (15)
    "I need {description} from {manufacturer}",
    "Do you have {description} by {manufacturer}?",
    "I'm looking for {manufacturer} {description}",
    "Can I get {description} model {model}?",
    "Do you stock {itemnum} from {manufacturer}?",
    "I need {manufacturer} {description}",
    "Looking for {description} made by {manufacturer}",
    "Can you find {model} by {manufacturer}?",
    "Do you carry {description}, {manufacturer} brand?",
    "I'm searching for {itemnum} or model {model}",
    "Need {description} from {manufacturer}",
    "Any {manufacturer} {description} available?",
    "I require model {model} from {manufacturer}",
    "Show me {description}, model {model}",
    "Find {itemnum} made by {manufacturer}",

    # Three entity combinations (15)
    "I need {description} from {manufacturer}, model {model}",
    "Do you have {manufacturer} {description} model {model}?",
    "I'm looking for {itemnum}, {description} from {manufacturer}",
    "Can I get {description} by {manufacturer}, model {model}?",
    "Do you stock {itemnum} which is {description} from {manufacturer}?",
    "I need {manufacturer} {description}, part number {itemnum}",
    "Looking for {description} made by {manufacturer}, model {model}",
    "Can you find {itemnum}, that's the {description} from {manufacturer}?",
    "Do you carry {manufacturer} part {itemnum}, the {description}?",
    "I'm searching for {model} by {manufacturer}, it's a {description}",
    "Need {description} from {manufacturer}, model {model}",
    "Any {manufacturer} {description} model {model} available?",
    "I require {itemnum}, {description} manufactured by {manufacturer}",
    "Show me {manufacturer} model {model}, the {description}",
    "Find {description} part {itemnum} from {manufacturer}"
]


@dataclass
class EntitySpan:
    """Represents an entity span with normalization"""
    start: int
    end: int
    label: str
    text: str
    confidence: float = 0.0

    def __post_init__(self):
        # Normalize label by removing BIO prefix
        if self.label.startswith(('B-', 'I-')):
            self.label = self.label[2:]

        # Normalize text for comparison
        self.normalized_text = self.normalize_text(self.text)

    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalize text for comparison"""
        if not text:
            return ""

        # Convert to lowercase and strip whitespace
        normalized = text.lower().strip()

        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized)

        # Remove common punctuation that might cause mismatches
        normalized = re.sub(r'[.,;:!?"\'-]', '', normalized)

        return normalized

    def matches(self, other: 'EntitySpan', match_type: str = "exact") -> bool:
        """Check if this entity matches another entity"""
        if self.label != other.label:
            return False

        if match_type == "exact":
            # Exact span match
            return (self.start == other.start and
                    self.end == other.end and
                    self.normalized_text == other.normalized_text)

        elif match_type == "partial":
            # Partial overlap with same normalized text
            span_overlap = not (self.end <= other.start or self.start >= other.end)
            text_match = self.normalized_text == other.normalized_text
            return span_overlap and text_match

        elif match_type == "text_only":
            # Just text match (ignoring position)
            return self.normalized_text == other.normalized_text

        return False

    def overlap_ratio(self, other: 'EntitySpan') -> float:
        """Calculate overlap ratio with another span"""
        if self.label != other.label:
            return 0.0

        overlap_start = max(self.start, other.start)
        overlap_end = min(self.end, other.end)

        if overlap_start >= overlap_end:
            return 0.0

        overlap_length = overlap_end - overlap_start
        total_length = max(self.end - self.start, other.end - other.start)

        return overlap_length / total_length if total_length > 0 else 0.0


class ImprovedNEREvaluator:
    """Improved NER evaluator that handles common mismatch issues"""

    def __init__(self):
        # Common variations and normalizations
        self.manufacturer_aliases = {
            'balston': ['balston filt', 'balston filter', 'balston filtration'],
            'parker': ['parker hannifin', 'parker hnnfn'],
            'smc': ['smc corporation', 'smc corp'],
            # Add more as needed
        }

        # Part number patterns
        self.part_number_patterns = [
            re.compile(r'\bA1\d{5}\b', re.IGNORECASE),  # Your A1##### pattern
            re.compile(r'\b\d{3}-\d{2}-[A-Z]{2}\b', re.IGNORECASE),  # ###-##-XX pattern
            # Add more patterns as needed
        ]

    def normalize_manufacturer(self, manufacturer: str) -> str:
        """Normalize manufacturer names"""
        if not manufacturer:
            return ""

        normalized = manufacturer.lower().strip()

        # Check for known aliases
        for canonical, aliases in self.manufacturer_aliases.items():
            if normalized == canonical or normalized in aliases:
                return canonical

        return normalized

    def extract_entities_from_prediction(self, predicted_entities: List[Dict]) -> List[EntitySpan]:
        """Convert prediction format to EntitySpan objects"""
        entities = []

        # Handle both pipeline output and raw entity lists
        for ent in predicted_entities:
            # Check if this is pipeline output or raw entity dict
            if 'entity_group' in ent:
                # Pipeline output format
                label = ent['entity_group']
                text = ent['word'].replace('##', '')  # Remove tokenizer artifacts
                confidence = ent.get('score', 0.0)
                start = ent.get('start', 0)
                end = ent.get('end', 0)
            else:
                # Raw entity dict format
                label = ent.get('label', ent.get('entity_group', ''))
                text = ent.get('text', ent.get('word', ''))
                confidence = ent.get('score', ent.get('confidence', 0.0))
                start = ent.get('start', 0)
                end = ent.get('end', 0)

            span = EntitySpan(
                start=start,
                end=end,
                label=label,
                text=text,
                confidence=confidence
            )

            # Special handling for manufacturers
            if span.label == 'MANUFACTURER':
                span.normalized_text = self.normalize_manufacturer(span.text)

            entities.append(span)

        return entities

    def extract_entities_from_expected(self, expected_entities: Dict[str, str]) -> List[EntitySpan]:
        """Convert expected format to EntitySpan objects"""
        entities = []
        for label, text in expected_entities.items():
            if not text or pd.isna(text):
                continue

            span = EntitySpan(
                start=0,  # We don't have position info for expected
                end=len(str(text)),
                label=label,
                text=str(text),
                confidence=1.0  # Expected entities have perfect confidence
            )

            # Special handling for manufacturers
            if span.label == 'MANUFACTURER':
                span.normalized_text = self.normalize_manufacturer(span.text)

            entities.append(span)

        return entities

    def find_best_matches(self, predicted: List[EntitySpan], expected: List[EntitySpan]) -> Tuple[
        List[Tuple[EntitySpan, EntitySpan]], List[EntitySpan], List[EntitySpan]]:
        """Find best matches between predicted and expected entities"""

        matches = []
        unmatched_predicted = list(predicted)
        unmatched_expected = list(expected)

        # First pass: text-only matches (since we don't have position info for expected)
        for pred in predicted[:]:
            for exp in expected[:]:
                if pred.matches(exp, "text_only"):
                    matches.append((pred, exp))
                    if pred in unmatched_predicted:
                        unmatched_predicted.remove(pred)
                    if exp in unmatched_expected:
                        unmatched_expected.remove(exp)
                    break

        # Second pass: partial text matches for remaining entities
        for pred in unmatched_predicted[:]:
            best_match = None
            best_score = 0.0

            for exp in unmatched_expected:
                if pred.label == exp.label:
                    # Calculate text similarity
                    pred_words = set(pred.normalized_text.split())
                    exp_words = set(exp.normalized_text.split())

                    if pred_words and exp_words:
                        intersection = pred_words & exp_words
                        union = pred_words | exp_words
                        jaccard_score = len(intersection) / len(union)

                        # Also check substring matches
                        substring_score = 0.0
                        if pred.normalized_text in exp.normalized_text or exp.normalized_text in pred.normalized_text:
                            substring_score = 0.8

                        combined_score = max(jaccard_score, substring_score)

                        if combined_score > 0.5 and combined_score > best_score:  # Require >50% similarity
                            best_match = exp
                            best_score = combined_score

            if best_match:
                matches.append((pred, best_match))
                unmatched_predicted.remove(pred)
                unmatched_expected.remove(best_match)

        return matches, unmatched_predicted, unmatched_expected

    def evaluate_prediction(self, predicted_entities: List[Dict], expected_entities: Dict[str, str]) -> Dict:
        """Evaluate a single prediction"""

        pred_spans = self.extract_entities_from_prediction(predicted_entities)
        exp_spans = self.extract_entities_from_expected(expected_entities)

        matches, false_positives, false_negatives = self.find_best_matches(pred_spans, exp_spans)

        # Calculate metrics
        true_positives = len(matches)
        false_positive_count = len(false_positives)
        false_negative_count = len(false_negatives)

        precision = true_positives / (true_positives + false_positive_count) if (
                                                                                            true_positives + false_positive_count) > 0 else 0.0
        recall = true_positives / (true_positives + false_negative_count) if (
                                                                                         true_positives + false_negative_count) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        # Calculate per-entity metrics
        entity_metrics = {}
        for entity_type in ['PART_NUMBER', 'PART_NAME', 'MANUFACTURER', 'MODEL']:
            pred_count = sum(1 for s in pred_spans if s.label == entity_type)
            exp_count = sum(1 for s in exp_spans if s.label == entity_type)
            match_count = sum(1 for p, e in matches if p.label == entity_type)

            ent_precision = match_count / pred_count if pred_count > 0 else 0.0
            ent_recall = match_count / exp_count if exp_count > 0 else 0.0
            ent_f1 = 2 * ent_precision * ent_recall / (ent_precision + ent_recall) if (
                                                                                                  ent_precision + ent_recall) > 0 else 0.0

            entity_metrics[entity_type] = {
                'precision': ent_precision,
                'recall': ent_recall,
                'f1': ent_f1,
                'predicted': pred_count,
                'expected': exp_count,
                'matched': match_count
            }

        return {
            'overall': {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'true_positives': true_positives,
                'false_positives': false_positive_count,
                'false_negatives': false_negative_count,
                'exact_match': len(false_positives) == 0 and len(false_negatives) == 0
            },
            'entities': entity_metrics,
            'matches': matches,
            'false_positives': false_positives,
            'false_negatives': false_negatives
        }


class ComprehensiveNERTester:
    """Main class for running comprehensive NER tests."""

    def __init__(self, excel_path: str, model_path: str):
        self.excel_path = excel_path
        self.model_path = model_path
        self.nlp = None
        self.tracker = PerformanceTracker()
        self.evaluator = ImprovedNEREvaluator()

    def load_model(self):
        """Load the trained NER model."""
        print(f"Loading NER model from {self.model_path}...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            model = AutoModelForTokenClassification.from_pretrained(self.model_path)

            self.nlp = pipeline(
                "ner",
                model=model,
                tokenizer=tokenizer,
                aggregation_strategy="simple",
                device=-1  # Use CPU
            )
            print("✅ Model loaded successfully!")
            return True
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return False

    def load_inventory_data(self, max_rows: int = None) -> pd.DataFrame:
        """Load inventory data from Excel."""
        print(f"Loading inventory data from {self.excel_path}")
        try:
            df = pd.read_excel(self.excel_path)
            df.columns = [str(c).strip() for c in df.columns]

            if max_rows and len(df) > max_rows:
                df = df.head(max_rows)
                print(f"Limited to first {max_rows} rows for testing")

            print(f"✅ Loaded {len(df)} inventory rows")
            return df
        except Exception as e:
            print(f"❌ Failed to load inventory data: {e}")
            return None

    def normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        if not text or pd.isna(text):
            return ""
        return str(text).strip()

    def extract_entities_from_prediction(self, results: List[Dict]) -> Dict[str, List[str]]:
        """Extract entities from model prediction results."""
        entities = {
            'PART_NAME': [],
            'MANUFACTURER': [],
            'PART_NUMBER': [],
            'MODEL': []
        }

        for result in results:
            entity_type = result['entity_group'].replace('B-', '').replace('I-', '')
            if entity_type in entities:
                # Clean up tokenizer artifacts (##)
                word = result['word'].replace('##', '')
                entities[entity_type].append(word)

        # Join subword tokens for each entity type
        for entity_type in entities:
            if entities[entity_type]:
                entities[entity_type] = [' '.join(entities[entity_type])]

        return entities

    def check_entity_match(self, predicted: List[str], expected: str) -> EntityMatch:
        """Check if predicted entities match expected value using improved evaluation."""
        expected_norm = self.normalize_text(expected)
        if not expected_norm:
            return EntityMatch(
                expected=expected,
                predicted=predicted,
                exact_match=True,
                partial_match=True,
                confidence_score=1.0,
                match_type='exact'
            )

        # Use the improved evaluator for more robust matching
        # Convert to format expected by evaluator
        predicted_entities = []
        for i, pred_text in enumerate(predicted):
            predicted_entities.append({
                'entity_group': 'TEMP',  # Will be normalized
                'word': pred_text,
                'score': 1.0,
                'start': 0,
                'end': len(pred_text)
            })

        expected_entities = {'TEMP': expected}

        eval_result = self.evaluator.evaluate_prediction(predicted_entities, expected_entities)

        # Convert back to EntityMatch format
        exact_match = eval_result['overall']['exact_match']
        partial_match = eval_result['overall']['f1'] > 0.5  # Consider F1 > 0.5 as partial match

        match_type = 'exact' if exact_match else ('partial' if partial_match else 'none')
        confidence = eval_result['overall']['f1']

        return EntityMatch(
            expected=expected,
            predicted=predicted,
            exact_match=exact_match,
            partial_match=partial_match,
            confidence_score=confidence,
            match_type=match_type
        )

    def generate_test_query(self, row: pd.Series, template: str) -> Tuple[str, Dict, str, str]:
        """Generate a test query from a template and inventory row."""
        # Extract values with proper case handling
        description = str(row.get('DESCRIPTION', '')).strip()
        manufacturer = str(row.get('OEMMFG', '')).strip()
        itemnum = str(row.get('ITEMNUM', '')).strip()
        model = str(row.get('MODEL', '')).strip()

        # Use lowercase for template substitution (more natural)
        query = template.format(
            description=description.lower(),
            manufacturer=manufacturer.lower(),
            itemnum=itemnum,
            model=model
        )

        # Keep original case for expected entities
        expected = {}
        if '{itemnum}' in template:
            expected['PART_NUMBER'] = itemnum
        if '{description}' in template:
            expected['PART_NAME'] = description
        if '{manufacturer}' in template:
            expected['MANUFACTURER'] = manufacturer
        if '{model}' in template:
            expected['MODEL'] = model

        # Categorize query complexity
        entity_count = len(expected)
        if entity_count == 1:
            category = 'single'
        elif entity_count == 2:
            category = 'double'
        else:
            category = 'triple'

        # Determine language style (simplified)
        if any(word in template.lower() for word in ['need', 'require', 'part number']):
            style = 'formal'
        elif any(word in template.lower() for word in ['any', 'some', 'stuff']):
            style = 'casual'
        else:
            style = 'contextual'

        return query, expected, category, style

    def test_single_query(self, row_idx: int, template_idx: int, row: pd.Series, template: str) -> QueryResult:
        """Test a single query and return results."""
        query, expected, category, style = self.generate_test_query(row, template)

        query_id = f"row_{row_idx}_template_{template_idx}"

        start_time = time.time()
        try:
            # Get model predictions
            predictions = self.nlp(query)
            execution_time = (time.time() - start_time) * 1000  # Convert to ms

            # Use improved evaluation
            eval_result = self.evaluator.evaluate_prediction(predictions, expected)

            # Create result object
            result = QueryResult(
                query_id=query_id,
                row_index=row_idx,
                template_index=template_idx,
                query_text=query,
                query_category=category,
                language_style=style,
                total_entities_expected=len(expected),
                total_entities_found=len([m for m in eval_result['matches']]),
                execution_time_ms=execution_time
            )

            # Map matches back to individual entity results
            matched_entities = {match[1].label: match[0] for match in eval_result['matches']}

            for entity_type, expected_value in expected.items():
                if entity_type in matched_entities:
                    match_result = EntityMatch(
                        expected=expected_value,
                        predicted=[matched_entities[entity_type].text],
                        exact_match=True,
                        partial_match=True,
                        confidence_score=matched_entities[entity_type].confidence,
                        match_type='exact'
                    )
                else:
                    match_result = EntityMatch(
                        expected=expected_value,
                        predicted=[],
                        exact_match=False,
                        partial_match=False,
                        confidence_score=0.0,
                        match_type='none'
                    )

                # Set entity result
                if entity_type == 'PART_NUMBER':
                    result.part_number_result = match_result
                elif entity_type == 'PART_NAME':
                    result.part_name_result = match_result
                elif entity_type == 'MANUFACTURER':
                    result.manufacturer_result = match_result
                elif entity_type == 'MODEL':
                    result.model_result = match_result

            # Overall success based on F1 score
            result.overall_success = eval_result['overall']['f1'] >= 0.8

            return result

        except Exception as e:
            print(f"Error testing query '{query}': {e}")
            execution_time = (time.time() - start_time) * 1000

            # Return failed result
            return QueryResult(
                query_id=query_id,
                row_index=row_idx,
                template_index=template_idx,
                query_text=query,
                query_category=category,
                language_style=style,
                total_entities_expected=len(expected),
                total_entities_found=0,
                overall_success=False,
                execution_time_ms=execution_time
            )

    def run_comprehensive_test(self, max_rows: int = 50):
        """Run the comprehensive test suite."""
        print("=" * 80)
        print("STARTING COMPREHENSIVE NER MODEL TEST")
        print("=" * 80)

        # Load model
        if not self.load_model():
            return None

        # Load data
        df = self.load_inventory_data(max_rows)
        if df is None:
            return None

        total_tests = len(df) * len(ENHANCED_QUERY_TEMPLATES)
        print(f"Will run {total_tests} total tests ({len(df)} rows × {len(ENHANCED_QUERY_TEMPLATES)} templates)")

        completed_tests = 0

        # Test each row with each template
        for row_idx, row in df.iterrows():
            itemnum = row.get('ITEMNUM', 'Unknown')
            print(f"Testing row {row_idx + 1}/{len(df)}: {itemnum}")

            for template_idx, template in enumerate(ENHANCED_QUERY_TEMPLATES):
                result = self.test_single_query(row_idx, template_idx, row, template)
                self.tracker.add_result(result)

                completed_tests += 1
                if completed_tests % 100 == 0:
                    print(f"  Progress: {completed_tests}/{total_tests} tests completed")

        print(f"✅ Completed all {completed_tests} tests!")
        return self.tracker

    def save_and_report(self, output_file: str = None):
        """Generate and save final report."""
        print("\n" + "=" * 80)
        print("GENERATING PERFORMANCE REPORT")
        print("=" * 80)

        # Print summary
        self.tracker.print_summary_report()

        # Save detailed results
        saved_file = self.tracker.save_results(output_file)

        return saved_file


def main():
    """Main execution function."""
    # Configuration
    excel_path = os.path.join(ORC_TRAINING_DATA_LOADSHEET, "parts_loadsheet.xlsx")
    model_path = ORC_PARTS_MODEL_DIR

    print("Comprehensive NER Model Testing")
    print(f"Excel file: {excel_path}")
    print(f"Model path: {model_path}")

    # Get user input for test size
    try:
        max_rows = int(input("How many inventory rows to test? (default 50): ") or "50")
    except ValueError:
        max_rows = 50
        print("Using default of 50 rows")

    # Create tester and run tests
    tester = ComprehensiveNERTester(excel_path, model_path)
    tracker = tester.run_comprehensive_test(max_rows)

    if tracker:
        # Generate and save report
        output_file = f"ner_comprehensive_test_{max_rows}rows.json"
        tester.save_and_report(output_file)

        print(f"\n🎉 Testing complete! Results saved to {output_file}")
        print(f"Total tests run: {len(tracker.results)}")

        # Quick summary
        successful = sum(1 for r in tracker.results if r.overall_success)
        success_rate = successful / len(tracker.results) if tracker.results else 0
        print(f"Overall success rate: {success_rate:.2%}")
    else:
        print("❌ Testing failed")


if __name__ == "__main__":
    main()