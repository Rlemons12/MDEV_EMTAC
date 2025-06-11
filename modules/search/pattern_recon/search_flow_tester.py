#!/usr/bin/env python3
"""
Search Flow Tester with AI Query Generation

Tests your actual search flow (AggregateSearch.comprehensive_part_search) using:
1. Real part data from your database
2. AI-generated realistic queries
3. Your actual intent patterns
4. Your actual search strategies

This shows how the complete flow works: Pattern -> Parameters -> Search -> Results
"""

import os
import re
import csv
import json
import logging
import sys
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

# Simple path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
sys.path.insert(0, base_dir)

print(f"[SETUP] Added to path: {base_dir}")

# Import your actual search components
from modules.configuration.config_env import DatabaseConfig
from modules.search import SearchIntent, IntentPattern
from modules.emtacdb.emtacdb_fts import Part
from modules.search.aggregate_search import AggregateSearch
from modules.configuration.config import ANTHROPIC_API_KEY

print("[OK] All modules imported successfully!")

try:
    import anthropic

    AI_AVAILABLE = bool(ANTHROPIC_API_KEY)
    print(f"[OK] Anthropic available: {AI_AVAILABLE}")
except ImportError:
    AI_AVAILABLE = False
    print("[WARNING] Anthropic not available - will use template queries")


@dataclass
class SearchTestResult:
    """Result of testing one search scenario."""
    original_part: Dict[str, Any]
    user_query: str
    query_type: str
    triggered_patterns: List[Dict[str, Any]]
    search_parameters: Dict[str, Any]
    search_results: Dict[str, Any]
    found_original_part: bool
    search_method: str
    response_time: float
    success: bool
    error_message: Optional[str] = None


class SearchFlowTester:
    """Tests your complete search flow with AI-generated queries."""

    def __init__(self):
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Database setup
        self.db_config = DatabaseConfig()
        self.session = self.db_config.get_main_session()

        # Initialize your AggregateSearch
        self.aggregate_search = AggregateSearch(session=self.session)

        # AI setup
        self.ai_client = None
        if AI_AVAILABLE:
            self.ai_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

        print("[OK] Search Flow Tester initialized")

    def load_test_parts(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Load sample parts for testing."""
        try:
            parts_query = self.session.query(Part).filter(
                Part.part_number.isnot(None)
            ).limit(limit)

            parts = []
            for part in parts_query.all():
                part_data = {
                    'id': part.id,
                    'part_number': part.part_number,
                    'name': part.name or 'Unknown',
                    'oem_mfg': part.oem_mfg or 'Unknown',
                    'model': part.model or 'Unknown',
                    'class_flag': part.class_flag or '',
                    'type': part.type or '',
                    'notes': part.notes or '',
                    'documentation': part.documentation or ''
                }
                parts.append(part_data)

            self.logger.info(f"Loaded {len(parts)} test parts")
            return parts

        except Exception as e:
            self.logger.error(f"Error loading test parts: {e}")
            return []

    def load_patterns(self) -> List[Dict[str, Any]]:
        """Load all active patterns for testing."""
        try:
            query = self.session.query(
                IntentPattern.id,
                IntentPattern.pattern_text,
                IntentPattern.pattern_type,
                SearchIntent.name.label('intent_name')
            ).join(
                SearchIntent, IntentPattern.intent_id == SearchIntent.id
            ).filter(
                IntentPattern.is_active == True,
                SearchIntent.is_active == True
            )

            patterns = []
            for row in query.all():
                try:
                    compiled_regex = re.compile(row.pattern_text, re.IGNORECASE)
                    patterns.append({
                        'id': row.id,
                        'pattern_text': row.pattern_text,
                        'pattern_type': row.pattern_type,
                        'intent_name': row.intent_name,
                        'compiled_regex': compiled_regex
                    })
                except re.error:
                    continue

            self.logger.info(f"Loaded {len(patterns)} active patterns")
            return patterns

        except Exception as e:
            self.logger.error(f"Error loading patterns: {e}")
            return []

    def generate_ai_queries_for_part(self, part: Dict[str, Any], num_queries: int = 5) -> List[Dict[str, str]]:
        """Generate AI queries for a specific part."""

        if not self.ai_client:
            return self._generate_template_queries_for_part(part, num_queries)

        part_info = f"""Part Number: {part['part_number']}
Name: {part['name']}
Manufacturer: {part['oem_mfg']}
Model: {part['model']}
Type: {part['class_flag']} {part['type']}
Description: {part['notes'][:200]}..."""

        prompt = f"""You are a maintenance worker who needs to find this specific part using a search system. Generate {num_queries} realistic search queries that would help find this part:

{part_info}

Generate different types of queries:
1. **Direct part number**: When you know the exact part number
2. **Description search**: When you know what it is but not the part number  
3. **Manufacturer search**: When you know the brand/manufacturer
4. **Reverse lookup**: When you want to confirm a part number for a description
5. **Urgent/contextual**: Emergency or specific use case scenarios

Return as JSON array:
[
  {{"query": "find part A115957", "type": "direct"}},
  {{"query": "what is the part number for rebuild valve ROSS VALVES", "type": "reverse_lookup"}},
  {{"query": "I need a rebuild valve kit urgently", "type": "description"}},
  {{"query": "show me ROSS VALVES parts", "type": "manufacturer"}},
  {{"query": "rebuild kit for valve system", "type": "contextual"}}
]

Make them sound like real maintenance workers would ask!"""

        try:
            response = self.ai_client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1000,
                temperature=0.7,
                messages=[{"role": "user", "content": prompt}]
            )

            response_text = response.content[0].text

            # Extract JSON
            json_match = re.search(r'\[(.*?)\]', response_text, re.DOTALL)
            if json_match:
                json_text = '[' + json_match.group(1) + ']'
                queries = json.loads(json_text)
                return queries
            else:
                self.logger.warning("Could not parse AI response, using templates")
                return self._generate_template_queries_for_part(part, num_queries)

        except Exception as e:
            self.logger.warning(f"AI query generation failed: {e}")
            return self._generate_template_queries_for_part(part, num_queries)

    def _generate_template_queries_for_part(self, part: Dict[str, Any], num_queries: int = 5) -> List[Dict[str, str]]:
        """Fallback template-based query generation."""

        part_number = part['part_number']
        name = part['name']
        manufacturer = part['oem_mfg']

        queries = [
            {"query": f"find part {part_number}", "type": "direct"},
            {"query": f"search for {part_number}", "type": "direct"},
            {"query": f"what is the part number for {name}", "type": "reverse_lookup"},
            {"query": f"I need {name}", "type": "description"},
            {"query": f"show me {manufacturer} parts", "type": "manufacturer"},
            {"query": f"find {name} from {manufacturer}", "type": "description"},
            {"query": f"part number for {name}?", "type": "reverse_lookup"},
            {"query": f"{part_number} urgent", "type": "urgent"},
            {"query": f"lookup {part_number}", "type": "direct"},
            {"query": f"help me find {name}", "type": "description"}
        ]

        return queries[:num_queries]

    def test_query_through_complete_flow(self, part: Dict[str, Any], query: Dict[str, str]) -> SearchTestResult:
        """Test a single query through the complete search flow."""

        start_time = time.time()
        user_query = query['query']
        query_type = query['type']

        print(f"\n{'=' * 60}")
        print(f"TESTING SEARCH FLOW")
        print(f"{'=' * 60}")
        print(f"Original Part: {part['part_number']} - {part['name']}")
        print(f"User Query: '{user_query}' ({query_type})")

        try:
            # Step 1: Find which patterns match the query
            patterns = self.load_patterns()
            triggered_patterns = []

            print(f"\n[STEP 1] Checking patterns...")
            for pattern in patterns:
                match = pattern['compiled_regex'].search(user_query)
                if match:
                    extracted = match.group(1) if match.groups() else match.group(0)
                    triggered_patterns.append({
                        'pattern_id': pattern['id'],
                        'pattern_text': pattern['pattern_text'],
                        'intent_name': pattern['intent_name'],
                        'extracted_value': extracted
                    })
                    print(f"  ✅ Pattern {pattern['id']} [{pattern['intent_name']}]: '{extracted}'")

            if not triggered_patterns:
                print(f"  ❌ No patterns matched")

            # Step 2: Convert to search parameters (simulate your pattern processing)
            search_parameters = self._convert_to_search_parameters(triggered_patterns, user_query)
            print(f"\n[STEP 2] Search parameters: {search_parameters}")

            # Step 3: Execute your actual search flow
            print(f"\n[STEP 3] Executing AggregateSearch.comprehensive_part_search...")
            search_start = time.time()

            search_results = self.aggregate_search.comprehensive_part_search(search_parameters)

            search_time = time.time() - search_start
            print(f"[STEP 4] Search completed in {search_time:.3f}s")
            print(f"         Status: {search_results.get('status')}")
            print(f"         Results: {search_results.get('count', 0)} parts found")
            print(f"         Method: {search_results.get('search_method', 'unknown')}")

            # Step 4: Check if original part was found
            found_original = self._check_if_original_part_found(part, search_results)

            total_time = time.time() - start_time

            # Create test result
            test_result = SearchTestResult(
                original_part=part,
                user_query=user_query,
                query_type=query_type,
                triggered_patterns=triggered_patterns,
                search_parameters=search_parameters,
                search_results=search_results,
                found_original_part=found_original,
                search_method=search_results.get('search_method', 'unknown'),
                response_time=total_time,
                success=search_results.get('status') == 'success'
            )

            self._print_test_result_summary(test_result)

            return test_result

        except Exception as e:
            self.logger.error(f"Error testing query: {e}")

            test_result = SearchTestResult(
                original_part=part,
                user_query=user_query,
                query_type=query_type,
                triggered_patterns=[],
                search_parameters={},
                search_results={'status': 'error', 'message': str(e)},
                found_original_part=False,
                search_method='error',
                response_time=time.time() - start_time,
                success=False,
                error_message=str(e)
            )

            return test_result

    def _convert_to_search_parameters(self, triggered_patterns: List[Dict], user_query: str) -> Dict[str, Any]:
        """Convert triggered patterns to search parameters."""

        if not triggered_patterns:
            # No pattern matched - try general search
            return {
                'search_text': user_query,
                'limit': 20
            }

        # Use the first (highest priority) pattern
        primary_pattern = triggered_patterns[0]
        extracted_value = primary_pattern['extracted_value']
        intent_name = primary_pattern['intent_name']

        # Convert based on intent and pattern type
        if 'part number for' in primary_pattern['pattern_text'].lower():
            # Reverse lookup pattern - search by description
            return {
                'search_text': extracted_value,
                'limit': 20
            }
        elif intent_name == 'FIND_PART':
            # Check if extracted value looks like a part number or description
            if re.match(r'^[A-Z0-9\-\.]{3,}$', extracted_value.upper()):
                # Looks like part number
                return {
                    'part_number': extracted_value,
                    'limit': 20
                }
            else:
                # Looks like description
                return {
                    'search_text': extracted_value,
                    'limit': 20
                }
        else:
            # Default to search text
            return {
                'search_text': extracted_value,
                'limit': 20
            }

    def _check_if_original_part_found(self, original_part: Dict, search_results: Dict) -> bool:
        """Check if the original part was found in the search results."""

        if search_results.get('status') != 'success':
            return False

        results = search_results.get('results', [])
        original_part_number = original_part['part_number']

        for result in results:
            result_part_number = result.get('part_number', '')
            if result_part_number == original_part_number:
                return True

        return False

    def _print_test_result_summary(self, result: SearchTestResult):
        """Print a summary of the test result."""

        print(f"\n[SUMMARY]")
        print(f"Success: {'✅' if result.success else '❌'}")
        print(f"Found Original Part: {'✅' if result.found_original_part else '❌'}")
        print(f"Response Time: {result.response_time:.3f}s")
        print(f"Search Method: {result.search_method}")
        print(f"Patterns Triggered: {len(result.triggered_patterns)}")

        if result.search_results.get('results'):
            results = result.search_results['results'][:3]  # Show first 3
            print(f"Sample Results:")
            for i, res in enumerate(results, 1):
                print(f"  {i}. {res.get('part_number', 'Unknown')} - {res.get('name', 'Unknown')}")

    def test_multiple_parts(self, num_parts: int = 5, queries_per_part: int = 3) -> List[SearchTestResult]:
        """Test multiple parts with AI-generated queries."""

        print(f"\n{'=' * 80}")
        print(f"TESTING SEARCH FLOW WITH {num_parts} PARTS")
        print(f"{'=' * 80}")

        # Load test parts
        parts = self.load_test_parts(num_parts)
        if not parts:
            print("No parts loaded - cannot run tests")
            return []

        all_results = []

        for i, part in enumerate(parts, 1):
            print(f"\n[PART {i}/{len(parts)}] {part['part_number']} - {part['name']}")

            # Generate queries for this part
            if AI_AVAILABLE:
                print(f"  Generating AI queries...")
                queries = self.generate_ai_queries_for_part(part, queries_per_part)
            else:
                print(f"  Generating template queries...")
                queries = self._generate_template_queries_for_part(part, queries_per_part)

            # Test each query
            for j, query in enumerate(queries, 1):
                print(f"\n  [QUERY {j}/{len(queries)}] Testing: '{query['query']}'")
                result = self.test_query_through_complete_flow(part, query)
                all_results.append(result)

                # Brief result
                status = "✅" if result.found_original_part else "❌"
                print(f"    Result: {status} | Method: {result.search_method} | Time: {result.response_time:.3f}s")

        return all_results

    def generate_comprehensive_report(self, results: List[SearchTestResult]) -> Dict[str, Any]:
        """Generate comprehensive analysis report."""

        successful_searches = [r for r in results if r.success]
        found_original_parts = [r for r in results if r.found_original_part]

        # Analyze by query type
        by_query_type = defaultdict(list)
        for result in results:
            by_query_type[result.query_type].append(result)

        # Analyze by search method
        by_search_method = defaultdict(list)
        for result in results:
            by_search_method[result.search_method].append(result)

        # Analyze by pattern usage
        pattern_usage = defaultdict(int)
        for result in results:
            for pattern in result.triggered_patterns:
                pattern_usage[pattern['pattern_id']] += 1

        report = {
            'timestamp': datetime.now().isoformat(),
            'total_tests': len(results),
            'successful_searches': len(successful_searches),
            'found_original_parts': len(found_original_parts),
            'search_success_rate': len(successful_searches) / len(results) * 100 if results else 0,
            'part_found_rate': len(found_original_parts) / len(results) * 100 if results else 0,
            'avg_response_time': sum(r.response_time for r in results) / len(results) if results else 0,
            'query_type_analysis': {},
            'search_method_analysis': {},
            'pattern_usage': dict(pattern_usage),
            'ai_generated': AI_AVAILABLE
        }

        # Query type analysis
        for query_type, type_results in by_query_type.items():
            found = sum(1 for r in type_results if r.found_original_part)
            report['query_type_analysis'][query_type] = {
                'total_tests': len(type_results),
                'parts_found': found,
                'success_rate': found / len(type_results) * 100,
                'avg_response_time': sum(r.response_time for r in type_results) / len(type_results)
            }

        # Search method analysis
        for method, method_results in by_search_method.items():
            found = sum(1 for r in method_results if r.found_original_part)
            report['search_method_analysis'][method] = {
                'total_tests': len(method_results),
                'parts_found': found,
                'success_rate': found / len(method_results) * 100,
                'avg_response_time': sum(r.response_time for r in method_results) / len(method_results)
            }

        return report

    def save_results(self, results: List[SearchTestResult], filename: str = None) -> str:
        """Save test results to CSV."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"search_flow_test_results_{timestamp}.csv"

        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'original_part_number', 'original_part_name', 'user_query', 'query_type',
                'patterns_triggered', 'search_method', 'found_original_part', 'success',
                'response_time', 'results_count', 'search_parameters', 'error_message'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for result in results:
                writer.writerow({
                    'original_part_number': result.original_part['part_number'],
                    'original_part_name': result.original_part['name'],
                    'user_query': result.user_query,
                    'query_type': result.query_type,
                    'patterns_triggered': '; '.join(
                        [f"P{p['pattern_id']}({p['intent_name']})" for p in result.triggered_patterns]),
                    'search_method': result.search_method,
                    'found_original_part': result.found_original_part,
                    'success': result.success,
                    'response_time': result.response_time,
                    'results_count': result.search_results.get('count', 0),
                    'search_parameters': json.dumps(result.search_parameters),
                    'error_message': result.error_message or ''
                })

        self.logger.info(f"Results saved to: {filename}")
        return filename

    def close(self):
        """Clean up resources."""
        if self.aggregate_search:
            self.aggregate_search.close_session()
        if self.session:
            self.session.close()


def main():
    """Main execution function."""
    print("Search Flow Tester with AI Query Generation")
    print("=" * 60)

    # Initialize tester
    tester = SearchFlowTester()

    try:
        print("\nChoose test mode:")
        print("1. Single part test (detailed)")
        print("2. Multiple parts test (5 parts, 3 queries each)")
        print("3. Custom test")

        choice = input("\nEnter choice (1/2/3): ").strip()

        if choice == "1":
            # Single part detailed test
            parts = tester.load_test_parts(1)
            if not parts:
                print("No parts available for testing")
                return

            part = parts[0]
            print(f"\nTesting with part: {part['part_number']} - {part['name']}")

            # Generate queries
            queries = tester.generate_ai_queries_for_part(part, 5)
            results = []

            for query in queries:
                result = tester.test_query_through_complete_flow(part, query)
                results.append(result)

            # Save results
            filename = tester.save_results(results)

        elif choice == "2":
            # Multiple parts test
            results = tester.test_multiple_parts(num_parts=5, queries_per_part=3)

            # Generate and display report
            report = tester.generate_comprehensive_report(results)

            print(f"\n{'=' * 80}")
            print("COMPREHENSIVE SEARCH FLOW REPORT")
            print("=" * 80)
            print(f"Total Tests: {report['total_tests']}")
            print(f"Search Success Rate: {report['search_success_rate']:.1f}%")
            print(f"Part Found Rate: {report['part_found_rate']:.1f}%")
            print(f"Average Response Time: {report['avg_response_time']:.3f}s")
            print(f"AI Generated Queries: {report['ai_generated']}")

            print(f"\nQuery Type Performance:")
            for qtype, stats in report['query_type_analysis'].items():
                print(f"  {qtype:15}: {stats['success_rate']:5.1f}% ({stats['parts_found']}/{stats['total_tests']})")

            print(f"\nSearch Method Performance:")
            for method, stats in report['search_method_analysis'].items():
                print(f"  {method:20}: {stats['success_rate']:5.1f}% ({stats['parts_found']}/{stats['total_tests']})")

            # Save results
            filename = tester.save_results(results)

            # Save report
            report_file = filename.replace('.csv', '_report.json')
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)

            print(f"\nFiles created:")
            print(f"• Detailed results: {filename}")
            print(f"• Analysis report: {report_file}")

        elif choice == "3":
            # Custom test
            part_number = input("Enter part number to test (e.g., A115957): ").strip()
            if not part_number:
                print("No part number provided")
                return

            # Find the part
            parts = tester.session.query(Part).filter(Part.part_number == part_number).all()
            if not parts:
                print(f"Part {part_number} not found in database")
                return

            part_data = {
                'id': parts[0].id,
                'part_number': parts[0].part_number,
                'name': parts[0].name or 'Unknown',
                'oem_mfg': parts[0].oem_mfg or 'Unknown',
                'model': parts[0].model or 'Unknown',
                'notes': parts[0].notes or ''
            }

            # Generate queries
            queries = tester.generate_ai_queries_for_part(part_data, 5)
            results = []

            for query in queries:
                result = tester.test_query_through_complete_flow(part_data, query)
                results.append(result)

            # Save results
            filename = tester.save_results(results)

        else:
            print("Invalid choice")
            return

        print(f"\nTesting completed! Results saved to: {filename}")

    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        tester.close()


if __name__ == "__main__":
    main()