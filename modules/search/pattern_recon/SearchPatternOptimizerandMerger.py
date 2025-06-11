#!/usr/bin/env python3
"""
Search Pattern Optimizer and Merger

This script optimizes and merges search patterns from multiple sources:
1. Generated patterns from the part table analyzer
2. Extracted patterns from database analysis
3. Existing patterns in your database
4. Manual pattern additions

Features:
- Deduplicates similar patterns
- Optimizes pattern priorities based on effectiveness
- Merges overlapping patterns
- Validates pattern syntax
- Generates optimized SQL for deployment


"""

import re
import json
import logging
import argparse
from typing import Dict, List, Set, Any, Tuple, Optional
from collections import defaultdict, Counter
from dataclasses import dataclass
import psycopg2
from psycopg2.extras import RealDictCursor
from urllib.parse import urlparse
from datetime import datetime
from difflib import SequenceMatcher

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class OptimizedPattern:
    """An optimized search pattern"""
    pattern_text: str
    pattern_type: str
    priority: float
    success_rate: float
    confidence: float
    intent_names: List[str]
    target_columns: List[str]
    source: str
    usage_count: int = 0
    effectiveness_score: float = 0.0
    conflicts: List[str] = None

    def __post_init__(self):
        if self.conflicts is None:
            self.conflicts = []
        self.effectiveness_score = self.priority * self.success_rate * self.confidence


class PatternOptimizer:
    """Optimize and merge search patterns from multiple sources"""

    def __init__(self):
        self.input_patterns: List[Dict[str, Any]] = []
        self.existing_patterns: List[Dict[str, Any]] = []
        self.optimized_patterns: List[OptimizedPattern] = []
        self.optimization_stats = {
            'total_input': 0,
            'duplicates_removed': 0,
            'patterns_merged': 0,
            'invalid_patterns': 0,
            'optimized_patterns': 0
        }

        # Pattern similarity threshold for merging
        self.similarity_threshold = 0.85

        # Effectiveness weights
        self.weights = {
            'priority': 0.4,
            'success_rate': 0.3,
            'confidence': 0.2,
            'usage_count': 0.1
        }

    def load_patterns_from_file(self, file_path: str) -> None:
        """Load patterns from JSON file"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            # Handle different JSON structures
            if isinstance(data, list):
                self.input_patterns = data
            elif isinstance(data, dict):
                if 'extracted_patterns' in data:
                    self.input_patterns = data['extracted_patterns']
                elif 'patterns' in data:
                    self.input_patterns = data['patterns']
                else:
                    self.input_patterns = list(data.values())

            logger.info(f"Loaded {len(self.input_patterns)} patterns from {file_path}")

        except Exception as e:
            logger.error(f"Failed to load patterns from {file_path}: {e}")

    def load_existing_patterns_from_db(self, db_url: str) -> None:
        """Load existing patterns from database"""
        try:
            parsed = urlparse(db_url)
            connection = psycopg2.connect(
                host=parsed.hostname,
                port=parsed.port or 5432,
                database=parsed.path[1:],
                user=parsed.username,
                password=parsed.password,
                cursor_factory=RealDictCursor
            )

            cursor = connection.cursor()
            cursor.execute("""
                SELECT ip.pattern_text, ip.pattern_type, ip.priority, 
                       ip.success_rate, ip.usage_count, si.name as intent_name
                FROM intent_pattern ip
                JOIN search_intent si ON si.id = ip.intent_id
                WHERE ip.is_active = true
            """)

            rows = cursor.fetchall()

            for row in rows:
                self.existing_patterns.append({
                    'pattern_text': row['pattern_text'],
                    'pattern_type': row['pattern_type'],
                    'priority': float(row['priority']),
                    'success_rate': float(row['success_rate']),
                    'usage_count': int(row['usage_count']),
                    'intent_name': row['intent_name'],
                    'source': 'existing_database'
                })

            cursor.close()
            connection.close()

            logger.info(f"Loaded {len(self.existing_patterns)} existing patterns from database")

        except Exception as e:
            logger.error(f"Failed to load existing patterns: {e}")

    def calculate_pattern_similarity(self, pattern1: str, pattern2: str) -> float:
        """Calculate similarity between two regex patterns"""
        # Basic string similarity
        basic_similarity = SequenceMatcher(None, pattern1, pattern2).ratio()

        # Check for structural similarity in regex patterns
        # Remove regex-specific characters for comparison
        clean1 = re.sub(r'[(){}[\].*+?^$|\\]', '', pattern1).lower()
        clean2 = re.sub(r'[(){}[\].*+?^$|\\]', '', pattern2).lower()

        structural_similarity = SequenceMatcher(None, clean1, clean2).ratio()

        # Weight the similarities
        return (basic_similarity * 0.6) + (structural_similarity * 0.4)

    def validate_regex_pattern(self, pattern: str) -> Tuple[bool, str]:
        """Validate regex pattern syntax"""
        try:
            re.compile(pattern)
            return True, "Valid"
        except re.error as e:
            return False, str(e)

    def normalize_pattern(self, pattern_data: Dict[str, Any]) -> OptimizedPattern:
        """Normalize pattern data into OptimizedPattern"""
        pattern_text = pattern_data.get('pattern_text', '')
        pattern_type = pattern_data.get('pattern_type', 'general')
        priority = float(pattern_data.get('priority', 0.5))
        success_rate = float(pattern_data.get('success_rate', 0.7))
        usage_count = int(pattern_data.get('usage_count', 0))

        # Calculate confidence based on available data
        confidence = 0.7  # Default
        if usage_count > 10:
            confidence += 0.2
        if success_rate > 0.8:
            confidence += 0.1

        # Handle different field names for intent
        intent_names = []
        if 'intent_name' in pattern_data:
            intent_names.append(pattern_data['intent_name'])
        elif 'intent_names' in pattern_data:
            intent_names = pattern_data['intent_names']

        # Handle target columns
        target_columns = []
        if 'target_column' in pattern_data:
            target_columns.append(pattern_data['target_column'])
        elif 'target_columns' in pattern_data:
            target_columns = pattern_data['target_columns']

        source = pattern_data.get('source', 'unknown')

        return OptimizedPattern(
            pattern_text=pattern_text,
            pattern_type=pattern_type,
            priority=min(priority, 1.0),
            success_rate=min(success_rate, 1.0),
            confidence=min(confidence, 1.0),
            intent_names=intent_names,
            target_columns=target_columns,
            source=source,
            usage_count=usage_count
        )

    def find_similar_patterns(self, target_pattern: OptimizedPattern,
                              pattern_list: List[OptimizedPattern]) -> List[Tuple[OptimizedPattern, float]]:
        """Find patterns similar to the target pattern"""
        similar = []

        for pattern in pattern_list:
            if pattern == target_pattern:
                continue

            similarity = self.calculate_pattern_similarity(
                target_pattern.pattern_text,
                pattern.pattern_text
            )

            if similarity >= self.similarity_threshold:
                similar.append((pattern, similarity))

        return sorted(similar, key=lambda x: x[1], reverse=True)

    def merge_similar_patterns(self, patterns: List[OptimizedPattern]) -> OptimizedPattern:
        """Merge similar patterns into a single optimized pattern"""
        if len(patterns) == 1:
            return patterns[0]

        # Sort by effectiveness score
        patterns.sort(key=lambda p: p.effectiveness_score, reverse=True)
        base_pattern = patterns[0]

        # Merge attributes
        merged_intent_names = set(base_pattern.intent_names)
        merged_target_columns = set(base_pattern.target_columns)
        total_usage = base_pattern.usage_count
        sources = [base_pattern.source]

        for pattern in patterns[1:]:
            merged_intent_names.update(pattern.intent_names)
            merged_target_columns.update(pattern.target_columns)
            total_usage += pattern.usage_count
            if pattern.source not in sources:
                sources.append(pattern.source)

        # Calculate weighted averages
        total_weight = sum(p.effectiveness_score for p in patterns)

        if total_weight > 0:
            weighted_priority = sum(p.priority * p.effectiveness_score for p in patterns) / total_weight
            weighted_success_rate = sum(p.success_rate * p.effectiveness_score for p in patterns) / total_weight
            weighted_confidence = sum(p.confidence * p.effectiveness_score for p in patterns) / total_weight
        else:
            weighted_priority = base_pattern.priority
            weighted_success_rate = base_pattern.success_rate
            weighted_confidence = base_pattern.confidence

        merged_pattern = OptimizedPattern(
            pattern_text=base_pattern.pattern_text,  # Use the most effective pattern
            pattern_type=base_pattern.pattern_type,
            priority=weighted_priority,
            success_rate=weighted_success_rate,
            confidence=weighted_confidence,
            intent_names=list(merged_intent_names),
            target_columns=list(merged_target_columns),
            source=f"merged({','.join(sources)})",
            usage_count=total_usage
        )

        return merged_pattern

    def optimize_patterns(self) -> None:
        """Main optimization process"""
        logger.info("Starting pattern optimization...")

        # Combine all input patterns
        all_patterns = []
        self.optimization_stats['total_input'] = len(self.input_patterns) + len(self.existing_patterns)

        # Normalize input patterns
        for pattern_data in self.input_patterns + self.existing_patterns:
            try:
                normalized = self.normalize_pattern(pattern_data)

                # Validate regex
                is_valid, error_msg = self.validate_regex_pattern(normalized.pattern_text)
                if not is_valid:
                    logger.warning(f"Invalid regex pattern: {normalized.pattern_text} - {error_msg}")
                    self.optimization_stats['invalid_patterns'] += 1
                    continue

                all_patterns.append(normalized)

            except Exception as e:
                logger.warning(f"Failed to normalize pattern: {e}")
                self.optimization_stats['invalid_patterns'] += 1

        logger.info(f"Normalized {len(all_patterns)} valid patterns")

        # Group similar patterns
        processed = set()
        pattern_groups = []

        for i, pattern in enumerate(all_patterns):
            if i in processed:
                continue

            # Find similar patterns
            similar_patterns = self.find_similar_patterns(pattern, all_patterns)

            # Create group with current pattern and similar ones
            group = [pattern]
            for similar_pattern, similarity in similar_patterns:
                similar_index = all_patterns.index(similar_pattern)
                if similar_index not in processed:
                    group.append(similar_pattern)
                    processed.add(similar_index)

            processed.add(i)
            pattern_groups.append(group)

        logger.info(f"Grouped patterns into {len(pattern_groups)} groups")

        # Merge and optimize each group
        for group in pattern_groups:
            if len(group) > 1:
                self.optimization_stats['patterns_merged'] += len(group) - 1
                self.optimization_stats['duplicates_removed'] += len(group) - 1

            optimized = self.merge_similar_patterns(group)
            self.optimized_patterns.append(optimized)

        # Sort by effectiveness score
        self.optimized_patterns.sort(key=lambda p: p.effectiveness_score, reverse=True)

        self.optimization_stats['optimized_patterns'] = len(self.optimized_patterns)

        logger.info(f"Optimization complete: {len(self.optimized_patterns)} optimized patterns")

    def detect_pattern_conflicts(self) -> None:
        """Detect potential conflicts between patterns"""
        logger.info("Detecting pattern conflicts...")

        for i, pattern1 in enumerate(self.optimized_patterns):
            for j, pattern2 in enumerate(self.optimized_patterns[i + 1:], i + 1):
                # Check for overly similar patterns that might conflict
                similarity = self.calculate_pattern_similarity(
                    pattern1.pattern_text,
                    pattern2.pattern_text
                )

                if 0.7 <= similarity < self.similarity_threshold:
                    # Potential conflict - patterns are similar but not merged
                    conflict_msg = f"Similar to pattern {j}: {similarity:.2f} similarity"
                    pattern1.conflicts.append(conflict_msg)
                    pattern2.conflicts.append(f"Similar to pattern {i}: {similarity:.2f} similarity")

                # Check for regex conflicts (patterns that might match same text)
                if self._patterns_might_conflict(pattern1.pattern_text, pattern2.pattern_text):
                    conflict_msg = f"Potential regex conflict with pattern {j}"
                    pattern1.conflicts.append(conflict_msg)
                    pattern2.conflicts.append(f"Potential regex conflict with pattern {i}")

    def _patterns_might_conflict(self, pattern1: str, pattern2: str) -> bool:
        """Check if two regex patterns might match the same text"""
        # Simple heuristic: check for overlapping fixed text parts
        # Extract non-regex parts from patterns
        fixed_parts1 = re.findall(r'[a-zA-Z]{3,}', pattern1)
        fixed_parts2 = re.findall(r'[a-zA-Z]{3,}', pattern2)

        # Check for common fixed parts
        common_parts = set(fixed_parts1) & set(fixed_parts2)

        return len(common_parts) > 0 and len(common_parts) / max(len(fixed_parts1), len(fixed_parts2)) > 0.5

    def generate_optimized_sql(self) -> str:
        """Generate SQL for optimized patterns"""
        logger.info("Generating optimized SQL...")

        sql_statements = []
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

        # Header
        sql_statements.append("-- Optimized search patterns")
        sql_statements.append(f"-- Generated at: {current_time}")
        sql_statements.append(f"-- Total optimized patterns: {len(self.optimized_patterns)}")
        sql_statements.append(f"-- Optimization stats: {self.optimization_stats}")
        sql_statements.append("")

        # Group patterns by intent
        patterns_by_intent = defaultdict(list)
        for pattern in self.optimized_patterns:
            if pattern.intent_names:
                for intent_name in pattern.intent_names:
                    patterns_by_intent[intent_name].append(pattern)
            else:
                patterns_by_intent['FIND_PART'].append(pattern)  # Default intent

        # Generate SQL for each intent
        intent_id = 1
        for intent_name, patterns in patterns_by_intent.items():
            sql_statements.append(f"-- Patterns for intent: {intent_name}")

            # Intent insert (assuming it exists or will be created)
            pattern_values = []
            for pattern in patterns:
                escaped_pattern = pattern.pattern_text.replace("'", "''")

                pattern_values.append(
                    f"({intent_id}, '{escaped_pattern}', '{pattern.pattern_type}', "
                    f"{pattern.priority:.3f}, {pattern.success_rate:.3f}, {pattern.usage_count}, "
                    f"true, '{current_time}', '{current_time}')"
                )

            if pattern_values:
                sql_statements.append(
                    "INSERT INTO intent_pattern (intent_id, pattern_text, pattern_type, priority, success_rate, usage_count, is_active, created_at, updated_at) VALUES"
                )
                sql_statements.append(",\n".join(pattern_values) + ";\n")

            intent_id += 1

        return "\n".join(sql_statements)

    def generate_optimization_report(self) -> str:
        """Generate detailed optimization report"""
        report = []
        report.append("=" * 80)
        report.append("SEARCH PATTERN OPTIMIZATION REPORT")
        report.append("=" * 80)
        report.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # Optimization statistics
        report.append("OPTIMIZATION STATISTICS")
        report.append("-" * 40)
        for key, value in self.optimization_stats.items():
            report.append(f"{key.replace('_', ' ').title()}: {value}")

        efficiency = (self.optimization_stats['optimized_patterns'] /
                      max(self.optimization_stats['total_input'], 1)) * 100
        report.append(f"Optimization efficiency: {efficiency:.1f}%")
        report.append("")

        # Pattern effectiveness distribution
        if self.optimized_patterns:
            effectiveness_scores = [p.effectiveness_score for p in self.optimized_patterns]
            report.append("EFFECTIVENESS DISTRIBUTION")
            report.append("-" * 40)
            report.append(f"Highest effectiveness: {max(effectiveness_scores):.3f}")
            report.append(f"Lowest effectiveness: {min(effectiveness_scores):.3f}")
            report.append(f"Average effectiveness: {sum(effectiveness_scores) / len(effectiveness_scores):.3f}")
            report.append("")

        # Top performing patterns
        report.append("TOP PERFORMING PATTERNS")
        report.append("-" * 40)
        top_patterns = sorted(self.optimized_patterns,
                              key=lambda p: p.effectiveness_score, reverse=True)[:10]

        for i, pattern in enumerate(top_patterns, 1):
            report.append(f"{i}. Effectiveness: {pattern.effectiveness_score:.3f}")
            report.append(f"   Pattern: {pattern.pattern_text}")
            report.append(f"   Type: {pattern.pattern_type}")
            report.append(f"   Priority: {pattern.priority:.3f}, Success Rate: {pattern.success_rate:.3f}")
            report.append(f"   Intents: {', '.join(pattern.intent_names) if pattern.intent_names else 'None'}")
            report.append(f"   Source: {pattern.source}")
            if pattern.conflicts:
                report.append(f"   ⚠️  Conflicts: {len(pattern.conflicts)}")
            report.append("")

        # Conflict summary
        conflicted_patterns = [p for p in self.optimized_patterns if p.conflicts]
        if conflicted_patterns:
            report.append("PATTERN CONFLICTS")
            report.append("-" * 40)
            report.append(f"Patterns with conflicts: {len(conflicted_patterns)}")

            for pattern in conflicted_patterns[:5]:  # Show first 5
                report.append(f"Pattern: {pattern.pattern_text}")
                for conflict in pattern.conflicts:
                    report.append(f"  ⚠️  {conflict}")
                report.append("")

        # Pattern type distribution
        type_counts = Counter(p.pattern_type for p in self.optimized_patterns)
        report.append("PATTERN TYPE DISTRIBUTION")
        report.append("-" * 40)
        for pattern_type, count in type_counts.most_common():
            percentage = (count / len(self.optimized_patterns)) * 100
            report.append(f"{pattern_type}: {count} ({percentage:.1f}%)")
        report.append("")

        # Recommendations
        report.append("RECOMMENDATIONS")
        report.append("-" * 40)

        if self.optimization_stats['duplicates_removed'] > 10:
            report.append("✅ Significant duplicate reduction achieved")

        if len(conflicted_patterns) > 5:
            report.append("⚠️  Consider reviewing conflicted patterns for potential issues")

        low_effectiveness = [p for p in self.optimized_patterns if p.effectiveness_score < 0.3]
        if len(low_effectiveness) > 5:
            report.append(f"⚠️  {len(low_effectiveness)} patterns have low effectiveness - consider removal")

        high_usage = [p for p in self.optimized_patterns if p.usage_count > 100]
        if high_usage:
            report.append(f"✅ {len(high_usage)} patterns have high usage - prioritize these")

        return "\n".join(report)

    def export_optimized_data(self, file_path: str) -> None:
        """Export optimized patterns to JSON"""
        export_data = {
            'optimization_stats': self.optimization_stats,
            'optimized_patterns': [
                {
                    'pattern_text': p.pattern_text,
                    'pattern_type': p.pattern_type,
                    'priority': p.priority,
                    'success_rate': p.success_rate,
                    'confidence': p.confidence,
                    'effectiveness_score': p.effectiveness_score,
                    'intent_names': p.intent_names,
                    'target_columns': p.target_columns,
                    'source': p.source,
                    'usage_count': p.usage_count,
                    'conflicts': p.conflicts
                }
                for p in self.optimized_patterns
            ],
            'generation_time': datetime.now().isoformat()
        }

        with open(file_path, 'w') as f:
            json.dump(export_data, f, indent=2)

        logger.info(f"Optimized data exported to: {file_path}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Optimize and merge search patterns')
    parser.add_argument('--input-patterns', help='JSON file with patterns to optimize')
    parser.add_argument('--db-url', help='Database URL to load existing patterns')
    parser.add_argument('--merge-existing', action='store_true', help='Merge with existing database patterns')
    parser.add_argument('--output-sql', default='optimized_patterns.sql', help='Output SQL file')
    parser.add_argument('--output-report', default='optimization_report.txt', help='Optimization report')
    parser.add_argument('--output-json', default='optimized_patterns.json', help='Optimized patterns JSON')
    parser.add_argument('--similarity-threshold', type=float, default=0.85, help='Pattern similarity threshold')

    args = parser.parse_args()

    if not args.input_patterns and not args.db_url:
        parser.error("Must specify either --input-patterns or --db-url")

    # Initialize optimizer
    optimizer = PatternOptimizer()
    optimizer.similarity_threshold = args.similarity_threshold

    # Load patterns
    if args.input_patterns:
        optimizer.load_patterns_from_file(args.input_patterns)

    if args.db_url and args.merge_existing:
        optimizer.load_existing_patterns_from_db(args.db_url)

    if not optimizer.input_patterns and not optimizer.existing_patterns:
        logger.error("No patterns loaded for optimization")
        return 1

    # Optimize patterns
    optimizer.optimize_patterns()
    optimizer.detect_pattern_conflicts()

    # Generate outputs
    sql_output = optimizer.generate_optimized_sql()
    optimization_report = optimizer.generate_optimization_report()

    # Save files
    with open(args.output_sql, 'w') as f:
        f.write(sql_output)
    logger.info(f"Optimized SQL saved to: {args.output_sql}")

    with open(args.output_report, 'w') as f:
        f.write(optimization_report)
    logger.info(f"Optimization report saved to: {args.output_report}")

    optimizer.export_optimized_data(args.output_json)

    # Print summary
    print("\n" + "=" * 60)
    print("PATTERN OPTIMIZATION COMPLETE")
    print("=" * 60)
    print(f"📊 Input patterns: {optimizer.optimization_stats['total_input']}")
    print(f"🔍 Optimized patterns: {optimizer.optimization_stats['optimized_patterns']}")
    print(f"🚫 Duplicates removed: {optimizer.optimization_stats['duplicates_removed']}")
    print(f"🔧 Patterns merged: {optimizer.optimization_stats['patterns_merged']}")
    print(f"❌ Invalid patterns: {optimizer.optimization_stats['invalid_patterns']}")

    conflicted = len([p for p in optimizer.optimized_patterns if p.conflicts])
    if conflicted > 0:
        print(f"⚠️  Patterns with conflicts: {conflicted}")

    efficiency = (optimizer.optimization_stats['optimized_patterns'] /
                  max(optimizer.optimization_stats['total_input'], 1)) * 100
    print(f"📈 Optimization efficiency: {efficiency:.1f}%")

    print(f"\n📝 Files generated:")
    print(f"   SQL: {args.output_sql}")
    print(f"   Report: {args.output_report}")
    print(f"   JSON: {args.output_json}")

    # Show top patterns
    top_patterns = sorted(optimizer.optimized_patterns,
                          key=lambda p: p.effectiveness_score, reverse=True)[:3]
    print(f"\n🏆 Top 3 most effective patterns:")
    for i, pattern in enumerate(top_patterns, 1):
        print(f"   {i}. {pattern.pattern_text[:60]}..." if len(pattern.pattern_text) > 60
              else f"   {i}. {pattern.pattern_text}")
        print(f"      Effectiveness: {pattern.effectiveness_score:.3f}")

    print(f"\n✅ Patterns are ready for deployment!")
    print(f"💡 Review the optimization report for detailed analysis and recommendations.")

    return 0


if __name__ == "__main__":
    exit(main())