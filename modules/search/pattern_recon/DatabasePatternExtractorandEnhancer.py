#!/usr/bin/env python3
"""
Database Pattern Extractor and Enhancer - SYNTAX ERRORS FIXED

This script connects to your existing database to:
1. Analyze real part data patterns
2. Extract common search terms and combinations
3. Generate enhanced search patterns based on actual data
4. Create comprehensive intent patterns for your search system
"""

import re
import json
import logging
import argparse
from typing import Dict, List, Set, Any, Tuple, Optional
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
import psycopg2
from psycopg2.extras import RealDictCursor
from urllib.parse import urlparse
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class PatternStats:
    """Statistics for a discovered pattern"""
    pattern: str
    frequency: int
    confidence: float
    examples: List[str]
    pattern_type: str


@dataclass
class ColumnProfile:
    """Profile of a database column"""
    name: str
    total_records: int
    null_count: int
    unique_count: int
    avg_length: float
    common_patterns: List[PatternStats]
    value_samples: List[str]
    data_categories: Dict[str, int]


class DatabasePatternExtractor:
    """Extract search patterns from actual database content"""

    def __init__(self, db_url: str):
        self.db_url = db_url
        self.connection = None
        self.column_profiles: Dict[str, ColumnProfile] = {}
        self.extracted_patterns: List[Dict[str, Any]] = []
        self.search_intents: List[Dict[str, Any]] = []
        self.keywords: List[Dict[str, Any]] = []

    def connect(self) -> bool:
        """Connect to the database"""
        try:
            parsed = urlparse(self.db_url)
            self.connection = psycopg2.connect(
                host=parsed.hostname,
                port=parsed.port or 5432,
                database=parsed.path[1:],
                user=parsed.username,
                password=parsed.password,
                cursor_factory=RealDictCursor
            )
            logger.info("Successfully connected to database")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            return False

    def analyze_part_table(self, table_name: str = 'part', sample_size: int = 10000) -> None:
        """Analyze the part table structure and content"""
        if not self.connection:
            logger.error("No database connection")
            return

        try:
            cursor = self.connection.cursor()

            # Get table schema
            cursor.execute("""
                SELECT column_name, data_type, is_nullable 
                FROM information_schema.columns 
                WHERE table_name = %s
                ORDER BY ordinal_position
            """, (table_name,))

            columns = cursor.fetchall()
            logger.info(f"Found {len(columns)} columns in {table_name} table")

            # Analyze each column
            for col in columns:
                col_name = col['column_name']
                if col_name not in ['id', 'created_at', 'updated_at']:  # Skip system columns
                    self._analyze_column(cursor, table_name, col_name, sample_size)

            cursor.close()

        except Exception as e:
            logger.error(f"Error analyzing part table: {e}")

    def _analyze_column(self, cursor, table_name: str, column_name: str, sample_size: int) -> None:
        """Analyze a specific column"""
        logger.info(f"Analyzing column: {column_name}")

        try:
            # Get basic statistics
            cursor.execute(f"""
                SELECT 
                    COUNT(*) as total_records,
                    COUNT({column_name}) as non_null_count,
                    COUNT(DISTINCT {column_name}) as unique_count,
                    AVG(LENGTH(COALESCE({column_name}::text, ''))) as avg_length
                FROM {table_name}
            """)

            stats = cursor.fetchone()

            # Get sample values
            cursor.execute(f"""
                SELECT {column_name}
                FROM {table_name}
                WHERE {column_name} IS NOT NULL 
                    AND LENGTH(COALESCE({column_name}::text, '')) > 0
                ORDER BY RANDOM()
                LIMIT %s
            """, (min(sample_size, 1000),))

            samples = [row[column_name] for row in cursor.fetchall() if row[column_name]]

            if not samples:
                logger.warning(f"No valid samples found for column {column_name}")
                return

            # Analyze patterns
            patterns = self._extract_column_patterns(samples, column_name)
            categories = self._categorize_column_values(samples, column_name)

            # Create column profile
            profile = ColumnProfile(
                name=column_name,
                total_records=stats['total_records'],
                null_count=stats['total_records'] - stats['non_null_count'],
                unique_count=stats['unique_count'],
                avg_length=float(stats['avg_length']) if stats['avg_length'] else 0.0,
                common_patterns=patterns,
                value_samples=samples[:20],
                data_categories=categories
            )

            self.column_profiles[column_name] = profile

            logger.info(f"Column {column_name}: {len(samples)} samples, "
                        f"{len(patterns)} patterns, {len(categories)} categories")

        except Exception as e:
            logger.error(f"Error analyzing column {column_name}: {e}")

    def _extract_column_patterns(self, samples: List[str], column_name: str) -> List[PatternStats]:
        """Extract patterns from column samples"""
        patterns = []

        if column_name == 'part_number':
            patterns.extend(self._extract_part_number_patterns(samples))
        elif column_name in ['name', 'description', 'notes']:
            patterns.extend(self._extract_text_patterns(samples, column_name))
        elif column_name in ['oem_mfg', 'manufacturer']:
            patterns.extend(self._extract_manufacturer_patterns(samples))
        elif column_name == 'model':
            patterns.extend(self._extract_model_patterns(samples))
        else:
            patterns.extend(self._extract_generic_patterns(samples, column_name))

        return sorted(patterns, key=lambda x: x.frequency, reverse=True)[:10]

    def _extract_part_number_patterns(self, samples: List[str]) -> List[PatternStats]:
        """Extract part number patterns"""
        pattern_counters = Counter()
        pattern_examples = defaultdict(list)

        # Define part number regex patterns
        pn_patterns = {
            'letter_digits': r'^[A-Z][0-9]{5,}$',  # A123456
            'letters_digits': r'^[A-Z]{2,}[0-9]{3,}$',  # ABC123
            'alphanumeric_dash': r'^[A-Z0-9]{2,}-[A-Z0-9]{2,}$',  # ABC-123
            'alphanumeric_dot': r'^[A-Z0-9]{2,}\.[A-Z0-9]{2,}$',  # ABC.123
            'digits_only': r'^[0-9]{4,}$',  # 123456
            'mixed_complex': r'^[A-Z0-9]{3,}[-\.]?[A-Z0-9]*$',  # Complex patterns
        }

        for sample in samples:
            sample_upper = str(sample).upper().strip()
            if len(sample_upper) >= 3:
                for pattern_name, regex in pn_patterns.items():
                    if re.match(regex, sample_upper):
                        pattern_counters[pattern_name] += 1
                        if len(pattern_examples[pattern_name]) < 5:
                            pattern_examples[pattern_name].append(sample)
                        break

        # Convert to PatternStats objects
        pattern_stats = []
        total_samples = len(samples)

        for pattern_name, count in pattern_counters.items():
            confidence = count / total_samples
            if confidence >= 0.05:  # At least 5% of samples
                pattern_stats.append(PatternStats(
                    pattern=pn_patterns[pattern_name],
                    frequency=count,
                    confidence=confidence,
                    examples=pattern_examples[pattern_name],
                    pattern_type='part_number'
                ))

        return pattern_stats

    def _extract_text_patterns(self, samples: List[str], column_name: str) -> List[PatternStats]:
        """Extract patterns from text columns"""
        word_counter = Counter()
        phrase_counter = Counter()
        pattern_examples = defaultdict(list)

        for sample in samples:
            sample_str = str(sample).upper()

            # Extract words
            words = re.findall(r'\b[A-Z]{2,}\b', sample_str)
            word_counter.update(words)

            # Extract 2-word phrases
            phrases = []
            words_list = words
            for i in range(len(words_list) - 1):
                phrase = f"{words_list[i]} {words_list[i + 1]}"
                phrases.append(phrase)
            phrase_counter.update(phrases)

        pattern_stats = []
        total_samples = len(samples)

        # Top words as patterns
        for word, count in word_counter.most_common(15):
            if count >= 3 and len(word) >= 3:  # At least 3 occurrences
                confidence = count / total_samples
                examples = [s for s in samples if word in str(s).upper()][:5]

                pattern_stats.append(PatternStats(
                    pattern=fr'\b{word}\b',
                    frequency=count,
                    confidence=confidence,
                    examples=examples,
                    pattern_type='word'
                ))

        # Top phrases as patterns
        for phrase, count in phrase_counter.most_common(10):
            if count >= 2:  # At least 2 occurrences
                confidence = count / total_samples
                examples = [s for s in samples if phrase in str(s).upper()][:3]

                pattern_stats.append(PatternStats(
                    pattern=fr'\b{phrase}\b',
                    frequency=count,
                    confidence=confidence,
                    examples=examples,
                    pattern_type='phrase'
                ))

        return pattern_stats

    def _extract_manufacturer_patterns(self, samples: List[str]) -> List[PatternStats]:
        """Extract manufacturer patterns"""
        mfg_counter = Counter()
        pattern_examples = defaultdict(list)

        for sample in samples:
            sample_clean = str(sample).upper().strip()
            if len(sample_clean) >= 2:
                mfg_counter[sample_clean] += 1
                if len(pattern_examples[sample_clean]) < 3:
                    pattern_examples[sample_clean].append(sample)

        pattern_stats = []
        total_samples = len(samples)

        for manufacturer, count in mfg_counter.most_common(20):
            if count >= 2:  # At least 2 occurrences
                confidence = count / total_samples
                pattern_stats.append(PatternStats(
                    pattern=fr'\b{re.escape(manufacturer)}\b',
                    frequency=count,
                    confidence=confidence,
                    examples=pattern_examples[manufacturer],
                    pattern_type='manufacturer'
                ))

        return pattern_stats

    def _extract_model_patterns(self, samples: List[str]) -> List[PatternStats]:
        """Extract model number patterns"""
        pattern_counters = Counter()
        pattern_examples = defaultdict(list)

        model_patterns = {
            'alpha_numeric': r'^[A-Z]{2,}[0-9]{2,}$',  # ABC123
            'numeric_alpha': r'^[0-9]{2,}[A-Z]{2,}$',  # 123ABC
            'mixed_dash': r'^[A-Z0-9]+-[A-Z0-9]+$',  # A1-B2
            'series_number': r'^[A-Z]+[0-9]+[-\.]?[A-Z0-9]*$',  # SERIES123-A
            'complex_model': r'^[A-Z0-9]{3,}$',  # General alphanumeric
        }

        for sample in samples:
            sample_upper = str(sample).upper().strip()
            if len(sample_upper) >= 3:
                for pattern_name, regex in model_patterns.items():
                    if re.match(regex, sample_upper):
                        pattern_counters[pattern_name] += 1
                        if len(pattern_examples[pattern_name]) < 5:
                            pattern_examples[pattern_name].append(sample)
                        break

        pattern_stats = []
        total_samples = len(samples)

        for pattern_name, count in pattern_counters.items():
            confidence = count / total_samples
            if confidence >= 0.05:
                pattern_stats.append(PatternStats(
                    pattern=model_patterns[pattern_name],
                    frequency=count,
                    confidence=confidence,
                    examples=pattern_examples[pattern_name],
                    pattern_type='model'
                ))

        return pattern_stats

    def _extract_generic_patterns(self, samples: List[str], column_name: str) -> List[PatternStats]:
        """Extract generic patterns from any column"""
        value_counter = Counter()
        pattern_examples = defaultdict(list)

        for sample in samples:
            sample_str = str(sample).strip()
            if len(sample_str) >= 1:
                value_counter[sample_str] += 1
                if len(pattern_examples[sample_str]) < 3:
                    pattern_examples[sample_str].append(sample)

        pattern_stats = []
        total_samples = len(samples)

        for value, count in value_counter.most_common(10):
            if count >= 2:
                confidence = count / total_samples
                pattern_stats.append(PatternStats(
                    pattern=fr'\b{re.escape(value)}\b',
                    frequency=count,
                    confidence=confidence,
                    examples=pattern_examples[value],
                    pattern_type='value'
                ))

        return pattern_stats

    def _categorize_column_values(self, samples: List[str], column_name: str) -> Dict[str, int]:
        """Categorize values in a column"""
        categories = defaultdict(int)

        # Equipment categories with keywords
        equipment_keywords = {
            'valve': ['valve', 'valves', 'ball', 'gate', 'check', 'relief', 'control', 'bypass'],
            'bearing': ['bearing', 'bearings', 'ball', 'roller', 'thrust', 'sleeve', 'assembly'],
            'motor': ['motor', 'motors', 'electric', 'servo', 'ac', 'dc', 'stepper', 'drive'],
            'switch': ['switch', 'switches', 'limit', 'pressure', 'temperature', 'safety'],
            'sensor': ['sensor', 'sensors', 'temperature', 'pressure', 'level', 'proximity'],
            'pump': ['pump', 'pumps', 'centrifugal', 'hydraulic', 'diaphragm'],
            'filter': ['filter', 'filters', 'air', 'oil', 'hydraulic', 'fuel'],
            'belt': ['belt', 'belts', 'drive', 'timing', 'v-belt', 'serpentine'],
            'cable': ['cable', 'cables', 'wire', 'wiring', 'harness', 'cord'],
            'seal': ['seal', 'seals', 'gasket', 'o-ring', 'ring'],
            'spring': ['spring', 'springs', 'compression', 'extension'],
            'gear': ['gear', 'gears', 'spur', 'bevel', 'worm'],
            'relay': ['relay', 'relays', 'control', 'time', 'power']
        }

        for sample in samples:
            sample_lower = str(sample).lower()
            categorized = False

            for category, keywords in equipment_keywords.items():
                if any(keyword in sample_lower for keyword in keywords):
                    categories[category] += 1
                    categorized = True
                    break

            if not categorized:
                categories['other'] += 1

        return dict(categories)

    def generate_search_patterns(self) -> None:
        """Generate comprehensive search patterns based on database analysis"""
        logger.info("Generating search patterns from database analysis...")

        current_time = datetime.now().isoformat()

        # Generate patterns for each column profile
        for col_name, profile in self.column_profiles.items():
            self._generate_column_search_patterns(col_name, profile)

        # Generate combination patterns
        self._generate_combination_search_patterns()

        # Generate contextual patterns
        self._generate_contextual_patterns()

        logger.info(f"Generated {len(self.extracted_patterns)} search patterns")

    def _generate_column_search_patterns(self, col_name: str, profile: ColumnProfile) -> None:
        """Generate search patterns for a specific column"""
        base_patterns = []

        if col_name == 'part_number':
            base_patterns.extend([
                (r'find\s+(?:part\s+)?(?:number\s+)?([A-Za-z0-9\-\.]{3,})', 'direct part number search'),
                (r'search\s+(?:for\s+)?(?:part\s+)?([A-Za-z0-9\-\.]{3,})', 'search part number'),
                (r'(?:show|get|display)\s+(?:me\s+)?(?:part\s+)?([A-Za-z0-9\-\.]{3,})', 'show part number'),
                (r'([A-Za-z]\d{5,})', 'direct part number pattern'),
                (r'(\d{6,})', 'numeric part number'),
                (r'#\s*([A-Za-z0-9\-\.]{3,})', 'hash part number')
            ])

        elif col_name in ['name', 'description', 'notes']:
            # Add patterns based on discovered content
            for pattern_stat in profile.common_patterns[:10]:
                if pattern_stat.pattern_type in ['word', 'phrase']:
                    word = pattern_stat.pattern.replace(r'\b', '').replace('\\', '')
                    base_patterns.extend([
                        (fr'(?:find|search|show|get)\s+.*{word.lower()}.*', f'search for {word}'),
                        (fr'(?:i\s+)?need\s+.*{word.lower()}.*', f'need {word}'),
                        (fr'(?:looking\s+for|want)\s+.*{word.lower()}.*', f'looking for {word}')
                    ])

            # Generic text search patterns
            base_patterns.extend([
                (r'(?:i\'?m\s+)?looking\s+for\s+(.+)', 'looking for text'),
                (r'(?:i\s+)?need\s+(?:a\s+|an\s+|some\s+)?(.+)', 'need text'),
                (r'find\s+(?:me\s+)?(?:a\s+|an\s+|some\s+)?(.+)', 'find text'),
                (r'what\s+(.+?)\s+do\s+(?:we\s+|you\s+)?have(?:\?|$)', 'what do we have'),
                (r'show\s+(?:me\s+)?(.+)', 'show me text')
            ])

        elif col_name in ['oem_mfg', 'manufacturer']:
            # Manufacturer-specific patterns
            for pattern_stat in profile.common_patterns[:5]:
                if pattern_stat.pattern_type == 'manufacturer':
                    mfg = pattern_stat.pattern.replace(r'\b', '').replace('\\', '')
                    base_patterns.extend([
                        (fr'(?:parts\s+)?(?:from|by|made\s+by)\s+{mfg.lower()}', f'parts from {mfg}'),
                        (fr'{mfg.lower()}\s+(?:parts|components)', f'{mfg} parts')
                    ])

            # Generic manufacturer patterns
            base_patterns.extend([
                (r'(?:parts\s+)?(?:from\s+|by\s+|made\s+by\s+)(.+?)(?:\?|$)', 'parts from manufacturer'),
                (r'(.+?)\s+(?:parts|components)(?:\?|$)', 'manufacturer parts'),
                (r'what\s+(.+?)\s+(?:parts|components)\s+do\s+(?:we\s+|you\s+)?have(?:\?|$)', 'what manufacturer parts')
            ])

        elif col_name == 'model':
            base_patterns.extend([
                (r'(?:model\s+|type\s+|series\s+)([A-Za-z0-9\-\.]+)', 'model number search'),
                (r'for\s+(?:model\s+|type\s+)?([A-Za-z0-9\-\.]+)', 'for model search'),
                (r'([A-Za-z0-9\-\.]{3,})\s+(?:model|type|series)', 'model with suffix')
            ])

        # Convert patterns to database format
        for pattern_regex, description in base_patterns:
            priority = self._calculate_pattern_priority(pattern_regex, profile)
            success_rate = self._estimate_success_rate(pattern_regex, profile)

            self.extracted_patterns.append({
                'pattern_text': pattern_regex,
                'pattern_type': 'extraction',
                'priority': priority,
                'success_rate': success_rate,
                'description': description,
                'target_column': col_name,
                'source': 'database_analysis'
            })

    def _generate_combination_search_patterns(self) -> None:
        """Generate patterns that combine multiple columns"""
        combination_patterns = [
            (r'(.+?)\s+from\s+(.+?)(?:\s+model\s+(.+?))?(?:\?|$)',
             'part from manufacturer with optional model', ['name', 'oem_mfg', 'model']),
            (r'(.+?)\s+(?:part\s+)?number\s+([A-Za-z0-9\-\.]+)',
             'description with part number', ['name', 'part_number']),
            (r'(.+?)\s+for\s+(.+?)\s+model\s+(.+?)(?:\?|$)',
             'part for equipment model', ['name', 'oem_mfg', 'model']),
            (r'show\s+(?:me\s+)?(?:all\s+)?(.+?)\s+(?:parts|components)\s+(?:for\s+)?(.+?)(?:\?|$)',
             'show parts for equipment', ['name', 'description']),
            (r'what\s+(.+?)\s+(?:parts|components)\s+do\s+we\s+have\s+from\s+(.+?)(?:\?|$)',
             'what parts do we have from manufacturer', ['name', 'oem_mfg']),
        ]

        for pattern_regex, description, target_columns in combination_patterns:
            self.extracted_patterns.append({
                'pattern_text': pattern_regex,
                'pattern_type': 'extraction',
                'priority': 1.0,
                'success_rate': 0.90,
                'description': description,
                'target_columns': target_columns,
                'source': 'combination_analysis'
            })

    def _generate_contextual_patterns(self) -> None:
        """Generate contextual patterns based on discovered equipment categories"""
        # Collect all equipment categories
        all_categories = set()
        for profile in self.column_profiles.values():
            all_categories.update(profile.data_categories.keys())

        # Remove 'other' category
        all_categories.discard('other')

        # Generate equipment-specific patterns
        for category in all_categories:
            if category in ['valve', 'bearing', 'motor', 'switch', 'sensor', 'pump', 'filter']:
                patterns = [
                    (fr'(?:find|search|show|get)\s+(?:me\s+)?(?:a\s+|an\s+|some\s+)?{category}.*',
                     f'search for {category}'),
                    (fr'(?:i\s+)?need\s+(?:a\s+|an\s+|some\s+)?{category}.*', f'need {category}'),
                    (fr'what\s+{category}.*do\s+(?:we\s+|you\s+)?have(?:\?|$)', f'what {category} do we have'),
                    (fr'{category}\s+(?:parts|components|assembly)', f'{category} parts')
                ]

                for pattern_regex, description in patterns:
                    self.extracted_patterns.append({
                        'pattern_text': pattern_regex,
                        'pattern_type': 'general',
                        'priority': 0.85,
                        'success_rate': 0.80,
                        'description': description,
                        'equipment_category': category,
                        'source': 'contextual_analysis'
                    })

    def _calculate_pattern_priority(self, pattern_regex: str, profile: ColumnProfile) -> float:
        """Calculate priority for a pattern based on column profile"""
        base_priority = 0.8

        # Boost priority for high-uniqueness columns
        if profile.unique_count / profile.total_records > 0.8:
            base_priority += 0.1

        # Boost for part_number patterns
        if profile.name == 'part_number':
            base_priority += 0.2

        # Boost for extraction patterns
        if 'extraction' in pattern_regex or '(' in pattern_regex:
            base_priority += 0.1

        return min(base_priority, 1.0)

    def _estimate_success_rate(self, pattern_regex: str, profile: ColumnProfile) -> float:
        """Estimate success rate for a pattern"""
        base_rate = 0.75

        # Higher success rate for simple patterns
        if len(pattern_regex) < 50:
            base_rate += 0.1

        # Higher success rate for part numbers
        if profile.name == 'part_number':
            base_rate += 0.15

        # Higher success rate for high-data-quality columns
        null_ratio = profile.null_count / profile.total_records
        if null_ratio < 0.1:
            base_rate += 0.1

        return min(base_rate, 0.95)

    def generate_search_intents(self) -> None:
        """Generate search intents based on database analysis"""
        logger.info("Generating search intents...")

        # Main part search intent
        self.search_intents.append({
            'name': 'FIND_PART',
            'description': 'Find parts by number, name, or characteristics',
            'priority': 1.0,
            'search_method': 'comprehensive_part_search',
            'pattern_count': len([p for p in self.extracted_patterns
                                  if p.get('target_column') in ['part_number', 'name', 'description']]),
            'source': 'database_analysis'
        })

        # Manufacturer-specific intent
        if 'oem_mfg' in self.column_profiles:
            mfg_profile = self.column_profiles['oem_mfg']
            top_manufacturers = [p.pattern for p in mfg_profile.common_patterns[:5]]

            self.search_intents.append({
                'name': 'FIND_BY_MANUFACTURER',
                'description': 'Find parts by manufacturer or brand',
                'priority': 0.90,
                'search_method': 'comprehensive_part_search',
                'top_manufacturers': top_manufacturers,
                'source': 'database_analysis'
            })

        # Equipment category intents
        equipment_categories = set()
        for profile in self.column_profiles.values():
            equipment_categories.update(profile.data_categories.keys())

        equipment_categories.discard('other')

        for category in sorted(equipment_categories):
            if category in ['valve', 'bearing', 'motor', 'switch', 'sensor', 'pump', 'filter']:
                # Calculate total parts in this category
                total_parts = sum(profile.data_categories.get(category, 0)
                                  for profile in self.column_profiles.values())

                if total_parts >= 5:  # Only create intent if we have enough parts
                    self.search_intents.append({
                        'name': f'FIND_{category.upper()}',
                        'description': f'Find {category}-specific parts and assemblies',
                        'priority': 0.95,
                        'search_method': 'comprehensive_part_search',
                        'equipment_category': category,
                        'estimated_parts': total_parts,
                        'source': 'database_analysis'
                    })

    def generate_keywords(self) -> None:
        """Generate keywords based on database content"""
        logger.info("Generating keywords from database content...")

        # Extract words from all text columns
        word_counter = Counter()

        for profile in self.column_profiles.values():
            if profile.name in ['name', 'description', 'notes']:
                for pattern_stat in profile.common_patterns:
                    if pattern_stat.pattern_type in ['word', 'phrase']:
                        word = pattern_stat.pattern.replace(r'\b', '').replace('\\', '')
                        word_counter[word.lower()] += pattern_stat.frequency

        # Generate keywords with weights
        for word, frequency in word_counter.most_common(50):
            if len(word) >= 3 and word.isalpha():
                # Calculate weight based on frequency
                weight = min(1.0 + (frequency / 100), 2.0)

                # Determine category
                category = self._categorize_keyword(word)

                self.keywords.append({
                    'keyword': word,
                    'weight': round(weight, 2),
                    'category': category,
                    'frequency': frequency,
                    'source': 'database_analysis'
                })

        # Add action keywords
        action_keywords = [
            ('find', 'action', 1.5),
            ('search', 'action', 1.4),
            ('show', 'action', 1.3),
            ('get', 'action', 1.2),
            ('need', 'action', 1.2),
            ('looking', 'action', 1.1),
            ('display', 'action', 1.0),
            ('list', 'action', 1.0)
        ]

        for keyword, category, weight in action_keywords:
            self.keywords.append({
                'keyword': keyword,
                'weight': weight,
                'category': category,
                'frequency': 0,
                'source': 'standard'
            })

    def _categorize_keyword(self, word: str) -> str:
        """Categorize a keyword"""
        equipment_words = ['valve', 'bearing', 'motor', 'switch', 'sensor', 'pump', 'filter',
                           'belt', 'cable', 'seal', 'spring', 'gear', 'relay']
        action_words = ['find', 'search', 'show', 'get', 'need', 'display', 'list']
        part_words = ['part', 'component', 'assembly', 'spare', 'replacement']

        if word in equipment_words:
            return 'equipment'
        elif word in action_words:
            return 'action'
        elif word in part_words:
            return 'object'
        else:
            return 'descriptor'

    def generate_sql_output(self) -> str:
        """Generate SQL statements for all extracted data"""
        logger.info("Generating SQL output...")

        sql_statements = []
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

        # Header comment
        sql_statements.append("-- Generated search patterns from database analysis")
        sql_statements.append(f"-- Generated at: {current_time}")
        sql_statements.append(f"-- Total patterns: {len(self.extracted_patterns)}")
        sql_statements.append(f"-- Total intents: {len(self.search_intents)}")
        sql_statements.append(f"-- Total keywords: {len(self.keywords)}")
        sql_statements.append("")

        # Search intents
        sql_statements.append("-- Insert search intents")
        intent_values = []
        for i, intent in enumerate(self.search_intents, 1):
            intent_values.append(
                f"('{intent['name']}', '{intent['description']}', {intent['priority']}, "
                f"true, '{current_time}', '{current_time}', "
                f"'{intent['name'].replace('_', ' ').title()}', '{intent['search_method']}')"
            )

        if intent_values:
            sql_statements.append(
                "INSERT INTO search_intent (name, description, priority, is_active, created_at, updated_at, display_name, search_method) VALUES"
            )
            sql_statements.append(",\n".join(intent_values) + ";\n")

        # Intent patterns
        sql_statements.append("-- Insert intent patterns")
        pattern_values = []
        for intent_id, intent in enumerate(self.search_intents, 1):
            intent_patterns = [p for p in self.extracted_patterns
                               if p.get('target_column') in ['part_number', 'name', 'description', 'oem_mfg', 'model'] or
                               intent['name'] in p.get('description', '')]

            for pattern in intent_patterns[:20]:  # Limit patterns per intent
                escaped_pattern = pattern['pattern_text'].replace("'", "''")
                pattern_values.append(
                    f"({intent_id}, '{escaped_pattern}', '{pattern['pattern_type']}', "
                    f"{pattern['priority']}, {pattern['success_rate']}, 0, true, "
                    f"'{current_time}', '{current_time}')"
                )

        if pattern_values:
            sql_statements.append(
                "INSERT INTO intent_pattern (intent_id, pattern_text, pattern_type, priority, success_rate, usage_count, is_active, created_at, updated_at) VALUES"
            )
            sql_statements.append(",\n".join(pattern_values) + ";\n")

        # Intent keywords
        sql_statements.append("-- Insert intent keywords")
        keyword_values = []
        for intent_id, intent in enumerate(self.search_intents, 1):
            # Add relevant keywords to each intent
            relevant_keywords = []
            if 'PART' in intent['name']:
                relevant_keywords.extend(['find', 'part', 'component', 'search', 'show'])
            if 'MANUFACTURER' in intent['name']:
                relevant_keywords.extend(['from', 'by', 'manufacturer', 'brand', 'made'])
            if intent.get('equipment_category'):
                relevant_keywords.append(intent['equipment_category'])

            for keyword_data in self.keywords:
                if keyword_data['keyword'] in relevant_keywords:
                    keyword_values.append(
                        f"({intent_id}, '{keyword_data['keyword']}', {keyword_data['weight']}, "
                        f"'{keyword_data['category']}', true, '{current_time}', '{current_time}')"
                    )

        if keyword_values:
            sql_statements.append(
                "INSERT INTO intent_keyword (intent_id, keyword_text, weight, keyword_type, is_active, created_at, updated_at) VALUES"
            )
            sql_statements.append(",\n".join(keyword_values) + ";\n")

        return "\n".join(sql_statements)

    def generate_analysis_report(self) -> str:
        """Generate comprehensive analysis report"""
        report = []
        report.append("=" * 80)
        report.append("DATABASE PATTERN EXTRACTION ANALYSIS REPORT")
        report.append("=" * 80)
        report.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # Database summary
        total_records = sum(profile.total_records for profile in self.column_profiles.values()) // len(self.column_profiles)
        total_unique = sum(profile.unique_count for profile in self.column_profiles.values())

        report.append("DATABASE SUMMARY")
        report.append("-" * 40)
        report.append(f"Estimated total records: {total_records:,}")
        report.append(f"Analyzed columns: {len(self.column_profiles)}")
        report.append(f"Total unique values: {total_unique:,}")
        report.append("")

        # Column profiles
        report.append("COLUMN PROFILES")
        report.append("-" * 40)
        for col_name, profile in self.column_profiles.items():
            report.append(f"Column: {col_name}")
            report.append(f"  Records: {profile.total_records:,}")
            report.append(f"  Unique values: {profile.unique_count:,}")
            report.append(f"  Null values: {profile.null_count:,}")
            report.append(f"  Average length: {profile.avg_length:.1f}")
            report.append(f"  Top patterns: {len(profile.common_patterns)}")

            if profile.data_categories:
                top_categories = sorted(profile.data_categories.items(), key=lambda x: x[1], reverse=True)[:5]
                report.append(f"  Categories: {', '.join([f'{cat}({count})' for cat, count in top_categories])}")

            report.append("")

        # Pattern summary
        pattern_types = Counter(p['pattern_type'] for p in self.extracted_patterns)
        report.append("EXTRACTED PATTERNS")
        report.append("-" * 40)
        report.append(f"Total patterns: {len(self.extracted_patterns)}")
        for pattern_type, count in pattern_types.items():
            report.append(f"  {pattern_type}: {count}")
        report.append("")

        # Sample patterns
        report.append("SAMPLE PATTERNS")
        report.append("-" * 40)
        for i, pattern in enumerate(self.extracted_patterns[:10]):
            report.append(f"{i + 1}. {pattern['pattern_text']}")
            report.append(f"   Type: {pattern['pattern_type']}, Priority: {pattern['priority']}")
            report.append(f"   Target: {pattern.get('target_column', 'multiple')}")
            report.append("")

        # Intent summary
        report.append("GENERATED INTENTS")
        report.append("-" * 40)
        for intent in self.search_intents:
            report.append(f"Intent: {intent['name']}")
            report.append(f"  Priority: {intent['priority']}")
            report.append(f"  Description: {intent['description']}")
            if 'estimated_parts' in intent:
                report.append(f"  Estimated matching parts: {intent['estimated_parts']}")
            report.append("")

        # Keywords summary
        report.append("KEYWORD ANALYSIS")
        report.append("-" * 40)
        keyword_categories = Counter(k['category'] for k in self.keywords)
        report.append(f"Total keywords: {len(self.keywords)}")
        for category, count in keyword_categories.items():
            report.append(f"  {category}: {count}")

        top_keywords = sorted(self.keywords, key=lambda x: x['weight'], reverse=True)[:15]
        report.append("\nTop keywords by weight:")
        for kw in top_keywords:
            report.append(f"  {kw['keyword']} ({kw['category']}, {kw['weight']})")

        return "\n".join(report)

    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Extract search patterns from database')
    parser.add_argument('--db-url', required=True, help='Database URL')
    parser.add_argument('--table', default='part', help='Table name to analyze')
    parser.add_argument('--sample-size', type=int, default=10000, help='Sample size for analysis')
    parser.add_argument('--output-sql', default='extracted_patterns.sql', help='Output SQL file')
    parser.add_argument('--output-report', default='extraction_report.txt', help='Output analysis report')
    parser.add_argument('--output-json', default='extracted_data.json', help='Output JSON data file')

    args = parser.parse_args()

    # Initialize extractor
    extractor = DatabasePatternExtractor(args.db_url)

    if not extractor.connect():
        logger.error("Failed to connect to database")
        return 1

    try:
        # Analyze the database
        logger.info(f"Analyzing table: {args.table}")
        extractor.analyze_part_table(args.table, args.sample_size)

        # Generate patterns and intents
        extractor.generate_search_patterns()
        extractor.generate_search_intents()
        extractor.generate_keywords()

        # Generate outputs
        sql_output = extractor.generate_sql_output()
        analysis_report = extractor.generate_analysis_report()

        # Save SQL file
        with open(args.output_sql, 'w') as f:
            f.write(sql_output)
        logger.info(f"SQL output saved to: {args.output_sql}")

        # Save analysis report
        with open(args.output_report, 'w') as f:
            f.write(analysis_report)
        logger.info(f"Analysis report saved to: {args.output_report}")

        # Save JSON data
        json_data = {
            'column_profiles': {name: asdict(profile) for name, profile in extractor.column_profiles.items()},
            'extracted_patterns': extractor.extracted_patterns,
            'search_intents': extractor.search_intents,
            'keywords': extractor.keywords,
            'generation_time': datetime.now().isoformat()
        }

        with open(args.output_json, 'w') as f:
            json.dump(json_data, f, indent=2, default=str)
        logger.info(f"JSON data saved to: {args.output_json}")

        # Print summary
        print("\n" + "=" * 60)
        print("DATABASE PATTERN EXTRACTION COMPLETE")
        print("=" * 60)
        print(f"📊 Analyzed {len(extractor.column_profiles)} columns")
        print(f"🔍 Extracted {len(extractor.extracted_patterns)} search patterns")
        print(f"🎯 Generated {len(extractor.search_intents)} search intents")
        print(f"🔑 Found {len(extractor.keywords)} keywords")
        print(f"📝 SQL file: {args.output_sql}")
        print(f"📋 Report file: {args.output_report}")
        print(f"📄 JSON file: {args.output_json}")

        # Show top findings
        print("\n🔥 TOP FINDINGS:")

        # Most common equipment categories
        all_categories = {}
        for profile in extractor.column_profiles.values():
            for cat, count in profile.data_categories.items():
                all_categories[cat] = all_categories.get(cat, 0) + count

        top_categories = sorted(all_categories.items(), key=lambda x: x[1], reverse=True)[:5]
        print("📦 Equipment categories:")
        for cat, count in top_categories:
            print(f"   {cat}: {count} parts")

        # Most valuable patterns
        top_patterns = sorted(extractor.extracted_patterns, key=lambda x: x['priority'] * x['success_rate'], reverse=True)[:3]
        print("🎯 High-value patterns:")
        for i, pattern in enumerate(top_patterns, 1):
            print(f"   {i}. {pattern['pattern_text'][:50]}...")

        print("\n✅ Ready to integrate with your search system!")

        return 0

    except Exception as e:
        logger.error(f"Error during extraction: {e}")
        return 1

    finally:
        extractor.close()


if __name__ == "__main__":
    exit(main())