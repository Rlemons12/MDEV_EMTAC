#!/usr/bin/env python3
"""
Part Table Search Pattern Generator - COMPLETELY CLEAN VERSION

This script analyzes the part table to generate comprehensive search patterns,
keywords, and intents that enable natural language searching across all columns.
"""

import re
import json
import logging
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from collections import defaultdict, Counter
import argparse
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ColumnAnalysis:
    """Analysis results for a single column"""
    column_name: str
    data_type: str
    sample_values: List[str]
    patterns: List[str]
    common_words: List[str]
    value_categories: Dict[str, List[str]]
    search_complexity: str


@dataclass
class SearchPattern:
    """A search pattern definition"""
    pattern_text: str
    pattern_type: str
    priority: float
    success_rate: float
    description: str
    target_columns: List[str]


@dataclass
class SearchIntent:
    """A search intent definition"""
    name: str
    description: str
    priority: float
    search_method: str
    patterns: List[SearchPattern]
    keywords: List[str]


class PartTableAnalyzer:
    """Analyzes part table structure and generates search patterns"""

    def __init__(self):
        self.column_analyses: Dict[str, ColumnAnalysis] = {}
        self.generated_patterns: List[SearchPattern] = []
        self.generated_intents: List[SearchIntent] = []
        self.generated_keywords: List[Tuple[str, str, float]] = []

    def analyze_sample_data(self, sample_data: List[Dict[str, Any]]) -> None:
        """Analyze sample part data to understand patterns"""
        logger.info(f"Analyzing {len(sample_data)} sample parts...")

        if not sample_data:
            logger.warning("No sample data provided, using default patterns")
            self._create_default_column_analyses()
            return

        columns = list(sample_data[0].keys()) if sample_data else []
        logger.info(f"Found columns: {columns}")

        for column in columns:
            self._analyze_column(column, sample_data)

    def _analyze_column(self, column_name: str, data: List[Dict[str, Any]]) -> None:
        """Analyze a specific column to understand its data patterns"""
        logger.info(f"Analyzing column: {column_name}")

        values = [str(row.get(column_name, '')).strip()
                  for row in data
                  if row.get(column_name) is not None and str(row.get(column_name)).strip()]

        if not values:
            logger.warning(f"No valid values found for column {column_name}")
            return

        sample_values = values[:100]
        data_type = self._infer_data_type(sample_values)
        patterns = self._extract_patterns(sample_values, column_name)
        common_words = self._extract_common_words(sample_values)
        value_categories = self._categorize_values(sample_values, column_name)
        search_complexity = self._assess_search_complexity(sample_values, column_name)

        self.column_analyses[column_name] = ColumnAnalysis(
            column_name=column_name,
            data_type=data_type,
            sample_values=sample_values[:10],
            patterns=patterns,
            common_words=common_words,
            value_categories=value_categories,
            search_complexity=search_complexity
        )

        logger.info(f"Column {column_name}: {data_type}, {len(patterns)} patterns, "
                    f"{len(common_words)} common words, complexity: {search_complexity}")

    def _infer_data_type(self, values: List[str]) -> str:
        """Infer the data type of a column from sample values"""
        if not values:
            return 'unknown'

        numeric_count = sum(1 for v in values if v.isdigit())
        if numeric_count > len(values) * 0.8:
            return 'numeric'

        part_number_patterns = [
            r'^[A-Z0-9]{2,}[-\.][A-Z0-9]+$',
            r'^[A-Z]{2,}\d{3,}$',
            r'^\d{4,}[-\.][A-Z0-9]+$',
            r'^[A-Z]\d{5,}$'
        ]

        part_number_count = 0
        for value in values:
            if any(re.match(pattern, value.upper()) for pattern in part_number_patterns):
                part_number_count += 1

        if part_number_count > len(values) * 0.6:
            return 'part_number'

        avg_length = sum(len(v) for v in values) / len(values)
        if avg_length > 20:
            return 'long_text'
        elif avg_length > 5:
            return 'short_text'
        else:
            return 'code'

    def _extract_patterns(self, values: List[str], column_name: str) -> List[str]:
        """Extract regex patterns from column values"""
        patterns = []

        if 'part' in column_name.lower() and 'number' in column_name.lower():
            patterns.extend([
                r'[A-Z0-9]{2,}[-\.][A-Z0-9]+',
                r'[A-Z]{2,}\d{3,}',
                r'\d{4,}[-\.][A-Z0-9]+',
                r'[A-Z]\d{5,}'
            ])
        elif column_name.lower() in ['name', 'description', 'notes']:
            word_patterns = set()
            for value in values[:20]:
                words = re.findall(r'\b[A-Za-z]{3,}\b', value)
                word_patterns.update(words[:3])
            patterns.extend([fr'\b{word}\b' for word in list(word_patterns)[:10]])
        elif 'model' in column_name.lower():
            patterns.extend([
                r'[A-Z0-9\-\.]+',
                r'[A-Z]{2,}[-\.]\d+',
                r'\d+[A-Z]+',
            ])
        elif column_name.lower() in ['oem_mfg', 'manufacturer']:
            manufacturers = set()
            for value in values[:20]:
                if len(value) > 2 and len(value) < 30:
                    manufacturers.add(value.upper())
            patterns.extend([fr'\b{mfg}\b' for mfg in list(manufacturers)[:10]])

        return patterns

    def _extract_common_words(self, values: List[str]) -> List[str]:
        """Extract common words from column values"""
        word_counter = Counter()

        for value in values:
            words = re.findall(r'\b[A-Za-z]{3,}\b', value.upper())
            word_counter.update(words)

        return [word for word, count in word_counter.most_common(20)
                if count > 1 and len(word) >= 3]

    def _categorize_values(self, values: List[str], column_name: str) -> Dict[str, List[str]]:
        """Categorize values into semantic groups"""
        categories = defaultdict(list)

        equipment_terms = {
            'valve': ['valve', 'valves', 'ball', 'gate', 'check', 'relief', 'control'],
            'bearing': ['bearing', 'bearings', 'ball', 'roller', 'thrust'],
            'motor': ['motor', 'motors', 'electric', 'servo', 'ac', 'dc'],
            'pump': ['pump', 'pumps', 'centrifugal', 'hydraulic'],
            'sensor': ['sensor', 'sensors', 'temperature', 'pressure', 'level'],
            'switch': ['switch', 'switches', 'limit', 'pressure', 'safety'],
            'belt': ['belt', 'belts', 'drive', 'timing', 'v-belt'],
            'filter': ['filter', 'filters', 'air', 'oil', 'hydraulic'],
            'cable': ['cable', 'cables', 'wire', 'cord'],
            'seal': ['seal', 'seals', 'gasket', 'o-ring'],
            'spring': ['spring', 'springs', 'compression', 'extension'],
            'gear': ['gear', 'gears', 'spur', 'bevel', 'worm'],
            'relay': ['relay', 'relays', 'control', 'time', 'power']
        }

        for value in values[:50]:
            value_lower = value.lower()
            for category, terms in equipment_terms.items():
                if any(term in value_lower for term in terms):
                    categories[category].append(value)
                    break
            else:
                categories['other'].append(value)

        return {k: v[:10] for k, v in categories.items() if v}

    def _assess_search_complexity(self, values: List[str], column_name: str) -> str:
        """Assess how complex searching this column might be"""
        if not values:
            return 'simple'

        avg_length = sum(len(v) for v in values) / len(values)
        unique_ratio = len(set(values)) / len(values)

        if avg_length < 15 and unique_ratio > 0.8:
            return 'simple'
        elif avg_length > 50 or 'description' in column_name.lower() or 'notes' in column_name.lower():
            return 'complex'
        else:
            return 'medium'

    def _create_default_column_analyses(self) -> None:
        """Create default column analyses for common part table columns"""
        logger.info("Creating default column analyses...")

        default_columns = {
            'part_number': {
                'data_type': 'part_number',
                'patterns': [r'[A-Z0-9]{2,}[-\.][A-Z0-9]+', r'[A-Z]{2,}\d{3,}', r'[A-Z]\d{5,}'],
                'common_words': ['VALVE', 'BEARING', 'MOTOR', 'SWITCH'],
                'search_complexity': 'simple'
            },
            'name': {
                'data_type': 'short_text',
                'patterns': [r'\bVALVE\b', r'\bBEARING\b', r'\bMOTOR\b', r'\bASSEMBLY\b'],
                'common_words': ['VALVE', 'BEARING', 'MOTOR', 'ASSEMBLY', 'SWITCH'],
                'search_complexity': 'medium'
            },
            'oem_mfg': {
                'data_type': 'short_text',
                'patterns': [r'\bABB\b', r'\bSIEMENS\b', r'\bEMERSON\b'],
                'common_words': ['ABB', 'SIEMENS', 'EMERSON', 'HONEYWELL'],
                'search_complexity': 'simple'
            },
            'model': {
                'data_type': 'code',
                'patterns': [r'[A-Z0-9\-\.]+', r'[A-Z]{2,}[-\.]\d+'],
                'common_words': ['SERIES', 'MODEL', 'TYPE'],
                'search_complexity': 'simple'
            },
            'notes': {
                'data_type': 'long_text',
                'patterns': [r'\b[A-Z]{3,}\b'],
                'common_words': ['REPLACEMENT', 'COMPATIBLE', 'ORIGINAL'],
                'search_complexity': 'complex'
            }
        }

        for col_name, data in default_columns.items():
            self.column_analyses[col_name] = ColumnAnalysis(
                column_name=col_name,
                data_type=data['data_type'],
                sample_values=['SAMPLE_VALUE_1', 'SAMPLE_VALUE_2'],
                patterns=data['patterns'],
                common_words=data['common_words'],
                value_categories={'equipment': ['valve', 'bearing', 'motor']},
                search_complexity=data['search_complexity']
            )

    def generate_search_patterns(self) -> None:
        """Generate comprehensive search patterns for all columns"""
        logger.info("Generating search patterns...")

        for column_name, analysis in self.column_analyses.items():
            self._generate_column_patterns(column_name, analysis)

        self._generate_combination_patterns()
        self._generate_fuzzy_patterns()

        logger.info(f"Generated {len(self.generated_patterns)} search patterns")

    def _generate_column_patterns(self, column_name: str, analysis: ColumnAnalysis) -> None:
        """Generate search patterns for a specific column"""
        patterns = []

        if column_name == 'part_number':
            patterns.extend(self._generate_part_number_patterns())
        elif column_name in ['name', 'description', 'notes']:
            patterns.extend(self._generate_text_search_patterns(column_name, analysis))
        elif column_name in ['oem_mfg', 'manufacturer']:
            patterns.extend(self._generate_manufacturer_patterns(analysis))
        elif column_name == 'model':
            patterns.extend(self._generate_model_patterns(analysis))
        else:
            patterns.extend(self._generate_generic_patterns(column_name, analysis))

        self.generated_patterns.extend(patterns)

    def _generate_part_number_patterns(self) -> List[SearchPattern]:
        """Generate patterns specifically for part number searches"""
        return [
            SearchPattern(
                pattern_text=r'find\s+(?:this\s+|the\s+)?part\s+(?:number\s+)?([A-Za-z0-9\-\.]{3,})',
                pattern_type='extraction',
                priority=1.00,
                success_rate=0.95,
                description='Find part by number with various prefixes',
                target_columns=['part_number']
            ),
            SearchPattern(
                pattern_text=r'search\s+(?:for\s+)?part\s+(?:number\s+)?([A-Za-z0-9\-\.]{3,})',
                pattern_type='extraction',
                priority=1.00,
                success_rate=0.95,
                description='Search for part by number',
                target_columns=['part_number']
            ),
            SearchPattern(
                pattern_text=r'(?:i\s+)?need\s+(?:the\s+)?part\s+number\s+for\s+(.+?)(?:\?|$)',
                pattern_type='extraction',
                priority=1.00,
                success_rate=0.95,
                description='Need part number for description',
                target_columns=['name', 'description']
            ),
            SearchPattern(
                pattern_text=r'what\s+(?:is\s+)?(?:the\s+)?part\s+number\s+for\s+(.+?)(?:\?|$)',
                pattern_type='extraction',
                priority=1.00,
                success_rate=0.95,
                description='What is part number for description',
                target_columns=['name', 'description']
            ),
            SearchPattern(
                pattern_text=r'([A-Za-z]\d{5,})',
                pattern_type='extraction',
                priority=0.80,
                success_rate=0.85,
                description='Direct part number pattern (letter + 5+ digits)',
                target_columns=['part_number']
            ),
            SearchPattern(
                pattern_text=r'(\d{6,})',
                pattern_type='extraction',
                priority=0.70,
                success_rate=0.80,
                description='Direct numeric part number (6+ digits)',
                target_columns=['part_number']
            ),
            SearchPattern(
                pattern_text=r'#\s*([A-Za-z0-9\-\.]{3,})',
                pattern_type='extraction',
                priority=0.90,
                success_rate=0.90,
                description='Part number with hash prefix',
                target_columns=['part_number']
            )
        ]

    def _generate_text_search_patterns(self, column_name: str, analysis: ColumnAnalysis) -> List[SearchPattern]:
        """Generate patterns for text-based columns like name, description"""
        patterns = []

        base_patterns = [
            (r'(?:i\'?m\s+)?looking\s+for\s+(?:a\s+|an\s+|some\s+)?(.+)', 'looking for text search'),
            (r'(?:i\s+)?need\s+(?:a\s+|an\s+|some\s+)?(.+)', 'need text search'),
            (r'find\s+(?:me\s+)?(?:a\s+|an\s+|some\s+)?(.+)', 'find text search'),
            (r'search\s+(?:for\s+)?(.+)', 'search for text'),
            (r'show\s+(?:me\s+)?(.+)', 'show me text search'),
            (r'what\s+(.+?)\s+do\s+(?:we\s+|you\s+)?have(?:\?|$)', 'what do we have search')
        ]

        for pattern_text, description in base_patterns:
            patterns.append(SearchPattern(
                pattern_text=pattern_text,
                pattern_type='extraction',
                priority=0.90,
                success_rate=0.85,
                description=f'{description} for {column_name}',
                target_columns=[column_name]
            ))

        for word in analysis.common_words[:10]:
            patterns.append(SearchPattern(
                pattern_text=fr'(?:find\s+|search\s+|show\s+|get\s+).*{word.lower()}.*',
                pattern_type='general',
                priority=0.80,
                success_rate=0.75,
                description=f'Search for {word} in {column_name}',
                target_columns=[column_name]
            ))

        return patterns

    def _generate_manufacturer_patterns(self, analysis: ColumnAnalysis) -> List[SearchPattern]:
        """Generate patterns for manufacturer/OEM searches"""
        patterns = [
            SearchPattern(
                pattern_text=r'(?:parts\s+)?(?:from\s+|by\s+|made\s+by\s+)(.+?)(?:\?|$)',
                pattern_type='extraction',
                priority=1.00,
                success_rate=0.90,
                description='Parts from manufacturer',
                target_columns=['oem_mfg']
            ),
            SearchPattern(
                pattern_text=r'(.+?)\s+(?:parts|components|products)(?:\?|$)',
                pattern_type='extraction',
                priority=0.90,
                success_rate=0.85,
                description='Manufacturer parts search',
                target_columns=['oem_mfg']
            ),
            SearchPattern(
                pattern_text=r'what\s+(.+?)\s+(?:parts|components)\s+do\s+(?:we\s+|you\s+)?have(?:\?|$)',
                pattern_type='extraction',
                priority=0.90,
                success_rate=0.85,
                description='What manufacturer parts do we have',
                target_columns=['oem_mfg']
            )
        ]

        for word in analysis.common_words[:5]:
            if len(word) > 2:
                patterns.append(SearchPattern(
                    pattern_text=fr'\b{word.lower()}\b.*(?:parts|components)',
                    pattern_type='general',
                    priority=0.85,
                    success_rate=0.80,
                    description=f'Search for {word} parts',
                    target_columns=['oem_mfg']
                ))

        return patterns

    def _generate_model_patterns(self, analysis: ColumnAnalysis) -> List[SearchPattern]:
        """Generate patterns for model number searches"""
        return [
            SearchPattern(
                pattern_text=r'(?:model\s+|type\s+|series\s+)([A-Za-z0-9\-\.]+)',
                pattern_type='extraction',
                priority=1.00,
                success_rate=0.90,
                description='Model number search',
                target_columns=['model']
            ),
            SearchPattern(
                pattern_text=r'for\s+(?:model\s+|type\s+)?([A-Za-z0-9\-\.]+)',
                pattern_type='extraction',
                priority=0.90,
                success_rate=0.85,
                description='For model search',
                target_columns=['model']
            ),
            SearchPattern(
                pattern_text=r'([A-Za-z0-9\-\.]{4,})\s+(?:model|type|series)',
                pattern_type='extraction',
                priority=0.85,
                success_rate=0.80,
                description='Model with suffix',
                target_columns=['model']
            )
        ]

    def _generate_generic_patterns(self, column_name: str, analysis: ColumnAnalysis) -> List[SearchPattern]:
        """Generate generic patterns for any column"""
        patterns = []
        column_friendly = column_name.replace('_', ' ')

        patterns.extend([
            SearchPattern(
                pattern_text=fr'(?:show\s+|find\s+|search\s+)(?:me\s+)?(?:the\s+)?{column_friendly}\s+(.+)',
                pattern_type='extraction',
                priority=0.80,
                success_rate=0.75,
                description=f'Search {column_friendly}',
                target_columns=[column_name]
            ),
            SearchPattern(
                pattern_text=fr'what\s+{column_friendly}\s+(.+?)(?:\?|$)',
                pattern_type='extraction',
                priority=0.80,
                success_rate=0.75,
                description=f'What {column_friendly} query',
                target_columns=[column_name]
            )
        ])

        return patterns

    def _generate_combination_patterns(self) -> None:
        """Generate patterns that combine multiple columns"""
        combination_patterns = [
            SearchPattern(
                pattern_text=r'(.+?)\s+from\s+(.+?)(?:\s+model\s+(.+?))?(?:\?|$)',
                pattern_type='extraction',
                priority=1.00,
                success_rate=0.90,
                description='Part from manufacturer with optional model',
                target_columns=['name', 'oem_mfg', 'model']
            ),
            SearchPattern(
                pattern_text=r'(.+?)\s+(?:part\s+)?number\s+([A-Za-z0-9\-\.]+)',
                pattern_type='extraction',
                priority=1.00,
                success_rate=0.95,
                description='Description with part number',
                target_columns=['name', 'part_number']
            ),
            SearchPattern(
                pattern_text=r'(.+?)\s+for\s+(.+?)\s+model\s+(.+?)(?:\?|$)',
                pattern_type='extraction',
                priority=0.95,
                success_rate=0.88,
                description='Part for equipment model',
                target_columns=['name', 'oem_mfg', 'model']
            ),
            SearchPattern(
                pattern_text=r'show\s+(?:me\s+)?(?:all\s+)?(.+?)\s+(?:parts|components)\s+(?:for\s+)?(.+?)(?:\?|$)',
                pattern_type='extraction',
                priority=0.90,
                success_rate=0.85,
                description='Show parts for equipment',
                target_columns=['name', 'description']
            )
        ]

        self.generated_patterns.extend(combination_patterns)

    def _generate_fuzzy_patterns(self) -> None:
        """Generate fuzzy search patterns for partial matches"""
        fuzzy_patterns = [
            SearchPattern(
                pattern_text=r'.*(?:similar|like|close)\s+to\s+(.+)',
                pattern_type='fuzzy',
                priority=0.70,
                success_rate=0.65,
                description='Similar to search',
                target_columns=['name', 'description']
            ),
            SearchPattern(
                pattern_text=r'(?:anything|parts)\s+(?:similar|like)\s+(.+)',
                pattern_type='fuzzy',
                priority=0.70,
                success_rate=0.65,
                description='Anything similar search',
                target_columns=['name', 'description']
            ),
            SearchPattern(
                pattern_text=r'alternative\s+(?:to\s+|for\s+)(.+)',
                pattern_type='fuzzy',
                priority=0.75,
                success_rate=0.70,
                description='Alternative part search',
                target_columns=['name', 'description']
            )
        ]

        self.generated_patterns.extend(fuzzy_patterns)

    def generate_search_intents(self) -> None:
        """Generate high-level search intents"""
        logger.info("Generating search intents...")

        part_patterns = [p for p in self.generated_patterns
                         if any(col in ['part_number', 'name', 'description']
                                for col in p.target_columns)]

        self.generated_intents.append(SearchIntent(
            name='FIND_PART',
            description='Find parts by number, name, or characteristics',
            priority=1.00,
            search_method='comprehensive_part_search',
            patterns=part_patterns,
            keywords=['find', 'part', 'component', 'number', 'search']
        ))

        mfg_patterns = [p for p in self.generated_patterns
                        if 'oem_mfg' in p.target_columns]

        self.generated_intents.append(SearchIntent(
            name='FIND_BY_MANUFACTURER',
            description='Find parts by manufacturer or brand',
            priority=0.90,
            search_method='comprehensive_part_search',
            patterns=mfg_patterns,
            keywords=['from', 'by', 'made', 'manufacturer', 'brand', 'oem']
        ))

        equipment_categories = self._get_equipment_categories()
        for category, terms in equipment_categories.items():
            category_patterns = [p for p in self.generated_patterns
                                 if any(term.lower() in p.pattern_text.lower()
                                        for term in terms)]

            if category_patterns:
                self.generated_intents.append(SearchIntent(
                    name=f'FIND_{category.upper()}',
                    description=f'Find {category}-specific parts and assemblies',
                    priority=0.95,
                    search_method='comprehensive_part_search',
                    patterns=category_patterns,
                    keywords=terms
                ))

        model_patterns = [p for p in self.generated_patterns
                          if 'model' in p.target_columns]

        if model_patterns:
            self.generated_intents.append(SearchIntent(
                name='FIND_BY_MODEL',
                description='Find parts by model number or series',
                priority=0.85,
                search_method='comprehensive_part_search',
                patterns=model_patterns,
                keywords=['model', 'type', 'series', 'version']
            ))

        logger.info(f"Generated {len(self.generated_intents)} search intents")

    def _get_equipment_categories(self) -> Dict[str, List[str]]:
        """Get equipment categories and their associated terms"""
        return {
            'valve': ['valve', 'valves', 'ball', 'gate', 'check', 'relief', 'control', 'bypass'],
            'bearing': ['bearing', 'bearings', 'ball', 'roller', 'thrust', 'sleeve'],
            'motor': ['motor', 'motors', 'electric', 'servo', 'ac', 'dc', 'stepper'],
            'switch': ['switch', 'switches', 'limit', 'pressure', 'temperature', 'safety'],
            'belt': ['belt', 'belts', 'drive', 'timing', 'v-belt', 'serpentine'],
            'cable': ['cable', 'cables', 'wire', 'cord', 'harness'],
            'sensor': ['sensor', 'sensors', 'temperature', 'pressure', 'level', 'proximity'],
            'seal': ['seal', 'seals', 'gasket', 'o-ring', 'ring'],
            'relay': ['relay', 'relays', 'control', 'time', 'power'],
            'pump': ['pump', 'pumps', 'centrifugal', 'hydraulic', 'diaphragm'],
            'spring': ['spring', 'springs', 'compression', 'extension', 'torsion'],
            'filter': ['filter', 'filters', 'air', 'oil', 'hydraulic', 'fuel'],
            'gear': ['gear', 'gears', 'spur', 'bevel', 'worm', 'planetary'],
            'coupling': ['coupling', 'couplings', 'flexible', 'rigid'],
            'tube': ['tube', 'tubes', 'tubing', 'pipe', 'hose']
        }

    def generate_keywords(self) -> None:
        """Generate keywords for search intents"""
        logger.info("Generating keywords...")

        action_keywords = [
            ('find', 'action', 1.50),
            ('search', 'action', 1.40),
            ('looking', 'action', 1.30),
            ('need', 'action', 1.20),
            ('get', 'action', 1.10),
            ('show', 'action', 1.10),
            ('locate', 'action', 1.10),
            ('retrieve', 'action', 1.00),
            ('display', 'action', 1.00),
            ('list', 'action', 1.00)
        ]

        object_keywords = [
            ('part', 'object', 1.50),
            ('parts', 'object', 1.50),
            ('component', 'object', 1.30),
            ('components', 'object', 1.30),
            ('number', 'object', 1.20),
            ('spare', 'object', 1.20),
            ('spares', 'object', 1.20),
            ('replacement', 'object', 1.10),
            ('item', 'object', 1.00),
            ('piece', 'object', 1.00)
        ]

        equipment_keywords = []
        for analysis in self.column_analyses.values():
            for word in analysis.common_words[:10]:
                equipment_keywords.append((word.lower(), 'equipment', 1.00))

        seen_equipment = set()
        unique_equipment = []
        for keyword, category, weight in equipment_keywords:
            if keyword not in seen_equipment:
                seen_equipment.add(keyword)
                unique_equipment.append((keyword, category, weight))

        context_keywords = [
            ('from', 'context', 0.80),
            ('by', 'context', 0.80),
            ('made', 'context', 0.80),
            ('manufacturer', 'context', 0.90),
            ('brand', 'context', 0.80),
            ('oem', 'context', 0.90),
            ('have', 'context', 0.80),
            ('available', 'context', 0.80),
            ('stock', 'context', 0.80),
            ('inventory', 'context', 0.80)
        ]

        question_keywords = [
            ('what', 'question', 0.80),
            ('which', 'question', 0.80),
            ('where', 'question', 0.70),
            ('how', 'question', 0.60)
        ]

        self.generated_keywords.extend(action_keywords)
        self.generated_keywords.extend(object_keywords)
        self.generated_keywords.extend(unique_equipment[:20])
        self.generated_keywords.extend(context_keywords)
        self.generated_keywords.extend(question_keywords)

        logger.info(f"Generated {len(self.generated_keywords)} keywords")

    def generate_sql_output(self) -> str:
        """Generate SQL INSERT statements for all generated data"""
        logger.info("Generating SQL output...")

        sql_statements = []
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

        sql_statements.append("-- Insert search intents")
        sql_statements.append(
            "INSERT INTO search_intent (name, description, priority, is_active, created_at, updated_at, display_name, search_method) VALUES")

        intent_values = []
        for i, intent in enumerate(self.generated_intents, 1):
            intent_values.append(
                f"('{intent.name}', '{intent.description}', {intent.priority}, true, "
                f"'{current_time}', '{current_time}', '{intent.name.replace('_', ' ').title()}', '{intent.search_method}')"
            )

        sql_statements.append(",\n".join(intent_values) + ";\n")

        sql_statements.append("-- Insert intent patterns")
        sql_statements.append(
            "INSERT INTO intent_pattern (intent_id, pattern_text, pattern_type, priority, success_rate, usage_count, is_active, created_at, updated_at) VALUES")

        pattern_values = []
        pattern_id = 1
        for intent_id, intent in enumerate(self.generated_intents, 1):
            for pattern in intent.patterns:
                escaped_pattern = pattern.pattern_text.replace("'", "''")
                pattern_values.append(
                    f"({intent_id}, '{escaped_pattern}', '{pattern.pattern_type}', "
                    f"{pattern.priority}, {pattern.success_rate}, 0, true, '{current_time}', '{current_time}')"
                )
                pattern_id += 1

        sql_statements.append(",\n".join(pattern_values) + ";\n")

        sql_statements.append("-- Insert intent keywords")
        sql_statements.append(
            "INSERT INTO intent_keyword (intent_id, keyword_text, weight, keyword_type, is_active, created_at, updated_at) VALUES")

        keyword_values = []
        for intent_id, intent in enumerate(self.generated_intents, 1):
            for keyword in intent.keywords:
                keyword_type = self._determine_keyword_type(keyword)
                weight = self._get_keyword_weight(keyword)

                keyword_values.append(
                    f"({intent_id}, '{keyword}', {weight}, '{keyword_type}', true, '{current_time}', '{current_time}')"
                )

        if self.generated_intents:
            for keyword, category, weight in self.generated_keywords:
                keyword_values.append(
                    f"(1, '{keyword}', {weight}, '{category}', true, '{current_time}', '{current_time}')"
                )

        sql_statements.append(",\n".join(keyword_values) + ";\n")

        return "\n".join(sql_statements)

    def _determine_keyword_type(self, keyword: str) -> str:
        """Determine the type of a keyword"""
        action_words = ['find', 'search', 'get', 'show', 'locate', 'display', 'list']
        question_words = ['what', 'which', 'where', 'how', 'when']
        object_words = ['part', 'component', 'number', 'spare', 'replacement']

        if keyword.lower() in action_words:
            return 'action'
        elif keyword.lower() in question_words:
            return 'question'
        elif keyword.lower() in object_words:
            return 'object'
        else:
            return 'equipment'

    def _get_keyword_weight(self, keyword: str) -> float:
        """Get the weight for a keyword"""
        for kw, category, weight in self.generated_keywords:
            if kw == keyword.lower():
                return weight

        action_words = ['find', 'search', 'get', 'show']
        if keyword.lower() in action_words:
            return 1.2
        else:
            return 1.0

    def generate_analysis_report(self) -> str:
        """Generate a comprehensive analysis report"""
        report = []
        report.append("=" * 80)
        report.append("PART TABLE SEARCH PATTERN ANALYSIS REPORT")
        report.append("=" * 80)
        report.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Analyzed columns: {len(self.column_analyses)}")
        report.append(f"Generated patterns: {len(self.generated_patterns)}")
        report.append(f"Generated intents: {len(self.generated_intents)}")
        report.append(f"Generated keywords: {len(self.generated_keywords)}")
        report.append("")

        report.append("COLUMN ANALYSIS SUMMARY")
        report.append("-" * 40)
        for col_name, analysis in self.column_analyses.items():
            report.append(f"Column: {col_name}")
            report.append(f"  Data Type: {analysis.data_type}")
            report.append(f"  Patterns: {len(analysis.patterns)}")
            report.append(f"  Common Words: {len(analysis.common_words)}")
            report.append(f"  Complexity: {analysis.search_complexity}")
            if analysis.common_words:
                report.append(f"  Top Words: {', '.join(analysis.common_words[:5])}")
            report.append("")

        pattern_types = {}
        for pattern in self.generated_patterns:
            pattern_types.setdefault(pattern.pattern_type, 0)
            pattern_types[pattern.pattern_type] += 1

        report.append("PATTERN TYPE SUMMARY")
        report.append("-" * 40)
        for pattern_type, count in pattern_types.items():
            report.append(f"{pattern_type}: {count} patterns")
        report.append("")

        report.append("SEARCH INTENT SUMMARY")
        report.append("-" * 40)
        for intent in self.generated_intents:
            report.append(f"Intent: {intent.name}")
            report.append(f"  Priority: {intent.priority}")
            report.append(f"  Patterns: {len(intent.patterns)}")
            report.append(f"  Keywords: {len(intent.keywords)}")
            report.append(f"  Description: {intent.description}")
            report.append("")

        report.append("SAMPLE PATTERNS")
        report.append("-" * 40)
        for i, pattern in enumerate(self.generated_patterns[:10]):
            report.append(f"{i + 1}. {pattern.pattern_text}")
            report.append(f"   Type: {pattern.pattern_type}, Priority: {pattern.priority}")
            report.append(f"   Targets: {', '.join(pattern.target_columns)}")
            report.append("")

        return "\n".join(report)


def load_sample_data_from_db(db_url: str) -> List[Dict[str, Any]]:
    """Load sample data from database"""
    try:
        import psycopg2
        from urllib.parse import urlparse

        parsed = urlparse(db_url)
        conn = psycopg2.connect(
            host=parsed.hostname,
            port=parsed.port,
            database=parsed.path[1:],
            user=parsed.username,
            password=parsed.password
        )

        cursor = conn.cursor()
        cursor.execute("SELECT * FROM part LIMIT 1000")

        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()

        sample_data = []
        for row in rows:
            sample_data.append(dict(zip(columns, row)))

        cursor.close()
        conn.close()

        logger.info(f"Loaded {len(sample_data)} sample records from database")
        return sample_data

    except Exception as e:
        logger.error(f"Failed to load data from database: {e}")
        return []


def load_sample_data_from_file(file_path: str) -> List[Dict[str, Any]]:
    """Load sample data from JSON file"""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        if isinstance(data, list):
            logger.info(f"Loaded {len(data)} sample records from file")
            return data
        else:
            logger.error("File should contain a list of dictionaries")
            return []

    except Exception as e:
        logger.error(f"Failed to load data from file: {e}")
        return []


def create_sample_data() -> List[Dict[str, Any]]:
    """Create sample part data for testing"""
    return [
        {
            "id": 1,
            "part_number": "A115957",
            "name": "VALVE BYPASS 1-1/2\" 110-120V",
            "oem_mfg": "EMERSON",
            "model": "CV3000",
            "notes": "Replacement valve for heating system"
        },
        {
            "id": 2,
            "part_number": "B224681",
            "name": "BEARING ASSEMBLY ROLLER",
            "oem_mfg": "SKF",
            "model": "6204-2Z",
            "notes": "Deep groove ball bearing, sealed"
        },
        {
            "id": 3,
            "part_number": "M332445",
            "name": "MOTOR ELECTRIC 3HP 1800RPM",
            "oem_mfg": "ABB",
            "model": "M3AA132M",
            "notes": "Three-phase induction motor"
        },
        {
            "id": 4,
            "part_number": "S445221",
            "name": "SWITCH LIMIT SPDT 10A",
            "oem_mfg": "HONEYWELL",
            "model": "LSA1A",
            "notes": "Limit switch with roller actuator"
        },
        {
            "id": 5,
            "part_number": "F556332",
            "name": "FILTER AIR PLEATED 16X20X1",
            "oem_mfg": "MANN",
            "model": "C25860",
            "notes": "HVAC air filter, MERV 8"
        }
    ]


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Generate search patterns for part table')
    parser.add_argument('--db-url', help='Database URL for loading sample data')
    parser.add_argument('--sample-file', help='JSON file with sample data')
    parser.add_argument('--output-sql', default='part_search_patterns.sql', help='Output SQL file')
    parser.add_argument('--output-report', default='analysis_report.txt', help='Output analysis report')
    parser.add_argument('--use-sample', action='store_true', help='Use built-in sample data')

    args = parser.parse_args()

    analyzer = PartTableAnalyzer()

    sample_data = []
    if args.db_url:
        sample_data = load_sample_data_from_db(args.db_url)
    elif args.sample_file:
        sample_data = load_sample_data_from_file(args.sample_file)
    elif args.use_sample:
        sample_data = create_sample_data()
        logger.info("Using built-in sample data")
    else:
        logger.info("No data source specified, using defaults")

    analyzer.analyze_sample_data(sample_data)
    analyzer.generate_search_patterns()
    analyzer.generate_search_intents()
    analyzer.generate_keywords()

    sql_output = analyzer.generate_sql_output()

    with open(args.output_sql, 'w') as f:
        f.write(sql_output)
    logger.info(f"SQL output written to {args.output_sql}")

    report = analyzer.generate_analysis_report()
    with open(args.output_report, 'w') as f:
        f.write(report)
    logger.info(f"Analysis report written to {args.output_report}")

    print("\n" + "=" * 60)
    print("PART SEARCH PATTERN GENERATION COMPLETE")
    print("=" * 60)
    print(f"📊 Analyzed {len(analyzer.column_analyses)} columns")
    print(f"🔍 Generated {len(analyzer.generated_patterns)} search patterns")
    print(f"🎯 Created {len(analyzer.generated_intents)} search intents")
    print(f"🔑 Generated {len(analyzer.generated_keywords)} keywords")
    print(f"📝 SQL file: {args.output_sql}")
    print(f"📋 Report file: {args.output_report}")
    print("\nNext steps:")
    print("1. Review the generated SQL file")
    print("2. Execute the SQL statements in your database")
    print("3. Test the search patterns with your application")
    print("4. Adjust patterns based on real-world usage")


if __name__ == "__main__":
    main()