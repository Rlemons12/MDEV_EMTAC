from modules.configuration.config_env import DatabaseConfig
from modules.configuration.log_config import debug_id, info_id, error_id, warning_id, get_request_id, logger
from modules.emtacdb.emtacdb_fts import Part, Image, PartsPositionImageAssociation, Drawing, DrawingPartAssociation
from sqlalchemy import and_, text, or_
import pandas as pd
import json
from datetime import datetime

import psycopg2
from psycopg2.extras import execute_values
import os
import sys
import uuid
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager

# Import logging and database configurations
from modules.configuration.log_config import (
    logger, debug_id, info_id, warning_id, error_id,
    set_request_id, get_request_id, log_timed_operation,
    with_request_id
)
from modules.configuration.config_env import DatabaseConfig


class PostgreSQLDatabaseManager:
    """Enhanced base class for PostgreSQL database management operations with modern patterns."""

    def __init__(self, session=None, request_id=None):
        self.session_provided = session is not None
        self.db_config = DatabaseConfig()
        self.session = session or self.db_config.get_main_session()
        self.request_id = request_id or get_request_id()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.session_provided:
            self.session.close()
            debug_id("Closed PostgreSQL database session", self.request_id)

    @contextmanager
    def transaction(self):
        """Enhanced context manager for database transactions with proper rollback."""
        try:
            yield self.session
            self.session.commit()
            debug_id("PostgreSQL transaction committed successfully", self.request_id)
        except Exception as e:
            self.session.rollback()
            error_id(f"PostgreSQL transaction failed, rolled back: {str(e)}", self.request_id, exc_info=True)
            raise

    @contextmanager
    def savepoint(self):
        """Context manager for PostgreSQL savepoints."""
        savepoint = self.session.begin_nested()
        try:
            yield self.session
            savepoint.commit()
            debug_id("PostgreSQL savepoint committed", self.request_id)
        except Exception as e:
            savepoint.rollback()
            debug_id(f"PostgreSQL savepoint rolled back: {e}", self.request_id)
            raise

    def commit(self):
        """Commit the current transaction."""
        try:
            self.session.commit()
            debug_id("PostgreSQL transaction committed", self.request_id)
        except Exception as e:
            self.session.rollback()
            error_id(f"PostgreSQL transaction failed, rolled back: {str(e)}", self.request_id, exc_info=True)
            raise

    def execute_raw_sql(self, sql, params=None):
        """Execute raw SQL with optional parameters and enhanced error handling."""
        try:
            result = self.session.execute(text(sql), params or {})
            debug_id("PostgreSQL raw SQL executed successfully", self.request_id)
            return result
        except Exception as e:
            error_id(f"Error executing PostgreSQL raw SQL: {str(e)}", self.request_id, exc_info=True)
            raise

    def bulk_insert(self, table_name, data, columns):
        """Enhanced bulk insert using PostgreSQL-specific optimizations."""
        try:
            if not data:
                warning_id("No data provided for bulk insert", self.request_id)
                return

            # Set PostgreSQL-specific optimizations
            with self.savepoint():
                self.session.execute(text("SET work_mem = '256MB'"))
                self.session.execute(text("SET maintenance_work_mem = '512MB'"))

            # Get the raw connection
            connection = self.session.connection().connection
            cursor = connection.cursor()

            # Prepare the SQL
            cols = ', '.join(f'"{col}"' for col in columns)
            sql = f'INSERT INTO "{table_name}" ({cols}) VALUES %s'

            # Use execute_values for efficient bulk insert
            execute_values(cursor, sql, data, page_size=1000)

            info_id(f"Bulk inserted {len(data)} rows into {table_name}", self.request_id)

            # Analyze table after bulk insert for better query planning
            self._analyze_table(table_name)

        except Exception as e:
            error_id(f"Error in PostgreSQL bulk insert: {str(e)}", self.request_id, exc_info=True)
            raise

    def _analyze_table(self, table_name):
        """Analyze table for better query performance."""
        try:
            with self.savepoint():
                self.session.execute(text(f'ANALYZE "{table_name}"'))
            debug_id(f"Analyzed PostgreSQL table {table_name}", self.request_id)
        except Exception as e:
            debug_id(f"Table analysis skipped for {table_name}: {e}", self.request_id)


class PostgreSQLRelationshipManager(PostgreSQLDatabaseManager):
    """Enhanced PostgreSQL relationship manager with concurrent processing and modern patterns."""

    def associate_parts_with_images_by_title(self, part_ids=None, position_id=None, use_concurrent=True,
                                             fuzzy_matching=True):
        """
        Enhanced part-image association with concurrent processing and fuzzy matching.

        Args:
            part_ids: List of part IDs to process (None for all parts)
            position_id: Optional position ID to include in associations
            use_concurrent: Use concurrent processing for large datasets
            fuzzy_matching: Use PostgreSQL fuzzy matching for better results

        Returns:
            Dictionary mapping part IDs to lists of created associations
        """
        info_id("Starting enhanced PostgreSQL part-image association process", self.request_id)
        result = {}

        try:
            with self.transaction():
                # Get parts to process
                if part_ids is None:
                    parts = self.session.query(Part).all()
                else:
                    parts = self.session.query(Part).filter(Part.id.in_(part_ids)).all()

                info_id(f"Processing {len(parts)} parts for image associations", self.request_id)

                # Use concurrent processing for large datasets
                if use_concurrent and len(parts) > 10:
                    result = self._associate_parts_concurrent(parts, position_id, fuzzy_matching)
                else:
                    # Sequential processing
                    for part in parts:
                        associations = self._associate_single_part(part, position_id, fuzzy_matching)
                        result[part.id] = associations

                # Optimize database after bulk operations
                self._optimize_associations()

            return result
        except Exception as e:
            error_id(f"Error in enhanced PostgreSQL part-image association: {str(e)}", self.request_id, exc_info=True)
            raise

    def _associate_parts_concurrent(self, parts, position_id, fuzzy_matching):
        """Concurrent processing for large part datasets."""
        result = {}
        max_workers = min(len(parts), 4)

        info_id(f"Using {max_workers} concurrent workers for PostgreSQL part association", self.request_id)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._associate_part_worker, part, position_id, fuzzy_matching): part
                for part in parts
            }

            for future in as_completed(futures):
                part = futures[future]
                try:
                    associations = future.result()
                    result[part.id] = associations
                    debug_id(f"Completed associations for part {part.part_number}", self.request_id)
                except Exception as e:
                    error_id(f"Error associating part {part.id}: {e}", self.request_id)
                    result[part.id] = []

        return result

    def _associate_part_worker(self, part, position_id, fuzzy_matching):
        """Worker method for concurrent part association."""
        # Create a new manager instance for this worker
        with PostgreSQLDatabaseManager(request_id=self.request_id) as worker_manager:
            # Recreate the part object in the new session
            worker_part = worker_manager.session.query(Part).filter(Part.id == part.id).first()
            if worker_part:
                return self._associate_single_part_enhanced(worker_manager.session, worker_part, position_id,
                                                            fuzzy_matching)
            return []

    def _associate_single_part(self, part, position_id=None, fuzzy_matching=True):
        """Enhanced helper method to associate a single part with matching images."""
        return self._associate_single_part_enhanced(self.session, part, position_id, fuzzy_matching)

    def _associate_single_part_enhanced(self, session, part, position_id=None, fuzzy_matching=True):
        """Enhanced single part association with fuzzy matching."""
        created = []

        try:
            # Build query based on fuzzy matching preference
            if fuzzy_matching:
                try:
                    # Use PostgreSQL similarity for fuzzy matching
                    matching_images = session.query(Image).filter(
                        text("similarity(title, :part_number) > 0.3")
                    ).params(part_number=part.part_number).all()
                except Exception:
                    # Fallback to case-insensitive like matching
                    matching_images = session.query(Image).filter(
                        Image.title.ilike(f"%{part.part_number}%")
                    ).all()
            else:
                # Exact case-insensitive matching
                matching_images = session.query(Image).filter(
                    Image.title.ilike(part.part_number)
                ).all()

            info_id(f"Found {len(matching_images)} images matching part {part.part_number}", self.request_id)

            if not matching_images:
                return created

            # Batch check for existing associations
            existing_associations = set()
            query = session.query(PartsPositionImageAssociation).filter(
                and_(
                    PartsPositionImageAssociation.part_id == part.id,
                    PartsPositionImageAssociation.image_id.in_([img.id for img in matching_images])
                )
            )

            if position_id is not None:
                query = query.filter(PartsPositionImageAssociation.position_id == position_id)

            for assoc in query.all():
                existing_associations.add((assoc.image_id, assoc.position_id))

            # Create new associations
            for image in matching_images:
                key = (image.id, position_id)
                if key not in existing_associations:
                    assoc = PartsPositionImageAssociation(
                        part_id=part.id,
                        image_id=image.id,
                        position_id=position_id
                    )
                    session.add(assoc)
                    created.append(assoc)

            if created:
                session.flush()
                debug_id(f"Created {len(created)} new associations for part {part.part_number}", self.request_id)

        except Exception as e:
            error_id(f"Error in enhanced single part association: {e}", self.request_id)

        return created

    def associate_drawings_with_parts_by_number(self, batch_size=100):
        """
        Enhanced drawing-part association with batch processing and better performance.
        Handles multiple comma-separated part numbers per drawing.

        Args:
            batch_size: Number of drawings to process in each batch

        Returns:
            Dict mapping drawing_id to list of created associations
        """
        info_id("Starting enhanced PostgreSQL drawing-part association process", self.request_id)

        try:
            with self.transaction():
                # Get all drawings with spare part numbers using optimized query
                drawings_query = self.session.query(Drawing).filter(
                    and_(
                        Drawing.drw_spare_part_number.isnot(None),
                        Drawing.drw_spare_part_number != ''
                    )
                )

                total_count = drawings_query.count()
                info_id(f"Found {total_count} drawings with spare part numbers", self.request_id)

                associations_by_drawing = {}
                processed = 0

                # Process in batches for better memory management
                for offset in range(0, total_count, batch_size):
                    batch_drawings = drawings_query.offset(offset).limit(batch_size).all()

                    for drawing in batch_drawings:
                        try:
                            drawing_associations = self._process_single_drawing_enhanced(drawing)
                            if drawing_associations:
                                associations_by_drawing[drawing.id] = drawing_associations

                            processed += 1
                            if processed % 50 == 0:
                                info_id(f"Processed {processed}/{total_count} drawings", self.request_id)

                        except Exception as e:
                            error_id(f"Error processing drawing {drawing.id}: {e}", self.request_id)
                            continue

                # Optimize database after bulk operations
                self._optimize_associations()

                info_id(f"Created new associations for {len(associations_by_drawing)} drawings", self.request_id)
                return associations_by_drawing

        except Exception as e:
            error_id(f"Error in enhanced PostgreSQL drawing-part association: {str(e)}", self.request_id, exc_info=True)
            raise

    def _process_single_drawing_enhanced(self, drawing):
        """Enhanced processing of a single drawing for part associations."""
        if not drawing.drw_spare_part_number or not drawing.drw_spare_part_number.strip():
            return []

        debug_id(f"Processing drawing {drawing.drw_number} with spare part number(s): {drawing.drw_spare_part_number}",
                 self.request_id)

        # Split and clean part numbers
        part_numbers = [pn.strip() for pn in drawing.drw_spare_part_number.split(',') if pn.strip()]
        drawing_associations = []

        if not part_numbers:
            return drawing_associations

        # Get all matching parts in optimized queries
        all_matching_parts = []
        for part_number in part_numbers:
            try:
                # Use case-insensitive matching with wildcards
                matching_parts = self.session.query(Part).filter(
                    Part.part_number.ilike(f"%{part_number}%")
                ).all()
                all_matching_parts.extend(matching_parts)
                debug_id(f"Found {len(matching_parts)} parts matching '{part_number}'", self.request_id)
            except Exception as e:
                debug_id(f"Error searching for part number '{part_number}': {e}", self.request_id)
                continue

        if not all_matching_parts:
            return drawing_associations

        # Get existing associations to avoid duplicates
        existing_part_ids = set()
        if all_matching_parts:
            existing_assocs = self.session.query(DrawingPartAssociation).filter(
                and_(
                    DrawingPartAssociation.drawing_id == drawing.id,
                    DrawingPartAssociation.part_id.in_([p.id for p in all_matching_parts])
                )
            ).all()

            existing_part_ids = {assoc.part_id for assoc in existing_assocs}

        # Create new associations
        for part in all_matching_parts:
            if part.id not in existing_part_ids:
                association = DrawingPartAssociation(
                    drawing_id=drawing.id,
                    part_id=part.id
                )
                self.session.add(association)
                drawing_associations.append(association)
                debug_id(f"Created association between drawing {drawing.drw_number} and part {part.part_number}",
                         self.request_id)

        if drawing_associations:
            self.session.flush()

        return drawing_associations

    def bulk_associate_parts_images(self, associations_data, batch_size=1000):
        """
        Enhanced bulk create part-image associations with better performance.

        Args:
            associations_data: List of dicts with keys: part_id, image_id, position_id
            batch_size: Number of associations to process in each batch
        """
        try:
            if not associations_data:
                warning_id("No association data provided for bulk operation", self.request_id)
                return

            info_id(f"Processing {len(associations_data)} associations in batches of {batch_size}", self.request_id)

            with self.transaction():
                # Process in batches to avoid memory issues
                for i in range(0, len(associations_data), batch_size):
                    batch = associations_data[i:i + batch_size]
                    self._process_association_batch(batch)

                    info_id(
                        f"Processed batch {i // batch_size + 1}/{(len(associations_data) + batch_size - 1) // batch_size}",
                        self.request_id)

                # Optimize after bulk operations
                self._optimize_associations()

        except Exception as e:
            error_id(f"Error in enhanced bulk part-image association: {str(e)}", self.request_id, exc_info=True)
            raise

    def _process_association_batch(self, batch):
        """Process a batch of associations with optimized duplicate checking."""
        if not batch:
            return

        # Build efficient query to check existing associations
        conditions = []
        for assoc_data in batch:
            condition = and_(
                PartsPositionImageAssociation.part_id == assoc_data['part_id'],
                PartsPositionImageAssociation.image_id == assoc_data['image_id'],
                PartsPositionImageAssociation.position_id == assoc_data.get('position_id')
            )
            conditions.append(condition)

        # Get existing associations in one query
        existing_keys = set()
        if conditions:
            existing_assocs = self.session.query(PartsPositionImageAssociation).filter(
                or_(*conditions)
            ).all()

            for assoc in existing_assocs:
                existing_keys.add((assoc.part_id, assoc.image_id, assoc.position_id))

        # Create only new associations
        new_associations = []
        for assoc_data in batch:
            key = (assoc_data['part_id'], assoc_data['image_id'], assoc_data.get('position_id'))
            if key not in existing_keys:
                new_associations.append(PartsPositionImageAssociation(**assoc_data))

        if new_associations:
            self.session.add_all(new_associations)
            self.session.flush()
            debug_id(f"Created {len(new_associations)} new associations in batch", self.request_id)

    def _optimize_associations(self):
        """Optimize association tables after bulk operations."""
        try:
            self._analyze_table('parts_position_image_association')
            self._analyze_table('drawing_part_association')
        except Exception as e:
            debug_id(f"Association optimization skipped: {e}", self.request_id)


class PostgreSQLDuplicateManager(PostgreSQLDatabaseManager):
    """Enhanced duplicate detection and management with modern patterns and fuzzy matching."""

    def find_duplicate_parts(self, threshold=0.9, use_fuzzy_matching=True, batch_size=500):
        """
        Enhanced duplicate detection with configurable strategies and batch processing.
        Uses PostgreSQL's text search capabilities.

        Args:
            threshold: Similarity threshold (0.0-1.0)
            use_fuzzy_matching: Whether to use PostgreSQL's fuzzy string matching
            batch_size: Limit results to manage memory

        Returns:
            List of dictionaries containing potential duplicate part information
        """
        info_id(
            f"Finding duplicate parts with threshold {threshold} using {'fuzzy' if use_fuzzy_matching else 'exact'} matching",
            self.request_id)

        try:
            if use_fuzzy_matching:
                return self._find_duplicates_fuzzy(threshold, batch_size)
            else:
                return self._find_duplicates_exact(batch_size)

        except Exception as e:
            error_id(f"Error finding duplicate parts: {str(e)}", self.request_id, exc_info=True)
            raise

    def _find_duplicates_fuzzy(self, threshold, batch_size):
        """Use PostgreSQL's similarity function for fuzzy duplicate detection."""
        try:
            # Enable pg_trgm extension if not already enabled
            with self.savepoint():
                self.session.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))

            sql = text("""
                SELECT p1.id as id1, p1.part_number as part1, 
                       p2.id as id2, p2.part_number as part2,
                       similarity(p1.part_number, p2.part_number) as sim_score
                FROM part p1
                JOIN part p2 ON p1.id < p2.id
                WHERE similarity(p1.part_number, p2.part_number) > :threshold
                ORDER BY sim_score DESC
                LIMIT :batch_size
            """)

            result = self.execute_raw_sql(sql, {'threshold': threshold, 'batch_size': batch_size})
            duplicates = [
                {
                    'id1': row[0], 'part1': row[1],
                    'id2': row[2], 'part2': row[3],
                    'similarity': float(row[4]),
                    'match_type': 'fuzzy'
                }
                for row in result.fetchall()
            ]

            info_id(f"Found {len(duplicates)} potential fuzzy duplicate pairs", self.request_id)
            return duplicates

        except Exception as e:
            error_id(f"PostgreSQL fuzzy matching failed, falling back to exact matching: {e}", self.request_id)
            return self._find_duplicates_exact(batch_size)

    def _find_duplicates_exact(self, batch_size):
        """Exact matching approach with enhanced text processing."""
        try:
            sql = text("""
                SELECT p1.id as id1, p1.part_number as part1,
                       p2.id as id2, p2.part_number as part2,
                       1.0 as sim_score
                FROM part p1
                JOIN part p2 ON p1.id < p2.id
                WHERE LOWER(TRIM(p1.part_number)) = LOWER(TRIM(p2.part_number))
                   OR REPLACE(LOWER(TRIM(p1.part_number)), '-', '') = REPLACE(LOWER(TRIM(p2.part_number)), '-', '')
                   OR REPLACE(REPLACE(LOWER(TRIM(p1.part_number)), '-', ''), ' ', '') = 
                      REPLACE(REPLACE(LOWER(TRIM(p2.part_number)), '-', ''), ' ', '')
                ORDER BY p1.part_number
                LIMIT :batch_size
            """)

            result = self.execute_raw_sql(sql, {'batch_size': batch_size})
            duplicates = [
                {
                    'id1': row[0], 'part1': row[1],
                    'id2': row[2], 'part2': row[3],
                    'similarity': float(row[4]),
                    'match_type': 'exact'
                }
                for row in result.fetchall()
            ]

            info_id(f"Found {len(duplicates)} exact duplicate pairs", self.request_id)
            return duplicates

        except Exception as e:
            error_id(f"Exact duplicate detection failed: {e}", self.request_id)
            return []

    def merge_duplicate_parts(self, source_id, target_id, fields_to_merge=None, dry_run=False):
        """
        Enhanced part merging with transaction safety and dry-run capability.
        Merges two duplicate parts using PostgreSQL transactions.

        Args:
            source_id: ID of the source part (will be merged into target)
            target_id: ID of the target part (will be kept)
            fields_to_merge: List of fields to merge (None for all non-null fields)
            dry_run: If True, only report what would be done without making changes

        Returns:
            Dictionary with merge results and statistics
        """
        info_id(f"{'[DRY RUN] ' if dry_run else ''}Merging part {source_id} into part {target_id}", self.request_id)

        try:
            if dry_run:
                return self._merge_parts_dry_run(source_id, target_id, fields_to_merge)
            else:
                return self._merge_parts_execute(source_id, target_id, fields_to_merge)

        except Exception as e:
            error_id(f"Error merging parts: {str(e)}", self.request_id, exc_info=True)
            raise

    def _merge_parts_dry_run(self, source_id, target_id, fields_to_merge):
        """Dry run mode - analyze what would be merged without making changes."""
        source_part = self.session.query(Part).filter(Part.id == source_id).first()
        target_part = self.session.query(Part).filter(Part.id == target_id).first()

        if not source_part:
            raise ValueError(f"Source part {source_id} not found")
        if not target_part:
            raise ValueError(f"Target part {target_id} not found")

        # Count associations
        associations_count = self._count_part_associations(source_id)

        # Analyze field merges
        merged_fields = self._analyze_field_merges(source_part, target_part, fields_to_merge)

        merge_stats = {
            'source_part': source_part.part_number,
            'target_part': target_part.part_number,
            'associations_to_update': associations_count,
            'fields_to_merge': merged_fields,
            'dry_run': True,
            'would_succeed': True
        }

        info_id(f"[DRY RUN] Merge analysis completed: {merge_stats}", self.request_id)
        return merge_stats

    def _merge_parts_execute(self, source_id, target_id, fields_to_merge):
        """Execute the actual merge operation."""
        with self.transaction():
            source_part = self.session.query(Part).filter(Part.id == source_id).first()
            target_part = self.session.query(Part).filter(Part.id == target_id).first()

            if not source_part:
                raise ValueError(f"Source part {source_id} not found")
            if not target_part:
                raise ValueError(f"Target part {target_id} not found")

            merge_stats = {
                'source_part': source_part.part_number,
                'target_part': target_part.part_number,
                'associations_updated': 0,
                'fields_merged': [],
                'dry_run': False
            }

            # Update related associations to point to target part
            merge_stats['associations_updated'] = self._update_part_associations(source_id, target_id)

            # Merge specified fields or all non-null fields
            merged_fields = self._merge_part_fields(source_part, target_part, fields_to_merge)
            merge_stats['fields_merged'] = merged_fields

            # Delete the source part
            self.session.delete(source_part)
            self.session.flush()

            info_id(f"Successfully merged part {source_id} into {target_id}: {merge_stats}", self.request_id)
            return merge_stats

    def _update_part_associations(self, source_id, target_id):
        """Enhanced association updates with better conflict handling."""
        total_updated = 0

        try:
            # Update PartsPositionImageAssociation with conflict resolution
            result1 = self.execute_raw_sql("""
                UPDATE parts_position_image_association 
                SET part_id = :target_id 
                WHERE part_id = :source_id
                AND NOT EXISTS (
                    SELECT 1 FROM parts_position_image_association ppia2
                    WHERE ppia2.part_id = :target_id 
                    AND ppia2.image_id = parts_position_image_association.image_id
                    AND COALESCE(ppia2.position_id, -1) = COALESCE(parts_position_image_association.position_id, -1)
                )
            """, {'source_id': source_id, 'target_id': target_id})

            updated1 = result1.rowcount if hasattr(result1, 'rowcount') else 0

            # Update DrawingPartAssociation with conflict resolution
            result2 = self.execute_raw_sql("""
                UPDATE drawing_part_association 
                SET part_id = :target_id 
                WHERE part_id = :source_id
                AND NOT EXISTS (
                    SELECT 1 FROM drawing_part_association dpa2
                    WHERE dpa2.part_id = :target_id 
                    AND dpa2.drawing_id = drawing_part_association.drawing_id
                )
            """, {'source_id': source_id, 'target_id': target_id})

            updated2 = result2.rowcount if hasattr(result2, 'rowcount') else 0

            # Delete remaining duplicate associations
            self.execute_raw_sql("""
                DELETE FROM parts_position_image_association 
                WHERE part_id = :source_id
            """, {'source_id': source_id})

            self.execute_raw_sql("""
                DELETE FROM drawing_part_association 
                WHERE part_id = :source_id
            """, {'source_id': source_id})

            total_updated = updated1 + updated2
            debug_id(f"Updated {total_updated} associations", self.request_id)

        except Exception as e:
            error_id(f"Error updating part associations: {str(e)}", self.request_id, exc_info=True)
            raise

        return total_updated

    def _count_part_associations(self, part_id):
        """Count associations for dry-run analysis."""
        try:
            count1 = self.session.query(PartsPositionImageAssociation).filter(
                PartsPositionImageAssociation.part_id == part_id
            ).count()

            count2 = self.session.query(DrawingPartAssociation).filter(
                DrawingPartAssociation.part_id == part_id
            ).count()

            return count1 + count2
        except Exception as e:
            warning_id(f"Could not count associations: {e}", self.request_id)
            return 0

    def _analyze_field_merges(self, source_part, target_part, fields_to_merge):
        """Analyze which fields would be merged in dry-run mode."""
        field_analysis = []

        try:
            part_columns = [column.name for column in source_part.__table__.columns]

            if fields_to_merge:
                fields_to_process = [f for f in fields_to_merge if f in part_columns and f != 'id']
            else:
                fields_to_process = [f for f in part_columns if f != 'id']

            for field_name in fields_to_process:
                source_value = getattr(source_part, field_name, None)
                target_value = getattr(target_part, field_name, None)

                if source_value and not target_value:
                    field_analysis.append({
                        'field': field_name,
                        'current_target_value': target_value,
                        'new_value_from_source': source_value,
                        'action': 'merge'
                    })
                elif source_value and target_value and source_value != target_value:
                    field_analysis.append({
                        'field': field_name,
                        'current_target_value': target_value,
                        'source_value': source_value,
                        'action': 'conflict - keeping target'
                    })

        except Exception as e:
            warning_id(f"Error analyzing field merges: {e}", self.request_id)

        return field_analysis

    def _merge_part_fields(self, source_part, target_part, fields_to_merge):
        """Execute field merging with conflict resolution."""
        merged_fields = []

        try:
            part_columns = [column.name for column in source_part.__table__.columns]

            if fields_to_merge:
                fields_to_process = [f for f in fields_to_merge if f in part_columns and f != 'id']
            else:
                fields_to_process = [f for f in part_columns if f != 'id']

            for field_name in fields_to_process:
                source_value = getattr(source_part, field_name, None)
                target_value = getattr(target_part, field_name, None)

                # Merge non-null source values into null target fields
                if source_value and not target_value:
                    setattr(target_part, field_name, source_value)
                    merged_fields.append({
                        'field': field_name,
                        'old_value': target_value,
                        'new_value': source_value
                    })

            if merged_fields:
                self.session.flush()

        except Exception as e:
            warning_id(f"Error merging fields: {e}", self.request_id)

        return merged_fields


class EnhancedExcelToPostgreSQLMapper:
    """Enhanced Excel to PostgreSQL mapper with modern patterns and better error handling."""

    def __init__(self, excel_path, db_config=None):
        """
        Initialize the mapper with an Excel file path and PostgreSQL database configuration.

        Args:
            excel_path: Path to the Excel file
            db_config: DatabaseConfig instance. If None, a new one will be created.
        """
        self.request_id = set_request_id()
        info_id("Initializing EnhancedExcelToPostgreSQLMapper", self.request_id)

        self.excel_path = excel_path
        self.db_config = db_config if db_config else DatabaseConfig()

        if not os.path.exists(excel_path):
            raise FileNotFoundError(f"Excel file not found: {excel_path}")

        debug_id(f"Mapper initialized with Excel file: {excel_path}", self.request_id)

    def infer_postgresql_type(self, pandas_dtype):
        """Enhanced PostgreSQL type inference from pandas dtype."""
        if pd.api.types.is_integer_dtype(pandas_dtype):
            return 'INTEGER'
        elif pd.api.types.is_float_dtype(pandas_dtype):
            return 'NUMERIC'
        elif pd.api.types.is_bool_dtype(pandas_dtype):
            return 'BOOLEAN'
        elif pd.api.types.is_datetime64_any_dtype(pandas_dtype):
            return 'TIMESTAMP'
        else:
            return 'TEXT'

    @with_request_id
    def prompt_for_mapping(self, df):
        """
        Enhanced prompt for column mapping with better validation.
        Decorated with request ID tracking.
        """
        mapping = {}
        type_overrides = {}

        info_id("Excel columns found:", self.request_id)
        for i, col in enumerate(df.columns):
            dtype = self.infer_postgresql_type(df[col].dtype)
            sample_data = df[col].dropna().head(3).tolist() if not df[col].empty else []
            debug_id(f"  {i + 1}. '{col}' (type: {dtype}, samples: {sample_data})", self.request_id)

        info_id("For each Excel column, specify the PostgreSQL column name to map to (leave blank to skip):",
                self.request_id)

        for col in df.columns:
            dtype = self.infer_postgresql_type(df[col].dtype)
            mapped_col = input(f"Map Excel column '{col}' to PostgreSQL column (or blank to skip): ").strip()
            if mapped_col:
                type_choice = input(
                    f" - Data type for '{mapped_col}'? [INTEGER/NUMERIC/TEXT/BOOLEAN/TIMESTAMP, default: {dtype}]: "
                ).strip().upper()
                valid_types = ['INTEGER', 'NUMERIC', 'TEXT', 'BOOLEAN', 'TIMESTAMP']
                type_overrides[mapped_col] = type_choice if type_choice in valid_types else dtype
                mapping[col] = mapped_col
                debug_id(f"Mapped '{col}' to '{mapped_col}' with type {type_overrides[mapped_col]}", self.request_id)

        return mapping, type_overrides

    def create_mapping_table(self, session):
        """Enhanced mapping table creation with better error handling."""
        with log_timed_operation("create_mapping_table", self.request_id):
            try:
                session.execute(text("""
                CREATE TABLE IF NOT EXISTS excel_postgresql_mapping (
                    id SERIAL PRIMARY KEY,
                    mapping_name TEXT UNIQUE,
                    excel_file TEXT NOT NULL,
                    excel_sheet TEXT NOT NULL,
                    postgresql_table TEXT NOT NULL,
                    column_mapping JSONB NOT NULL,
                    column_types JSONB NOT NULL,
                    row_count INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """))

                # Create indexes for better performance
                session.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_excel_postgresql_mapping_name 
                ON excel_postgresql_mapping(mapping_name)
                """))

                session.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_excel_postgresql_mapping_table 
                ON excel_postgresql_mapping(postgresql_table)
                """))

                session.commit()
                debug_id("Enhanced PostgreSQL mapping table created or already exists", self.request_id)
            except Exception as e:
                session.rollback()
                error_id(f"Error creating PostgreSQL mapping table: {str(e)}", self.request_id)
                raise

    def store_mapping(self, session, mapping_name, excel_file, excel_sheet, postgresql_table, mapping, type_overrides,
                      row_count=0):
        """Enhanced mapping storage with upsert capability."""
        with log_timed_operation("store_mapping", self.request_id):
            try:
                # Use upsert to handle duplicate mapping names
                sql = text("""
                INSERT INTO excel_postgresql_mapping 
                (mapping_name, excel_file, excel_sheet, postgresql_table, column_mapping, column_types, row_count, created_at, updated_at)
                VALUES (:name, :file, :sheet, :table, :mapping, :types, :row_count, :created_at, :updated_at)
                ON CONFLICT (mapping_name) DO UPDATE SET
                    excel_file = EXCLUDED.excel_file,
                    excel_sheet = EXCLUDED.excel_sheet,
                    postgresql_table = EXCLUDED.postgresql_table,
                    column_mapping = EXCLUDED.column_mapping,
                    column_types = EXCLUDED.column_types,
                    row_count = EXCLUDED.row_count,
                    updated_at = EXCLUDED.updated_at
                """)

                session.execute(sql, {
                    'name': mapping_name,
                    'file': excel_file,
                    'sheet': excel_sheet,
                    'table': postgresql_table,
                    'mapping': json.dumps(mapping),
                    'types': json.dumps(type_overrides),
                    'row_count': row_count,
                    'created_at': datetime.now(),
                    'updated_at': datetime.now()
                })
                session.commit()
                info_id(f"PostgreSQL mapping information stored/updated for '{mapping_name}'", self.request_id)
            except Exception as e:
                session.rollback()
                error_id(f"Error storing PostgreSQL mapping information: {str(e)}", self.request_id)
                raise

    def create_table(self, session, table_name, mapping, type_overrides):
        """Enhanced table creation with better column handling."""
        with log_timed_operation(f"create_table_{table_name}", self.request_id):
            try:
                columns = []
                for excel_col, postgresql_col in mapping.items():
                    col_type = type_overrides[postgresql_col]
                    columns.append(f'"{postgresql_col}" {col_type}')

                col_defs = ", ".join(columns)
                sql = f'''
                CREATE TABLE IF NOT EXISTS "{table_name}" (
                    id SERIAL PRIMARY KEY, 
                    {col_defs},
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                '''

                session.execute(text(sql))

                # Create basic indexes
                for postgresql_col in mapping.values():
                    try:
                        index_sql = f'CREATE INDEX IF NOT EXISTS idx_{table_name}_{postgresql_col} ON "{table_name}" ("{postgresql_col}")'
                        session.execute(text(index_sql))
                    except:
                        pass  # Skip if index creation fails

                session.commit()
                info_id(f"Created enhanced PostgreSQL table '{table_name}' with {len(columns)} columns",
                        self.request_id)
            except Exception as e:
                session.rollback()
                error_id(f"Error creating PostgreSQL table '{table_name}': {str(e)}", self.request_id)
                raise

    def insert_data(self, session, table_name, df, mapping, batch_size=1000):
        """Enhanced data insertion with batch processing and better error handling."""
        with log_timed_operation(f"insert_data_{table_name}", self.request_id):
            try:
                mapped_cols = list(mapping.keys())
                postgresql_cols = [mapping[col] for col in mapped_cols]

                total_rows = len(df)
                info_id(f"Inserting {total_rows} rows into PostgreSQL table '{table_name}' in batches of {batch_size}",
                        self.request_id)

                # Process data in batches
                inserted_count = 0
                for start_idx in range(0, total_rows, batch_size):
                    end_idx = min(start_idx + batch_size, total_rows)
                    batch_df = df[mapped_cols].iloc[start_idx:end_idx]

                    # Prepare data for bulk insert
                    data_tuples = []
                    for _, row in batch_df.iterrows():
                        cleaned_row = []
                        for val in row:
                            if pd.isna(val):
                                cleaned_row.append(None)
                            elif isinstance(val, (int, float, str, bool)):
                                cleaned_row.append(val)
                            else:
                                cleaned_row.append(str(val))
                        data_tuples.append(tuple(cleaned_row))

                    # Use PostgreSQL-specific bulk insert
                    connection = session.connection().connection
                    cursor = connection.cursor()

                    insert_cols = ', '.join(f'"{col}"' for col in postgresql_cols)
                    sql = f'INSERT INTO "{table_name}" ({insert_cols}) VALUES %s'

                    execute_values(cursor, sql, data_tuples, page_size=500)

                    inserted_count += len(data_tuples)
                    if inserted_count % 5000 == 0:
                        info_id(f"Inserted {inserted_count}/{total_rows} rows", self.request_id)

                connection.commit()

                # Analyze table after bulk insert
                session.execute(text(f'ANALYZE "{table_name}"'))
                session.commit()

                info_id(f"Successfully inserted {inserted_count} rows into PostgreSQL table '{table_name}'",
                        self.request_id)
                return inserted_count

            except Exception as e:
                session.rollback()
                error_id(f"Error inserting data into PostgreSQL table '{table_name}': {str(e)}", self.request_id)
                raise

    @with_request_id
    def run(self, sheet_name=None, table_name=None, mapping_name=None, batch_size=1000):
        """Enhanced main execution method with better parameter handling."""
        try:
            # Read Excel file
            with log_timed_operation("read_excel", self.request_id):
                df = self._read_excel_with_validation(sheet_name)

            # Get table and mapping names
            table_name = table_name or self._get_table_name(sheet_name)
            mapping_name = mapping_name or self._get_mapping_name(sheet_name, table_name)

            # Column mapping
            mapping, type_overrides = self.prompt_for_mapping(df)

            if not mapping:
                warning_id("No columns mapped! Exiting.", self.request_id)
                return False

            # Database operations
            info_id("Establishing PostgreSQL database connection", self.request_id)
            session = self.db_config.get_main_session()

            try:
                with log_timed_operation("database_operations", self.request_id):
                    # Create mapping table if needed
                    self.create_mapping_table(session)

                    # Create data table
                    self.create_table(session, table_name, mapping, type_overrides)

                    # Insert data
                    row_count = self.insert_data(session, table_name, df, mapping, batch_size)

                    # Store mapping information with row count
                    self.store_mapping(session, mapping_name, self.excel_path, sheet_name,
                                       table_name, mapping, type_overrides, row_count)

                info_id(f"All PostgreSQL operations completed successfully for mapping '{mapping_name}'",
                        self.request_id)
                return True

            finally:
                debug_id("Closing PostgreSQL database session", self.request_id)
                session.close()

        except Exception as e:
            error_id(f"Error in enhanced Excel to PostgreSQL mapping process: {str(e)}", self.request_id)
            raise

    def _read_excel_with_validation(self, sheet_name):
        """Read Excel file with enhanced validation."""
        try:
            if sheet_name:
                info_id(f"Reading Excel sheet: {sheet_name}", self.request_id)
                df = pd.read_excel(self.excel_path, sheet_name=sheet_name)
            else:
                xls = pd.ExcelFile(self.excel_path)
                info_id(f"Sheets found: {xls.sheet_names}", self.request_id)
                sheet_name = input("Enter sheet name to import: ").strip()
                if sheet_name not in xls.sheet_names:
                    raise ValueError(f"Sheet '{sheet_name}' not found in Excel file")
                df = pd.read_excel(self.excel_path, sheet_name=sheet_name)

            if df.empty:
                raise ValueError(f"Sheet '{sheet_name}' is empty")

            info_id(f"Read {len(df)} rows and {len(df.columns)} columns from sheet '{sheet_name}'", self.request_id)
            return df

        except Exception as e:
            error_id(f"Failed to read Excel file: {e}", self.request_id)
            raise

    def _get_table_name(self, sheet_name):
        """Get table name with validation."""
        default_table = re.sub(r'[^a-zA-Z0-9_]', '_', sheet_name.lower()) if sheet_name else "imported_data"
        table_name = input(f"PostgreSQL table name? (default: {default_table}): ").strip() or default_table
        info_id(f"Using table name: {table_name}", self.request_id)
        return table_name

    def _get_mapping_name(self, sheet_name, table_name):
        """Get mapping name with validation."""
        default_mapping = f"{sheet_name}_to_{table_name}" if sheet_name else f"mapping_to_{table_name}"
        mapping_name = input("Name this mapping (for future use): ").strip() or default_mapping
        info_id(f"Using mapping name: {mapping_name}", self.request_id)
        return mapping_name


# Utility functions for PostgreSQL-specific operations
class PostgreSQLUtilities:
    """Enhanced utility functions specific to PostgreSQL operations."""

    @staticmethod
    def enable_extensions(session, extensions=['pg_trgm', 'unaccent', 'uuid-ossp']):
        """Enable commonly used PostgreSQL extensions with better error handling."""
        try:
            enabled = []
            for ext in extensions:
                try:
                    session.execute(text(f'CREATE EXTENSION IF NOT EXISTS "{ext}"'))
                    enabled.append(ext)
                except Exception as e:
                    debug_id(f"Could not enable extension '{ext}': {e}", get_request_id())
                    continue

            if enabled:
                session.commit()
                info_id(f"Enabled PostgreSQL extensions: {enabled}", get_request_id())
            return enabled
        except Exception as e:
            session.rollback()
            error_id(f"Error enabling PostgreSQL extensions: {str(e)}", get_request_id())
            raise

    @staticmethod
    def create_indexes(session, table_name, columns, index_types=None):
        """Enhanced index creation with different index types."""
        try:
            created_indexes = []
            index_types = index_types or {}

            for column in columns:
                index_name = f"idx_{table_name}_{column}"
                index_type = index_types.get(column, 'btree')

                try:
                    if index_type == 'gin':
                        sql = f'CREATE INDEX IF NOT EXISTS {index_name} ON "{table_name}" USING gin ("{column}")'
                    elif index_type == 'gist':
                        sql = f'CREATE INDEX IF NOT EXISTS {index_name} ON "{table_name}" USING gist ("{column}")'
                    else:
                        sql = f'CREATE INDEX IF NOT EXISTS {index_name} ON "{table_name}" ("{column}")'

                    session.execute(text(sql))
                    created_indexes.append(f"{index_name} ({index_type})")
                except Exception as e:
                    debug_id(f"Could not create index on {column}: {e}", get_request_id())
                    continue

            if created_indexes:
                session.commit()
                info_id(f"Created indexes for table {table_name}: {created_indexes}", get_request_id())
            return created_indexes
        except Exception as e:
            session.rollback()
            error_id(f"Error creating indexes: {str(e)}", get_request_id())
            raise

    @staticmethod
    def analyze_table(session, table_name):
        """Enhanced table analysis with statistics reporting."""
        try:
            session.execute(text(f'ANALYZE "{table_name}"'))

            # Get basic statistics
            stats_query = text("""
                SELECT 
                    schemaname, tablename, n_tup_ins, n_tup_upd, n_tup_del, 
                    n_live_tup, n_dead_tup, last_analyze
                FROM pg_stat_user_tables 
                WHERE tablename = :table_name
            """)

            result = session.execute(stats_query, {'table_name': table_name}).fetchone()

            session.commit()
            info_id(f"Analyzed table {table_name} for query optimization", get_request_id())

            if result:
                debug_id(f"Table stats - Live tuples: {result[5]}, Dead tuples: {result[6]}", get_request_id())

            return result
        except Exception as e:
            error_id(f"Error analyzing table {table_name}: {str(e)}", get_request_id())
            raise

    @staticmethod
    def get_table_info(session, table_name):
        """Get comprehensive table information."""
        try:
            info_query = text("""
                SELECT 
                    column_name, data_type, is_nullable, column_default,
                    character_maximum_length, numeric_precision, numeric_scale
                FROM information_schema.columns 
                WHERE table_name = :table_name
                ORDER BY ordinal_position
            """)

            columns = session.execute(info_query, {'table_name': table_name}).fetchall()

            size_query = text("""
                SELECT pg_size_pretty(pg_total_relation_size(:table_name)) as table_size,
                       pg_size_pretty(pg_relation_size(:table_name)) as data_size
            """)

            size_info = session.execute(size_query, {'table_name': table_name}).fetchone()

            return {
                'columns': [dict(zip(['name', 'type', 'nullable', 'default', 'max_length', 'precision', 'scale'], col))
                            for col in columns],
                'total_size': size_info[0] if size_info else 'Unknown',
                'data_size': size_info[1] if size_info else 'Unknown'
            }
        except Exception as e:
            error_id(f"Error getting table info for {table_name}: {str(e)}", get_request_id())
            return None


# ==========================================
# BACKWARD COMPATIBILITY CLASSES
# All use PostgreSQL underneath but keep same names for existing scripts
# ==========================================

class DatabaseManager(PostgreSQLDatabaseManager):
    """
    PostgreSQL-backed database manager - maintains compatibility with existing scripts.
    All operations now use PostgreSQL instead of SQLite.
    """
    pass


class RelationshipManager(PostgreSQLRelationshipManager):
    """
    PostgreSQL-backed relationship manager - maintains compatibility with existing scripts.
    All operations now use PostgreSQL instead of SQLite.
    """

    def associate_parts_with_images_by_title(self, part_ids=None, position_id=None):
        """
        Backward compatible method signature - now uses PostgreSQL with enhanced features.
        """
        return super().associate_parts_with_images_by_title(
            part_ids=part_ids,
            position_id=position_id,
            use_concurrent=True,
            fuzzy_matching=True
        )

    def associate_drawings_with_parts_by_number(self):
        """
        Backward compatible method signature - now uses PostgreSQL with enhanced features.
        """
        return super().associate_drawings_with_parts_by_number(batch_size=100)


class DuplicateManager(PostgreSQLDuplicateManager):
    """
    PostgreSQL-backed duplicate manager - maintains compatibility with existing scripts.
    All operations now use PostgreSQL instead of SQLite.
    """

    def find_duplicate_parts(self, threshold=0.9):
        """
        Backward compatible method signature - now uses PostgreSQL with enhanced features.
        """
        return super().find_duplicate_parts(
            threshold=threshold,
            use_fuzzy_matching=True,
            batch_size=500
        )

    def merge_duplicate_parts(self, source_id, target_id, fields_to_merge=None):
        """
        Backward compatible method signature - now uses PostgreSQL with enhanced features.
        """
        return super().merge_duplicate_parts(
            source_id=source_id,
            target_id=target_id,
            fields_to_merge=fields_to_merge,
            dry_run=False
        )


class EnhancedExcelToSQLiteMapper(EnhancedExcelToPostgreSQLMapper):
    """
    PostgreSQL-backed Excel mapper - maintains compatibility with existing scripts.
    Despite the name, all operations now use PostgreSQL instead of SQLite.
    """

    def __init__(self, excel_path, db_config=None):
        """Initialize with PostgreSQL backend despite SQLite name."""
        super().__init__(excel_path, db_config)
        # Update request ID to reflect PostgreSQL usage
        info_id("Note: EnhancedExcelToSQLiteMapper now uses PostgreSQL backend", self.request_id)

    def infer_sqlite_type(self, pandas_dtype):
        """Backward compatibility method - now returns PostgreSQL types."""
        return self.infer_postgresql_type(pandas_dtype)

    def create_mapping_table(self, session):
        """Creates PostgreSQL mapping table with backward compatible interface."""
        with log_timed_operation("create_mapping_table", self.request_id):
            try:
                # Create PostgreSQL table with SQLite-compatible column names for backward compatibility
                session.execute(text("""
                CREATE TABLE IF NOT EXISTS excel_sqlite_mapping (
                    id SERIAL PRIMARY KEY,
                    mapping_name TEXT UNIQUE,
                    excel_file TEXT NOT NULL,
                    excel_sheet TEXT NOT NULL,
                    sqlite_table TEXT NOT NULL,
                    column_mapping TEXT NOT NULL,
                    column_types TEXT NOT NULL,
                    row_count INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """))

                session.commit()
                debug_id("PostgreSQL mapping table created (SQLite-compatible schema)", self.request_id)
            except Exception as e:
                session.rollback()
                error_id(f"Error creating mapping table: {str(e)}", self.request_id)
                raise

    def store_mapping(self, session, mapping_name, excel_file, excel_sheet, sqlite_table, mapping, type_overrides,
                      row_count=0):
        """Store mapping with backward compatible parameter names."""
        with log_timed_operation("store_mapping", self.request_id):
            try:
                sql = text("""
                INSERT INTO excel_sqlite_mapping 
                (mapping_name, excel_file, excel_sheet, sqlite_table, column_mapping, column_types, row_count, created_at)
                VALUES (:name, :file, :sheet, :table, :mapping, :types, :row_count, :created_at)
                ON CONFLICT (mapping_name) DO UPDATE SET
                    excel_file = EXCLUDED.excel_file,
                    excel_sheet = EXCLUDED.excel_sheet,
                    sqlite_table = EXCLUDED.sqlite_table,
                    column_mapping = EXCLUDED.column_mapping,
                    column_types = EXCLUDED.column_types,
                    row_count = EXCLUDED.row_count,
                    created_at = EXCLUDED.created_at
                """)

                session.execute(sql, {
                    'name': mapping_name,
                    'file': excel_file,
                    'sheet': excel_sheet,
                    'table': sqlite_table,
                    'mapping': json.dumps(mapping),
                    'types': json.dumps(type_overrides),
                    'row_count': row_count,
                    'created_at': datetime.now()
                })
                session.commit()
                info_id(f"Mapping information stored for '{mapping_name}' (PostgreSQL backend)", self.request_id)
            except Exception as e:
                session.rollback()
                error_id(f"Error storing mapping information: {str(e)}", self.request_id)
                raise