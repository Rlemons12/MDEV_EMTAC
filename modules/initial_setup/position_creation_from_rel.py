import sys
import os
import argparse
from datetime import datetime
from collections import defaultdict

# --- IMPORT YOUR APP CONTEXT ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- IMPORT MODELS, LOGGER, CONFIG ---
from modules.emtacdb.emtacdb_fts import (
    Area, EquipmentGroup, Model, AssetNumber, Location, Subassembly,
    ComponentAssembly, AssemblyView, Position
)
from modules.configuration.config_env import DatabaseConfig
from modules.configuration.log_config import logger, set_request_id, info_id, debug_id, error_id, warning_id

# For request tracking
set_request_id()


class PositionCreator:
    def __init__(self, session, commit_batch=100, check_duplicates=True, progress_interval=1000):
        self.session = session
        self.commit_batch = commit_batch
        self.check_duplicates = check_duplicates
        self.progress_interval = progress_interval

        # Statistics tracking
        self.stats = {
            'created_count': 0,
            'failed_count': 0,
            'duplicate_count': 0,
            'total_attempts': 0,
            'start_time': datetime.now()
        }

        # For duplicate checking
        self.existing_positions = set()
        if check_duplicates:
            self._load_existing_positions()

    def _load_existing_positions(self):
        """Load existing position combinations to avoid duplicates."""
        info_id("Loading existing positions for duplicate checking...")
        positions = self.session.query(Position).all()
        for pos in positions:
            # Create a tuple representing the unique combination
            pos_key = (
                pos.area_id,
                pos.equipment_group_id,
                pos.model_id,
                pos.asset_number_id,
                pos.location_id,
                pos.subassembly_id,
                pos.component_assembly_id,
                pos.assembly_view_id,
                pos.site_location_id
            )
            self.existing_positions.add(pos_key)
        info_id(f"Loaded {len(self.existing_positions)} existing positions for duplicate checking.")

    def _position_exists(self, area_id, equipment_group_id, model_id, asset_number_id=None,
                         location_id=None, subassembly_id=None, component_assembly_id=None,
                         assembly_view_id=None, site_location_id=None):
        """Check if a position combination already exists."""
        if not self.check_duplicates:
            return False

        pos_key = (
            area_id, equipment_group_id, model_id, asset_number_id,
            location_id, subassembly_id, component_assembly_id,
            assembly_view_id, site_location_id
        )
        return pos_key in self.existing_positions

    def _create_position(self, area_id, equipment_group_id, model_id, asset_number_id=None,
                         location_id=None, subassembly_id=None, component_assembly_id=None,
                         assembly_view_id=None, site_location_id=None, path_description=""):
        """Create a single position with error handling and duplicate checking."""
        self.stats['total_attempts'] += 1

        # Check for duplicates
        if self._position_exists(area_id, equipment_group_id, model_id, asset_number_id,
                                 location_id, subassembly_id, component_assembly_id,
                                 assembly_view_id, site_location_id):
            self.stats['duplicate_count'] += 1
            debug_id(f"Skipping duplicate position: {path_description}")
            return None

        try:
            pos_id = Position.add_to_db(
                session=self.session,
                area_id=area_id,
                equipment_group_id=equipment_group_id,
                model_id=model_id,
                asset_number_id=asset_number_id,
                location_id=location_id,
                subassembly_id=subassembly_id,
                component_assembly_id=component_assembly_id,
                assembly_view_id=assembly_view_id,
                site_location_id=site_location_id
            )

            # Add to existing positions set to avoid creating duplicates in this run
            if self.check_duplicates:
                pos_key = (area_id, equipment_group_id, model_id, asset_number_id,
                           location_id, subassembly_id, component_assembly_id,
                           assembly_view_id, site_location_id)
                self.existing_positions.add(pos_key)

            self.stats['created_count'] += 1
            debug_id(f"Created position {pos_id}: {path_description}")
            return pos_id

        except Exception as ex:
            self.stats['failed_count'] += 1
            error_id(f"Failed to create position for {path_description}: {ex}")
            return None

    def _report_progress(self, force=False):
        """Report progress at regular intervals."""
        total_processed = self.stats['created_count'] + self.stats['failed_count'] + self.stats['duplicate_count']

        if force or (total_processed > 0 and total_processed % self.progress_interval == 0):
            elapsed = datetime.now() - self.stats['start_time']
            rate = total_processed / elapsed.total_seconds() if elapsed.total_seconds() > 0 else 0

            info_id(f"Progress: {total_processed}/{self.stats['total_attempts']} processed "
                    f"({self.stats['created_count']} created, {self.stats['duplicate_count']} duplicates, "
                    f"{self.stats['failed_count']} failed) - {rate:.1f} positions/sec")

    def _commit_batch(self):
        """Commit current batch and report progress."""
        total_processed = self.stats['created_count'] + self.stats['failed_count'] + self.stats['duplicate_count']

        if self.commit_batch and total_processed % self.commit_batch == 0:
            try:
                self.session.commit()
                info_id(f"Committed batch at {total_processed} processed positions")
            except Exception as ex:
                error_id(f"Failed to commit batch: {ex}")
                self.session.rollback()
                raise

        self._report_progress()

    def create_all_positions(self, area_limit=None, skip_existing_check=False):
        """
        Create all possible positions in the hierarchy.

        Args:
            area_limit (int): Limit processing to first N areas (for testing)
            skip_existing_check (bool): Skip checking for existing positions
        """
        info_id("Starting comprehensive position creation...")

        try:
            # Query areas with optional limit
            areas_query = self.session.query(Area)
            if area_limit:
                areas_query = areas_query.limit(area_limit)
                info_id(f"Limited to first {area_limit} areas for processing")

            areas = areas_query.all()
            info_id(f"Processing {len(areas)} areas...")

            for area_idx, area in enumerate(areas, 1):
                info_id(f"Processing Area {area_idx}/{len(areas)}: {area.name} (ID: {area.id})")

                for eg in area.equipment_group:
                    debug_id(f"Processing EquipmentGroup: {eg.name} (ID: {eg.id})")

                    for model in eg.model:
                        debug_id(f"Processing Model: {model.name} (ID: {model.id})")

                        # --- AssetNumber Path ---
                        for asset in model.asset_number:
                            path_desc = f"Area={area.id}, EG={eg.id}, Model={model.id}, Asset={asset.id}"
                            self._create_position(
                                area_id=area.id,
                                equipment_group_id=eg.id,
                                model_id=model.id,
                                asset_number_id=asset.id,
                                path_description=f"AssetNumber path: {path_desc}"
                            )
                            self._commit_batch()

                        # --- Location Path ---
                        for location in model.location:
                            self._process_location_hierarchy(area, eg, model, location)

            # Final commit and summary
            self.session.commit()
            self._report_progress(force=True)
            self._print_final_summary()

        except Exception as e:
            error_id(f"Fatal error during position creation: {e}", exc_info=True)
            self.session.rollback()
            raise

    def _process_location_hierarchy(self, area, eg, model, location):
        """Process the location hierarchy (Location → Subassembly → ComponentAssembly → AssemblyView)."""
        base_path = f"Area={area.id}, EG={eg.id}, Model={model.id}, Location={location.id}"

        # Create position for location if no subassemblies
        if not location.subassembly:
            self._create_position(
                area_id=area.id,
                equipment_group_id=eg.id,
                model_id=model.id,
                location_id=location.id,
                path_description=f"Location path: {base_path}"
            )
            self._commit_batch()

        # Process subassemblies
        for subassembly in location.subassembly:
            sub_path = f"{base_path}, Sub={subassembly.id}"

            # Create position for subassembly if no component assemblies
            if not subassembly.component_assembly:
                self._create_position(
                    area_id=area.id,
                    equipment_group_id=eg.id,
                    model_id=model.id,
                    location_id=location.id,
                    subassembly_id=subassembly.id,
                    path_description=f"Subassembly path: {sub_path}"
                )
                self._commit_batch()

            # Process component assemblies
            for comp_assembly in subassembly.component_assembly:
                comp_path = f"{sub_path}, CompAsm={comp_assembly.id}"

                # Create position for component assembly if no assembly views
                if not comp_assembly.assembly_view:
                    self._create_position(
                        area_id=area.id,
                        equipment_group_id=eg.id,
                        model_id=model.id,
                        location_id=location.id,
                        subassembly_id=subassembly.id,
                        component_assembly_id=comp_assembly.id,
                        path_description=f"ComponentAssembly path: {comp_path}"
                    )
                    self._commit_batch()

                # Process assembly views
                for av in comp_assembly.assembly_view:
                    av_path = f"{comp_path}, AV={av.id}"
                    self._create_position(
                        area_id=area.id,
                        equipment_group_id=eg.id,
                        model_id=model.id,
                        location_id=location.id,
                        subassembly_id=subassembly.id,
                        component_assembly_id=comp_assembly.id,
                        assembly_view_id=av.id,
                        path_description=f"AssemblyView path: {av_path}"
                    )
                    self._commit_batch()

    def _print_final_summary(self):
        """Print final summary statistics."""
        elapsed = datetime.now() - self.stats['start_time']
        total_processed = self.stats['created_count'] + self.stats['failed_count'] + self.stats['duplicate_count']
        rate = total_processed / elapsed.total_seconds() if elapsed.total_seconds() > 0 else 0

        info_id("=" * 60)
        info_id("POSITION CREATION SUMMARY")
        info_id("=" * 60)
        info_id(f"Total Attempts:     {self.stats['total_attempts']:,}")
        info_id(f"Successfully Created: {self.stats['created_count']:,}")
        info_id(f"Duplicates Skipped:   {self.stats['duplicate_count']:,}")
        info_id(f"Failed:               {self.stats['failed_count']:,}")
        info_id(f"Total Processed:      {total_processed:,}")
        info_id(f"Processing Time:      {elapsed}")
        info_id(f"Average Rate:         {rate:.1f} positions/second")
        info_id("=" * 60)

        if self.stats['failed_count'] > 0:
            warning_id(f"Warning: {self.stats['failed_count']} positions failed to create. Check logs for details.")


def main():
    parser = argparse.ArgumentParser(description="Create all possible equipment positions")
    parser.add_argument("--batch-size", type=int, default=100,
                        help="Number of operations per commit batch (default: 100)")
    parser.add_argument("--no-duplicate-check", action="store_true",
                        help="Skip duplicate checking (faster but may create duplicates)")
    parser.add_argument("--progress-interval", type=int, default=1000,
                        help="Report progress every N positions (default: 1000)")
    parser.add_argument("--area-limit", type=int,
                        help="Limit processing to first N areas (for testing)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be created without actually creating positions")

    args = parser.parse_args()

    # Database session
    db_config = DatabaseConfig()
    session = db_config.get_main_session()

    try:
        creator = PositionCreator(
            session=session,
            commit_batch=args.batch_size,
            check_duplicates=not args.no_duplicate_check,
            progress_interval=args.progress_interval
        )

        if args.dry_run:
            info_id("DRY RUN MODE - No positions will be created")
            # You could implement a dry-run mode here that just counts what would be created

        creator.create_all_positions(area_limit=args.area_limit)

    except KeyboardInterrupt:
        warning_id("Operation interrupted by user")
        session.rollback()
    except Exception as e:
        error_id(f"Fatal error: {e}", exc_info=True)
        session.rollback()
        raise
    finally:
        session.close()


if __name__ == "__main__":
    main()