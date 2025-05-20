#!/usr/bin/env python3
"""
Academic Subject Population Script - Modified Hierarchy

This script populates the database with academic subjects using a restructured hierarchy:
- Area → "Academic" (top level container)
- EquipmentGroup → Academic Fields (Physics, Business, etc.)
- Model → Subjects (Marketing, Mechanics, etc.)
- (and so on down the hierarchy)

Usage:
    python AcademicSubjectRestructured.py
"""

import os
import sys
import traceback
from sqlalchemy.exc import SQLAlchemyError

# Debug - Print current information
print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")
print(f"Current working directory: {os.getcwd()}")
print(f"Python path: {sys.path}")

# Add the project root to the Python path if needed
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"Added project root to sys.path: {project_root}")

try:
    # Import project configuration, database, and logging
    from modules.configuration.config_env import DatabaseConfig

    print("Successfully imported DatabaseConfig")

    from modules.configuration.log_config import logger, with_request_id, info_id, error_id, debug_id

    print("Successfully imported logging modules")

    # Import your models - adjust this import path as needed
    try:
        from modules.emtacdb.emtacdb_fts import Area, EquipmentGroup, Model

        print("Successfully imported database models")
    except ImportError as e:
        print(f"Failed to import database models: {e}")
        print("Trying alternative import paths...")

        # Try alternative import locations
        try:
            from modules.emtacdb.models import Area, EquipmentGroup, Model

            print("Successfully imported models from modules.emtacdb.models")
        except ImportError:
            print("Failed to import from modules.emtacdb.models")

            # Another potential location
            try:
                from modules.emtacdb.emtacdb_models import Area, EquipmentGroup, Model

                print("Successfully imported models from modules.emtacdb.emtacdb_models")
            except ImportError:
                print("Failed to import from modules.emtacdb.emtacdb_models")
                raise
except Exception as e:
    print(f"Import error: {e}")
    print("Traceback:")
    traceback.print_exc()
    sys.exit(1)

print("All imports successful!")

# Initialize DatabaseConfig - ONLY ONCE
try:
    db_config = DatabaseConfig()
    print("DatabaseConfig initialized successfully")
except Exception as e:
    print(f"Error initializing DatabaseConfig: {e}")
    traceback.print_exc()
    sys.exit(1)


@with_request_id
def add_area(session, name, description=None, request_id=None):
    """Add an academic field (Area) and return its ID."""
    try:
        # Check if area already exists
        existing = session.query(Area).filter(Area.name == name).first()
        if existing:
            info_id(f"Area '{name}' already exists with ID {existing.id}", request_id)
            return existing.id

        # Create new area
        area = Area(name=name, description=description)
        session.add(area)
        session.flush()  # Get the ID without committing
        info_id(f"Added area: {name} (ID: {area.id})", request_id)
        return area.id
    except Exception as e:
        error_id(f"Error adding area '{name}': {str(e)}", request_id)
        print(f"Error adding area '{name}': {str(e)}")
        traceback.print_exc()
        raise


@with_request_id
def add_equipment_group(session, name, area_id, description=None, request_id=None):
    """Add a subject (EquipmentGroup) and return its ID."""
    try:
        # Check if subject already exists
        existing = session.query(EquipmentGroup).filter(
            EquipmentGroup.name == name,
            EquipmentGroup.area_id == area_id
        ).first()
        if existing:
            info_id(f"Subject '{name}' already exists with ID {existing.id}", request_id)
            return existing.id

        # Create new subject
        subject = EquipmentGroup(name=name, area_id=area_id, description=description)
        session.add(subject)
        session.flush()  # Get the ID without committing
        info_id(f"Added subject: {name} (ID: {subject.id}) to area ID {area_id}", request_id)
        return subject.id
    except Exception as e:
        error_id(f"Error adding subject '{name}': {str(e)}", request_id)
        print(f"Error adding subject '{name}': {str(e)}")
        traceback.print_exc()
        raise


@with_request_id
def add_model(session, name, equipment_group_id, description=None, request_id=None):
    """Add a branch/subdiscipline (Model) and return its ID."""
    try:
        # Check if branch already exists
        existing = session.query(Model).filter(
            Model.name == name,
            Model.equipment_group_id == equipment_group_id
        ).first()
        if existing:
            info_id(f"Branch '{name}' already exists with ID {existing.id}", request_id)
            return existing.id

        # Create new branch
        branch = Model(name=name, equipment_group_id=equipment_group_id, description=description)
        session.add(branch)
        session.flush()  # Get the ID without committing
        info_id(f"Added branch: {name} (ID: {branch.id}) to subject ID {equipment_group_id}", request_id)
        return branch.id
    except Exception as e:
        error_id(f"Error adding branch '{name}': {str(e)}", request_id)
        print(f"Error adding branch '{name}': {str(e)}")
        traceback.print_exc()
        raise


def populate_business_field(session, academic_area_id):
    """Populate business academic field and subjects under the academic area."""
    logger.info("Populating Business academic field...")

    # Business Administration as an EquipmentGroup under Academic
    business_id = add_equipment_group(
        session,
        "Business Administration",
        academic_area_id,
        "Study of management and operation of business enterprises"
    )

    # Business subjects as Models
    marketing_id = add_model(
        session,
        "Marketing",
        business_id,
        "Study of promoting and selling products or services"
    )

    finance_id = add_model(
        session,
        "Finance",
        business_id,
        "Study of money management and asset allocation"
    )

    management_id = add_model(
        session,
        "Management",
        business_id,
        "Study of organizational leadership and administration"
    )

    accounting_id = add_model(
        session,
        "Accounting",
        business_id,
        "Study of recording, classifying, and summarizing financial transactions"
    )

    hr_id = add_model(
        session,
        "Human Resources",
        business_id,
        "Study of managing an organization's workforce"
    )

    operations_id = add_model(
        session,
        "Operations Management",
        business_id,
        "Study of designing and controlling production processes"
    )

    analytics_id = add_model(
        session,
        "Business Analytics",
        business_id,
        "Study of data analysis techniques for business insights"
    )

    entrepreneurship_id = add_model(
        session,
        "Entrepreneurship",
        business_id,
        "Study of starting and running new business ventures"
    )

    international_id = add_model(
        session,
        "International Business",
        business_id,
        "Study of global business operations and strategies"
    )

    logger.info("Finished populating Business academic field")
    return business_id


def populate_science_fields(session, academic_area_id):
    """Populate science academic fields under the academic area."""
    logger.info("Populating Science academic fields...")

    # Physics as an EquipmentGroup under Academic
    physics_id = add_equipment_group(
        session,
        "Physics",
        academic_area_id,
        "Study of matter, energy, and their interactions"
    )

    # Physics subjects as Models
    mechanics_id = add_model(
        session,
        "Mechanics",
        physics_id,
        "Study of motion and forces"
    )

    quantum_id = add_model(
        session,
        "Quantum Physics",
        physics_id,
        "Study of subatomic particles and their behaviors"
    )

    thermo_id = add_model(
        session,
        "Thermodynamics",
        physics_id,
        "Study of heat, energy, and work"
    )

    em_id = add_model(
        session,
        "Electromagnetism",
        physics_id,
        "Study of electromagnetic force and fields"
    )

    relativity_id = add_model(
        session,
        "Relativity",
        physics_id,
        "Study of space, time, and gravity"
    )

    # Chemistry as an EquipmentGroup under Academic
    chemistry_id = add_equipment_group(
        session,
        "Chemistry",
        academic_area_id,
        "Study of substances, their properties, and reactions"
    )

    # Chemistry subjects as Models
    organic_id = add_model(
        session,
        "Organic Chemistry",
        chemistry_id,
        "Study of carbon-containing compounds"
    )

    inorganic_id = add_model(
        session,
        "Inorganic Chemistry",
        chemistry_id,
        "Study of non-carbon compounds"
    )

    physical_chem_id = add_model(
        session,
        "Physical Chemistry",
        chemistry_id,
        "Study of how matter behaves on a molecular and atomic level"
    )

    biochem_id = add_model(
        session,
        "Biochemistry",
        chemistry_id,
        "Study of chemical processes within living organisms"
    )

    analytical_chem_id = add_model(
        session,
        "Analytical Chemistry",
        chemistry_id,
        "Study of separation, identification, and quantification of matter"
    )

    # Biology as an EquipmentGroup under Academic
    biology_id = add_equipment_group(
        session,
        "Biology",
        academic_area_id,
        "Study of living organisms"
    )

    # Biology subjects as Models
    molecular_id = add_model(
        session,
        "Molecular Biology",
        biology_id,
        "Study of biological activity at the molecular level"
    )

    genetics_id = add_model(
        session,
        "Genetics",
        biology_id,
        "Study of genes, heredity, and genetic variation"
    )

    ecology_id = add_model(
        session,
        "Ecology",
        biology_id,
        "Study of interactions between organisms and their environment"
    )

    logger.info("Finished populating Science academic fields")
    return [physics_id, chemistry_id, biology_id]


def populate_mathematics_fields(session, academic_area_id):
    """Populate mathematics academic fields under the academic area."""
    logger.info("Populating Mathematics academic fields...")

    # Pure Mathematics as an EquipmentGroup under Academic
    pure_math_id = add_equipment_group(
        session,
        "Pure Mathematics",
        academic_area_id,
        "Study of abstract concepts and structures"
    )

    # Pure Mathematics subjects as Models
    algebra_id = add_model(
        session,
        "Algebra",
        pure_math_id,
        "Study of mathematical symbols and rules"
    )

    calculus_id = add_model(
        session,
        "Calculus",
        pure_math_id,
        "Study of continuous change and functions"
    )

    geometry_id = add_model(
        session,
        "Geometry",
        pure_math_id,
        "Study of shapes, sizes, and properties of space"
    )

    number_theory_id = add_model(
        session,
        "Number Theory",
        pure_math_id,
        "Study of integers and integer-valued functions"
    )

    topology_id = add_model(
        session,
        "Topology",
        pure_math_id,
        "Study of properties preserved under continuous deformations"
    )

    # Applied Mathematics as an EquipmentGroup under Academic
    applied_math_id = add_equipment_group(
        session,
        "Applied Mathematics",
        academic_area_id,
        "Application of mathematical methods to solve real-world problems"
    )

    # Applied Mathematics subjects as Models
    stats_id = add_model(
        session,
        "Statistics",
        applied_math_id,
        "Study of data collection, analysis, and interpretation"
    )

    operations_research_id = add_model(
        session,
        "Operations Research",
        applied_math_id,
        "Application of analytical methods for decision-making"
    )

    modeling_id = add_model(
        session,
        "Mathematical Modeling",
        applied_math_id,
        "Using mathematics to describe real-world phenomena"
    )

    financial_math_id = add_model(
        session,
        "Financial Mathematics",
        applied_math_id,
        "Application of mathematical methods in finance"
    )

    logger.info("Finished populating Mathematics academic fields")
    return [pure_math_id, applied_math_id]


def populate_computer_science_fields(session, academic_area_id):
    """Populate computer science fields under the academic area."""
    logger.info("Populating Computer Science academic fields...")

    # Computer Science as an EquipmentGroup under Academic
    cs_id = add_equipment_group(
        session,
        "Computer Science",
        academic_area_id,
        "Study of computation, algorithms, and information processing"
    )

    # Computer Science subjects as Models
    algorithms_id = add_model(
        session,
        "Algorithms",
        cs_id,
        "Study of computational procedures and problem-solving methods"
    )

    data_structures_id = add_model(
        session,
        "Data Structures",
        cs_id,
        "Study of organizing and storing data efficiently"
    )

    ai_id = add_model(
        session,
        "Artificial Intelligence",
        cs_id,
        "Study of intelligent agent development and machine learning"
    )

    db_id = add_model(
        session,
        "Database Systems",
        cs_id,
        "Study of data organization, storage, and retrieval methods"
    )

    networks_id = add_model(
        session,
        "Computer Networks",
        cs_id,
        "Study of data communication systems and protocols"
    )

    se_id = add_model(
        session,
        "Software Engineering",
        cs_id,
        "Study of systematic development of software applications"
    )

    security_id = add_model(
        session,
        "Cybersecurity",
        cs_id,
        "Study of protecting computer systems from unauthorized access and attacks"
    )

    theory_id = add_model(
        session,
        "Theoretical Computer Science",
        cs_id,
        "Mathematical study of computation and algorithms"
    )

    hci_id = add_model(
        session,
        "Human-Computer Interaction",
        cs_id,
        "Study of interaction between humans and computers"
    )

    logger.info("Finished populating Computer Science academic fields")
    return cs_id


def populate_humanities_fields(session, academic_area_id):
    """Populate humanities fields under the academic area."""
    logger.info("Populating Humanities academic fields...")

    # Philosophy as an EquipmentGroup under Academic
    philosophy_id = add_equipment_group(
        session,
        "Philosophy",
        academic_area_id,
        "Study of fundamental questions about existence, knowledge, ethics, and more"
    )

    # Philosophy subjects as Models
    ethics_id = add_model(
        session,
        "Ethics",
        philosophy_id,
        "Study of moral principles and values"
    )

    epistemology_id = add_model(
        session,
        "Epistemology",
        philosophy_id,
        "Study of knowledge, belief, and justification"
    )

    metaphysics_id = add_model(
        session,
        "Metaphysics",
        philosophy_id,
        "Study of reality, existence, and being"
    )

    logic_id = add_model(
        session,
        "Logic",
        philosophy_id,
        "Study of valid reasoning and inference"
    )

    # History as an EquipmentGroup under Academic
    history_id = add_equipment_group(
        session,
        "History",
        academic_area_id,
        "Study of past events, societies, and civilizations"
    )

    # History subjects as Models
    world_history_id = add_model(
        session,
        "World History",
        history_id,
        "Study of history on a global scale"
    )

    us_history_id = add_model(
        session,
        "U.S. History",
        history_id,
        "Study of United States history"
    )

    ancient_history_id = add_model(
        session,
        "Ancient History",
        history_id,
        "Study of early human civilizations"
    )

    medieval_history_id = add_model(
        session,
        "Medieval History",
        history_id,
        "Study of the Middle Ages"
    )

    modern_history_id = add_model(
        session,
        "Modern History",
        history_id,
        "Study of recent centuries of human history"
    )

    # Literature as an EquipmentGroup under Academic
    literature_id = add_equipment_group(
        session,
        "Literature",
        academic_area_id,
        "Study of written works of art"
    )

    # Literature subjects as Models
    english_lit_id = add_model(
        session,
        "English Literature",
        literature_id,
        "Study of literature written in English"
    )

    world_lit_id = add_model(
        session,
        "World Literature",
        literature_id,
        "Study of literature from various cultures and languages"
    )

    poetry_id = add_model(
        session,
        "Poetry",
        literature_id,
        "Study of poetic works and forms"
    )

    drama_id = add_model(
        session,
        "Drama",
        literature_id,
        "Study of theatrical works"
    )

    logger.info("Finished populating Humanities academic fields")
    return [philosophy_id, history_id, literature_id]


def populate_social_sciences(session, academic_area_id):
    """Populate social sciences fields under the academic area."""
    logger.info("Populating Social Sciences academic fields...")

    # Psychology as an EquipmentGroup under Academic
    psychology_id = add_equipment_group(
        session,
        "Psychology",
        academic_area_id,
        "Study of mind and behavior"
    )

    # Psychology subjects as Models
    clinical_id = add_model(
        session,
        "Clinical Psychology",
        psychology_id,
        "Study and treatment of mental illness and distress"
    )

    cognitive_id = add_model(
        session,
        "Cognitive Psychology",
        psychology_id,
        "Study of mental processes including perception, thinking, and memory"
    )

    developmental_id = add_model(
        session,
        "Developmental Psychology",
        psychology_id,
        "Study of psychological growth across the lifespan"
    )

    social_psych_id = add_model(
        session,
        "Social Psychology",
        psychology_id,
        "Study of how individuals' thoughts and behavior are influenced by others"
    )

    # Economics as an EquipmentGroup under Academic
    economics_id = add_equipment_group(
        session,
        "Economics",
        academic_area_id,
        "Study of production, distribution, and consumption of goods and services"
    )

    # Economics subjects as Models
    micro_id = add_model(
        session,
        "Microeconomics",
        economics_id,
        "Study of individual and business decisions regarding resource allocation"
    )

    macro_id = add_model(
        session,
        "Macroeconomics",
        economics_id,
        "Study of economy-wide phenomena like inflation, growth, and unemployment"
    )

    international_econ_id = add_model(
        session,
        "International Economics",
        economics_id,
        "Study of economic interactions between countries"
    )

    econometrics_id = add_model(
        session,
        "Econometrics",
        economics_id,
        "Application of statistical methods to economic data"
    )

    # Sociology as an EquipmentGroup under Academic
    sociology_id = add_equipment_group(
        session,
        "Sociology",
        academic_area_id,
        "Study of society, social relationships, and culture"
    )

    # Sociology subjects as Models
    cultural_id = add_model(
        session,
        "Cultural Sociology",
        sociology_id,
        "Study of the influence of culture on social life"
    )

    urban_id = add_model(
        session,
        "Urban Sociology",
        sociology_id,
        "Study of social life and interactions in urban environments"
    )

    social_inequality_id = add_model(
        session,
        "Social Inequality",
        sociology_id,
        "Study of social differences and hierarchies"
    )

    logger.info("Finished populating Social Sciences academic fields")
    return [psychology_id, economics_id, sociology_id]


def populate_industrial_manufacturing_fields(session, industrial_area_id):
    """Populate industrial manufacturing fields and subjects."""
    logger.info("Populating Industrial Manufacturing fields...")

    # WELDING as an EquipmentGroup under Industrial Manufacturing
    welding_id = add_equipment_group(
        session,
        "Welding Technology",
        industrial_area_id,
        "Study of joining materials through fusion processes"
    )

    # Welding subjects/branches as Models
    add_model(
        session,
        "Arc Welding",
        welding_id,
        "Welding processes using an electric arc (SMAW, GMAW, GTAW)"
    )

    add_model(
        session,
        "Resistance Welding",
        welding_id,
        "Welding processes that use electrical resistance to generate heat"
    )

    add_model(
        session,
        "Oxyfuel Welding",
        welding_id,
        "Welding using fuel gases and oxygen to produce a flame"
    )

    add_model(
        session,
        "Welding Metallurgy",
        welding_id,
        "Study of metal properties and behaviors during welding"
    )

    add_model(
        session,
        "Welding Inspection",
        welding_id,
        "Quality control and testing of welded joints"
    )

    add_model(
        session,
        "Welding Automation",
        welding_id,
        "Automated and robotic welding systems and programming"
    )

    # MACHINING as an EquipmentGroup under Industrial Manufacturing
    machining_id = add_equipment_group(
        session,
        "Machining Technology",
        industrial_area_id,
        "Study of material removal processes to create parts"
    )

    # Machining subjects/branches as Models
    add_model(
        session,
        "CNC Machining",
        machining_id,
        "Computer numerical control machining processes and programming"
    )

    add_model(
        session,
        "Manual Machining",
        machining_id,
        "Traditional machine tool operation (lathes, mills, drill presses)"
    )

    add_model(
        session,
        "Precision Measurement",
        machining_id,
        "Metrology techniques and instruments for machined parts"
    )

    add_model(
        session,
        "CAD/CAM Systems",
        machining_id,
        "Computer-aided design and manufacturing for machining"
    )

    add_model(
        session,
        "Advanced Machining Processes",
        machining_id,
        "Non-traditional processes like EDM, waterjet, and laser cutting"
    )

    add_model(
        session,
        "Tool Design",
        machining_id,
        "Design of cutting tools, fixtures, and machine tooling"
    )

    # ELECTRICAL as an EquipmentGroup under Industrial Manufacturing
    electrical_id = add_equipment_group(
        session,
        "Industrial Electrical Systems",
        industrial_area_id,
        "Study of electrical systems in industrial settings"
    )

    # Electrical subjects/branches as Models
    add_model(
        session,
        "Power Distribution",
        electrical_id,
        "Industrial power systems and distribution networks"
    )

    add_model(
        session,
        "Motor Controls",
        electrical_id,
        "Electric motor operation, control, and protection systems"
    )

    add_model(
        session,
        "Industrial Controls",
        electrical_id,
        "PLCs, SCADA, and other industrial control systems"
    )

    add_model(
        session,
        "Electrical Troubleshooting",
        electrical_id,
        "Diagnosing and repairing electrical system faults"
    )

    add_model(
        session,
        "Electrical Safety",
        electrical_id,
        "Hazard identification and safety practices for electrical work"
    )

    add_model(
        session,
        "Industrial IoT",
        electrical_id,
        "Internet of Things applications in industrial settings"
    )

    # HYDRAULIC/PNEUMATIC as an EquipmentGroup under Industrial Manufacturing
    fluid_power_id = add_equipment_group(
        session,
        "Fluid Power Systems",
        industrial_area_id,
        "Study of hydraulic and pneumatic systems for power transmission"
    )

    # Hydraulic/Pneumatic subjects/branches as Models
    add_model(
        session,
        "Hydraulic Systems",
        fluid_power_id,
        "Liquid-based power transmission systems design and operation"
    )

    add_model(
        session,
        "Pneumatic Systems",
        fluid_power_id,
        "Compressed air power systems design and operation"
    )

    add_model(
        session,
        "Fluid Power Components",
        fluid_power_id,
        "Pumps, valves, actuators, and other hydraulic/pneumatic components"
    )

    add_model(
        session,
        "Fluid Power Maintenance",
        fluid_power_id,
        "Troubleshooting, repair, and preventive maintenance"
    )

    add_model(
        session,
        "Electrohydraulic Systems",
        fluid_power_id,
        "Integration of electronics with hydraulic systems"
    )

    add_model(
        session,
        "Fluid Power Circuit Design",
        fluid_power_id,
        "Designing and analyzing hydraulic and pneumatic circuits"
    )

    logger.info("Finished populating Industrial Manufacturing fields")
    return [welding_id, machining_id, electrical_id, fluid_power_id]


@with_request_id
def verify_database(request_id=None):
    """Verify that the database is ready for population."""
    try:
        # Get a session using the DatabaseConfig
        print("Getting database session...")
        session = db_config.get_main_session()
        print("Database session acquired")
        try:
            # Check if the required tables exist by querying them
            print("Verifying database tables...")
            try:
                area_count = session.query(Area).count()
                print(f"Found {area_count} areas in database")
                equipment_group_count = session.query(EquipmentGroup).count()
                print(f"Found {equipment_group_count} equipment groups in database")
                model_count = session.query(Model).count()
                print(f"Found {model_count} models in database")

                debug_id(
                    f"Found {area_count} areas, {equipment_group_count} equipment groups, and {model_count} models in database",
                    request_id)
                return True
            except Exception as e:
                print(f"Error querying tables: {e}")
                traceback.print_exc()
                raise
        except SQLAlchemyError as e:
            error_id(f"Error verifying database schema: {str(e)}", request_id)
            print("Database tables not found or accessible. Please ensure the database is properly set up.")
            print(f"Error details: {str(e)}")
            traceback.print_exc()
            return False
        finally:
            print("Closing database session")
            session.close()
    except Exception as e:
        error_id(f"Failed to connect to database: {str(e)}", request_id)
        print(f"Database connection failed: {str(e)}")
        traceback.print_exc()
        return False


@with_request_id
def main(request_id=None):
    """Main function to populate all academic fields."""
    print("Main function started")
    info_id("Starting academic subject population script with restructured hierarchy...", request_id)
    print("Academic Subject Population Script - Restructured Hierarchy")
    print("==========================================================")

    # Verify database is ready
    print("Verifying database...")
    if not verify_database():
        error_id("Database verification failed. Exiting.", request_id)
        print("Database verification failed. Please check your database setup.")
        return

    try:
        # Get a session from DatabaseConfig
        print("Getting main session for population...")
        session = db_config.get_main_session()
        print("Session acquired successfully")
        try:
            print("Creating top-level Academic area...")

            # Create the Academic Area (top level)
            academic_area_id = add_area(
                session,
                "Academic",
                "Academic and technical knowledge across various fields"
            )
            print(f"✓ Created Academic area with ID: {academic_area_id}")

            print("\nPopulating academic fields under Academic area...")

            # Populate all fields under the Academic area
            business_id = populate_business_field(session, academic_area_id)
            print("✓ Business field populated")

            science_ids = populate_science_fields(session, academic_area_id)
            print("✓ Science fields populated")

            math_ids = populate_mathematics_fields(session, academic_area_id)
            print("✓ Mathematics fields populated")

            cs_id = populate_computer_science_fields(session, academic_area_id)
            print("✓ Computer Science field populated")

            humanities_ids = populate_humanities_fields(session, academic_area_id)
            print("✓ Humanities fields populated")

            social_science_ids = populate_social_sciences(session, academic_area_id)
            print("✓ Social Sciences fields populated")

            print("\nPopulating industrial fields under Academic area...")

            # Populate industrial fields under the Academic area
            industrial_ids = populate_industrial_manufacturing_fields(session, academic_area_id)
            print("✓ Industrial Manufacturing fields populated")

            # Commit the changes
            print("Committing changes to database...")
            session.commit()
            print("Changes committed successfully")

            info_id("Successfully populated all fields under the Academic area", request_id)
            print("\nSuccess! All subjects have been populated under the Academic area.")
            print(f"Academic area ID: {academic_area_id}")
        except Exception as e:
            # Roll back the transaction on error
            print(f"Error during population, rolling back: {e}")
            session.rollback()
            error_id(f"Error in database transaction: {str(e)}", request_id)
            print(f"\nError: {str(e)}")
            traceback.print_exc()
            raise
        finally:
            # Close the session
            print("Closing database session")
            session.close()
    except Exception as e:
        error_id(f"Error in main execution: {str(e)}", request_id)
        print(f"\nError: {str(e)}")
        traceback.print_exc()

    info_id("Academic subject population script completed", request_id)
    print("\nScript completed. Check the logs for details.")


# THIS IS THE CRITICAL PART
if __name__ == "__main__":
    print("Executing main function from __main__ block")
    try:
        main()
    except Exception as e:
        print(f"Unhandled exception in main: {e}")
        traceback.print_exc()
else:
    print(f"Note: Script was imported as a module, __name__ = {__name__}")

print("Script execution complete")