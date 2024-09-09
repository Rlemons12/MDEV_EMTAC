from flask import Blueprint, request, redirect, url_for, flash
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
import logging
from blueprints import DATABASE_URL
from emtacdb_fts import Problem, Solution, ProblemPositionAssociation, Position

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create SQLAlchemy engine and session
engine = create_engine(DATABASE_URL)
Session = scoped_session(sessionmaker(bind=engine))

# Create the Blueprint
troubleshooting_guide_edit_update_bp = Blueprint('troubleshooting_guide_edit_update_bp', __name__)

@troubleshooting_guide_edit_update_bp.route('/troubleshooting_guide_edit_update', methods=['POST'])
def edit_update_problem_solution():
    logger.info("Edit Problem/Solution Update route accessed")

    # Get the form data
    problem_id = request.form.get('problem_id')
    problem_name = request.form.get('problem_name')
    solution_id = request.form.get('solution_id')
    problem_description = request.form.get('problem_description')
    solution_description = request.form.get('solution_description')

    logger.info(f"Received form data - Problem ID: {problem_id}, Problem Name: {problem_name}, "
                f"Solution ID: {solution_id}, Problem Description: {problem_description}, "
                f"Solution Description: {solution_description}")

    session = Session()
    try:
        # Retrieve and update the problem record
        problem = session.query(Problem).filter_by(id=problem_id).first()
        if problem:
            problem.name = problem_name
            problem.description = problem_description
            logger.info(f"Updated Problem: {problem.id} - {problem.name}")

        # Retrieve and update the solution record
        solution = session.query(Solution).filter_by(id=solution_id).first()
        if solution:
            solution.description = solution_description
            logger.info(f"Updated Solution: {solution.id}")

        # Commit the changes to the database
        session.commit()
        logger.info("Problem and Solution updated successfully")
        flash('Problem and Solution updated successfully', 'success')

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        session.rollback()
        flash(f'An error occurred: {e}', 'danger')

    finally:
        session.close()

    # Redirect back to the troubleshooting guide page after updating
    return redirect(url_for('troubleshooting_guide'))  # Assuming this is the page to go back to
