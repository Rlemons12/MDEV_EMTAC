# pst_troubleshooting_solution.py

from flask import Blueprint, jsonify
from sqlalchemy.exc import SQLAlchemyError
from config_env import DatabaseConfig  # Adjust import based on your project structure
from emtacdb_fts import Problem, Solution
import logging

# Initialize Database Config
db_config = DatabaseConfig()

logger = logging.getLogger(__name__)

# Define a new blueprint for solution-related routes
pst_troubleshooting_solution_bp = Blueprint('pst_troubleshooting_solution_bp', __name__)


@pst_troubleshooting_solution_bp.route('/get_solutions/<int:problem_id>', methods=['GET'])
def get_solutions(problem_id):
    """
    Retrieve solutions related to the specified problem.
    """
    session = db_config.get_main_session()
    try:
        # Query the Solution table for solutions with the specified problem_id
        solutions = session.query(Solution).filter_by(problem_id=problem_id).all()

        if not solutions:
            return jsonify({'error': 'No solutions found for this problem'}), 404

        # Format the solutions data including id, name, and description
        solutions_data = [{'id': solution.id, 'name': solution.name, 'description': solution.description} for solution
                          in solutions]

        return jsonify(solutions_data), 200

    except SQLAlchemyError as e:
        logger.error(f"Database error fetching solutions: {e}")
        return jsonify({'error': 'An error occurred while fetching solutions.'}), 500
    except Exception as e:
        logger.error(f"Unexpected error fetching solutions: {e}")
        return jsonify({'error': 'An unexpected error occurred while fetching solutions.'}), 500
    finally:
        session.close()
@pst_troubleshooting_solution_bp.route('/remove_solutions/', methods=['POST'])
def remove_solutions():
    """
    Remove solutions from a problem.
    """
    session = db_config.get_main_session()
    try:
        data = request.get_json()
        problem_id = data.get('problem_id')
        solution_ids = data.get('solution_ids')

        if not problem_id or not solution_ids:
            logger.error("Problem ID or Solution IDs missing in request.")
            return jsonify({'error': 'Problem ID and Solution IDs are required.'}), 400

        # Check if the problem exists
        problem = session.query(Problem).filter_by(id=problem_id).first()
        if not problem:
            logger.error(f"Problem with ID {problem_id} not found.")
            return jsonify({'error': 'Problem not found.'}), 404

        # Fetch solutions to be removed
        solutions = session.query(Solution).filter(Solution.id.in_(solution_ids), Solution.problem_id == problem_id).all()
        if not solutions:
            logger.error("No matching solutions found to remove.")
            return jsonify({'error': 'No matching solutions found to remove.'}), 404

        # Remove the solutions
        for solution in solutions:
            session.delete(solution)
        session.commit()

        logger.info(f"Removed solutions IDs {solution_ids} from problem ID {problem_id}.")
        return jsonify({'message': 'Solutions removed successfully.'}), 200

    except SQLAlchemyError as e:
        session.rollback()
        logger.error(f"Database error while removing solutions: {e}")
        return jsonify({'error': 'An error occurred while removing solutions.'}), 500
    except Exception as e:
        session.rollback()
        logger.error(f"Unexpected error while removing solutions: {e}")
        return jsonify({'error': 'An unexpected error occurred.'}), 500
    finally:
        session.close()

@pst_troubleshooting_solution_bp.route('/add_solution/', methods=['POST'])
def add_solution():
    """
    Add a new solution to a problem.
    """
    session = db_config.get_main_session()
    try:
        data = request.get_json()
        problem_id = data.get('problem_id')
        solution_name = data.get('name')

        if not problem_id or not solution_name:
            logger.error("Problem ID or Solution Name missing in request.")
            return jsonify({'error': 'Problem ID and Solution Name are required.'}), 400

        # Check if the problem exists
        problem = session.query(Problem).filter_by(id=problem_id).first()
        if not problem:
            logger.error(f"Problem with ID {problem_id} not found.")
            return jsonify({'error': 'Problem not found.'}), 404

        # Create and add the new solution
        new_solution = Solution(name=solution_name, problem_id=problem_id)
        session.add(new_solution)
        session.commit()

        logger.info(f"Added new solution '{solution_name}' to problem ID {problem_id}.")
        return jsonify({'message': 'Solution added successfully.', 'solution': {'id': new_solution.id, 'name': new_solution.name, 'description': new_solution.description}}), 201

    except SQLAlchemyError as e:
        session.rollback()
        logger.error(f"Database error while adding solution: {e}")
        return jsonify({'error': 'An error occurred while adding the solution.'}), 500
    except Exception as e:
        session.rollback()
        logger.error(f"Unexpected error while adding solution: {e}")
        return jsonify({'error': 'An unexpected error occurred.'}), 500
    finally:
        session.close()
