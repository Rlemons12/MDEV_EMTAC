# pst_troubleshooting_solution.py

from flask import Blueprint, jsonify
from sqlalchemy.exc import SQLAlchemyError
from config_env import DatabaseConfig  # Adjust import based on your project structure
from emtacdb_fts import Problem, Solution, Task, TaskSolutionAssociation,TaskPositionAssociation
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
    Delete solutions and their task associations for a given problem.
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

        # Fetch solutions linked to the problem that need to be deleted
        solutions_to_delete = session.query(Solution).filter(
            Solution.id.in_(solution_ids), Solution.problem_id == problem_id).all()
        if not solutions_to_delete:
            logger.error("No matching solutions found to delete.")
            return jsonify({'error': 'No matching solutions found to delete.'}), 404

        # Delete task associations in TaskSolutionAssociation for each solution
        for solution in solutions_to_delete:
            # This cascades deletion of TaskSolutionAssociation entries
            session.delete(solution)

        # Commit the transaction
        session.commit()

        logger.info(f"Deleted solutions with IDs {solution_ids} and their task associations for problem ID {problem_id}.")
        return jsonify({'message': 'Solutions and their task associations deleted successfully.'}), 200

    except SQLAlchemyError as e:
        session.rollback()
        logger.error(f"Database error while deleting solutions: {e}")
        return jsonify({'error': 'An error occurred while deleting solutions.'}), 500
    except Exception as e:
        session.rollback()
        logger.error(f"Unexpected error while deleting solutions: {e}")
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

@pst_troubleshooting_solution_bp.route('/get_tasks/<int:solution_id>', methods=['GET'])
def get_tasks(solution_id):
    """
    Retrieve tasks associated with a specific solution.
    """
    session = db_config.get_main_session()
    try:
        # Query the TaskSolutionAssociation table to find tasks linked to the solution
        task_associations = session.query(TaskSolutionAssociation).filter_by(solution_id=solution_id).all()
        task_ids = [assoc.task_id for assoc in task_associations]

        # Retrieve tasks with the collected task IDs
        tasks = session.query(Task).filter(Task.id.in_(task_ids)).all()

        # Prepare task data for the response
        tasks_data = [{'id': task.id, 'name': task.name, 'description': task.description} for task in tasks]

        return jsonify({'tasks': tasks_data}), 200
    except SQLAlchemyError as e:
        logger.error(f"Database error fetching tasks: {e}")
        return jsonify({'error': 'An error occurred while fetching tasks.'}), 500
    finally:
        session.close()

@pst_troubleshooting_solution_bp.route('/pst_troubleshooting_solution/add_task/', methods=['POST'])
def add_task():
    session = db_config.get_main_session()
    try:
        # Extract data from the request
        data = request.get_json()
        solution_id = data.get('solution_id')
        task_name = data.get('name')
        task_description = data.get('description')

        # Check if required fields are provided
        if not solution_id or not task_name:
            return jsonify({"error": "Solution ID and task name are required."}), 400

        # Create a new Task instance
        new_task = Task(
            name=task_name,
            description=task_description
        )

        # Add the new task to the session first
        session.add(new_task)
        session.commit()  # Commit to generate the task ID

        # Create an association entry in TaskSolutionAssociation
        new_task_solution_association = TaskSolutionAssociation(
            task_id=new_task.id,
            solution_id=solution_id
        )
        session.add(new_task_solution_association)
        session.commit()  # Commit to save the association

        # Return a success message
        return jsonify({"status": "success", "message": "Task added successfully"}), 200

    except Exception as e:
        # Handle any exceptions that may occur
        session.rollback()  # Rollback in case of error
        logger.error(f"Error adding task: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        session.close()
