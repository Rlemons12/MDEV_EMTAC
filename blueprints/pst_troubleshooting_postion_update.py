
@pst_troubleshooting_position_update_bp.route('/get_problem_details/<int:problem_id>', methods=['GET'])
def get_problem_details(problem_id):
    """
    Fetch the details of a problem along with its associated position information.
    """
    session = db_config.get_main_session()
    try:
        # Fetch the problem with the specified ID
        problem = session.query(Problem).filter_by(id=problem_id).first()

        if not problem:
            logger.error(f"Problem with ID {problem_id} not found.")
            return jsonify({'error': 'Problem not found.'}), 404

        # Find the associated position using ProblemPositionAssociation
        association = session.query(ProblemPositionAssociation).filter_by(problem_id=problem_id).first()

        if not association:
            logger.error(f"No position associated with problem ID {problem_id}.")
            return jsonify({'error': 'Position not found for this problem.'}), 404

        # Fetch the position details
        position = session.query(Position).filter_by(id=association.position_id).first()

        # Prepare the JSON response with both problem and position details
        response_data = {
            'problem': {
                'id': problem.id,
                'name': problem.name,
                'description': problem.description
            },
            'position': {
                'area_id': position.area.id if position.area else None,
                'equipment_group_id': position.equipment_group.id if position.equipment_group else None,
                'model_id': position.model.id if position.model else None,
                'asset_number': position.asset_number.number if position.asset_number else None,
                'location': position.location.name if position.location else None,
                'site_location_id': position.site_location.id if position.site_location else None
            }
        }

        logger.info(f"Fetched details for problem ID {problem_id} with associated position.")
        return jsonify(response_data), 200

    except SQLAlchemyError as e:
        logger.error(f"Database error while fetching problem details: {e}")
        return jsonify({'error': 'An error occurred while fetching problem details.'}), 500
    except Exception as e:
        logger.error(f"Unexpected error while fetching problem details: {e}")
        return jsonify({'error': 'An unexpected error occurred.'}), 500
    finally:
        session.close()