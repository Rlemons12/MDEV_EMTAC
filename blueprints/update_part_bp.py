from flask import Blueprint, render_template, request, redirect, url_for, flash
from emtacdb_fts import Part  # Assuming you have a session manager
from sqlalchemy.exc import IntegrityError
from config_env import DatabaseConfig

db_session = DatabaseConfig()

# Blueprint setup
update_part_bp = Blueprint('update_part_bp', __name__)


@update_part_bp.route('/edit_part/<int:part_id>', methods=['GET', 'POST'])
def edit_part(part_id):
    # Get the session
    db_session = db_config.get_main_session()

    # Fetch the part object based on the given part_id
    part = db_session.query(Part).filter_by(id=part_id).first()

    if not part:
        flash("Part not found.", "error")
        return redirect(url_for('your_main_route'))  # Replace with your main route

    if request.method == 'POST':
        # Update the part attributes based on form input
        part.part_number = request.form.get('part_number')
        part.name = request.form.get('name')
        part.oem_mfg = request.form.get('oem_mfg')
        part.model = request.form.get('model')
        part.class_flag = request.form.get('class_flag')
        part.ud6 = request.form.get('ud6')
        part.type = request.form.get('type')
        part.notes = request.form.get('notes')
        part.documentation = request.form.get('documentation')

        try:
            # Commit changes to the database
            db_session.commit()
            flash("Part updated successfully!", "success")
            return redirect(url_for('your_main_route'))  # Replace with your main route
        except IntegrityError:
            db_session.rollback()  # Rollback if there's an error
            flash("Part number must be unique.", "error")

    # Render the form pre-filled with the current part data
    return render_template('edit_part.html', part=part)


@update_part_bp.route('/search_part', methods=['GET'])
def search_part():
    # Get a new session from the DatabaseConfig
    db_session = DatabaseConfig().get_main_session()

    search_query = request.args.get('search_query')
    if search_query:
        parts = db_session.query(Part).filter(
            Part.part_number.ilike(f'%{search_query}%') |
            Part.name.ilike(f'%{search_query}%') |
            Part.oem_mfg.ilike(f'%{search_query}%') |
            Part.model.ilike(f'%{search_query}%')
        ).all()
    else:
        parts = []

    # Make sure to close the session when done
    db_session.close()

    return render_template('edit_part.html', parts=parts)

