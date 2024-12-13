from flask import Blueprint, render_template, request, redirect, url_for, flash
from modules.emtacdb.emtacdb_fts import Part  # Assuming you have a session manager
from sqlalchemy.exc import IntegrityError
from modules.configuration.config_env import DatabaseConfig

# Blueprint setup
update_part_bp = Blueprint('update_part_bp', __name__)

# Route: Edit Part
@update_part_bp.route('/edit_part/<int:part_id>', methods=['GET', 'POST'])
def edit_part(part_id):
    db_session = DatabaseConfig().get_main_session()

    # Fetch the part based on part_id
    part = db_session.query(Part).filter_by(id=part_id).first()

    if not part:
        flash("Part not found.", "error")
        return redirect(url_for('update_part_bp.search_part'))  # Updated to use a valid route

    if request.method == 'POST':
        # Update part attributes from form input
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
            db_session.commit()
            flash("Part updated successfully!", "success")
            return redirect(url_for('update_part_bp.search_part'))  # Updated to use the search_part route
        except IntegrityError:
            db_session.rollback()
            flash("Part number must be unique.", "error")

    return render_template('edit_part.html', part=part)


# Route: Search Part
@update_part_bp.route('/search_part', methods=['GET'])
def search_part():
    db_session = DatabaseConfig().get_main_session()
    search_query = request.args.get('search_query')

    part = None
    if search_query:
        part = db_session.query(Part).filter(
            Part.part_number.ilike(f'%{search_query}%') |
            Part.name.ilike(f'%{search_query}%') |
            Part.oem_mfg.ilike(f'%{search_query}%') |
            Part.model.ilike(f'%{search_query}%')
        ).first()

    db_session.close()

    return render_template('edit_part.html', part=part)
