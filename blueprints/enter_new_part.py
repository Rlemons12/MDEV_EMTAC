from flask import Blueprint, jsonify
from sqlalchemy.orm import sessionmaker
from config_env import DatabaseConfig
from emtacdb_fts import Part, Model

# Initialize the database configuration
db_config = DatabaseConfig()

enter_new_part_bp = Blueprint('enter_new_part_bp', __name__)

@enter_new_part_bp.route('/get_part_form_data', methods=['GET'])
def get_part_form_data():
    session = db_config.get_main_session()

    try:
        # Fetch models from the database
        models = session.query(Model).all()

        data = {
            'models': [{'id': model.id, 'name': model.name} for model in models],
        }

        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)})
    finally:
        session.close()

@enter_new_part_bp.route('/enter_part', methods=['POST'])
def enter_part():
    session = db_config.get_main_session()

    try:
        # Fetch form data
        part_number = request.form['part_number']
        name = request.form['name']
        oem_mfg = request.form['oem_mfg']
        model = request.form['model']
        class_flag = request.form['class_flag']
        ud6 = request.form['ud6']
        type = request.form['type']
        notes = request.form['notes']
        documentation = request.form['documentation']

        # Create a new Part entry
        new_part = Part(
            part_number=part_number,
            name=name,
            oem_mfg=oem_mfg,
            model=model,
            class_flag=class_flag,
            ud6=ud6,
            type=type,
            notes=notes,
            documentation=documentation
        )

        session.add(new_part)
        session.commit()

        flash('Part successfully entered!')
        return redirect(url_for('part_bp.enter_part'))
    except Exception as e:
        session.rollback()
        flash(f'Error entering part: {str(e)}', 'error')
        return redirect(url_for('part_bp.enter_part'))
    finally:
        session.close()
