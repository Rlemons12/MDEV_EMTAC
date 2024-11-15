@app.route('/add_tool', methods=['GET', 'POST'])
def add_tool():
    form = ToolForm()
    form.category_id.choices = [(c.id, c.name) for c in Category.query.all()]
    form.manufacturer_id.choices = [(m.id, m.name) for m in Manufacturer.query.all()]

    if form.validate_on_submit():
        new_tool = Tool(
            name=form.name.data,
            size=form.size.data,
            type=form.type.data,
            material=form.material.data,
            description=form.description.data,
            category_id=form.category_id.data,
            manufacturer_id=form.manufacturer_id.data
        )
        db.session.add(new_tool)
        db.session.commit()
        return redirect(url_for('add_tool'))

    return render_template('add_tool.html', form=form)
