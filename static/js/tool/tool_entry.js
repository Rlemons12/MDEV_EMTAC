// static/js/tool/tool_entry.js

document.addEventListener('DOMContentLoaded', () => {
    const categorySelect = document.getElementById('tool_category');
    const manufacturerSelect = document.getElementById('tool_manufacturer');
    const positionSelect = document.getElementById('tool_position');
    const packageSelect = document.getElementById('tool_package');

    populateDropdown('/get_tool_categories', categorySelect, 'Select Category');
    populateDropdown('/get_tool_manufacturers', manufacturerSelect, 'Select Manufacturer');
    populateDropdown('/get_tool_positions', positionSelect, 'Select Position');
    populateDropdown('/get_tool_packages', packageSelect, 'Select Package');
});

function populateDropdown(url, selectElement, defaultOptionText) {
    fetch(url)
        .then(response => response.json())
        .then(data => {
            // Clear existing options
            selectElement.innerHTML = '';

            // Add default option
            const defaultOption = document.createElement('option');
            defaultOption.value = '';
            defaultOption.textContent = defaultOptionText;
            selectElement.appendChild(defaultOption);

            // Determine the key to access the data based on URL
            let items = [];
            if (data.categories) {
                items = data.categories;
            } else if (data.manufacturers) {
                items = data.manufacturers;
            } else if (data.positions) {
                items = data.positions;
            } else if (data.packages) {
                items = data.packages;
            }

            // Populate the dropdown
            items.forEach(item => {
                addOption(selectElement, item);
            });
        })
        .catch(error => {
            console.error('Error fetching data:', error);
            // Optionally, handle the error in the UI (e.g., show a message)
        });
}

function addOption(selectElement, item, depth = 0) {
    const option = document.createElement('option');
    option.value = item.id;
    option.textContent = `${'--'.repeat(depth)} ${item.name}`;
    selectElement.appendChild(option);

    if (item.subcategories && item.subcategories.length > 0) {
        item.subcategories.forEach(sub => addOption(selectElement, sub, depth + 1));
    }
}
