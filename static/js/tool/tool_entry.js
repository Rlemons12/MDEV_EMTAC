// tool_entry.js
document.addEventListener('DOMContentLoaded', () => {
    // Elements for dropdowns
    const categorySelect = document.getElementById('tool_category');
    const manufacturerSelect = document.getElementById('tool_manufacturer');
    const positionSelect = document.getElementById('tool_position');
    const packageSelect = document.getElementById('tool_package');

    // Populate dropdowns
    populateDropdown('/tool/get_tool_categories', categorySelect, 'Select Category');
    populateDropdown('/tool/get_tool_manufacturers', manufacturerSelect, 'Select Manufacturer');
    populateDropdown('/get_tool_positions', positionSelect, 'Select Position');
    populateDropdown('/tool/get_tool_packages', packageSelect, 'Select Package');

    // Add event listeners to the accordion
    setupAccordionEvents();
});

// Function to populate dropdowns dynamically
function populateDropdown(url, selectElement, defaultOptionText) {
    fetch(url)
        .then(response => response.json())
        .then(data => {
            selectElement.innerHTML = ''; // Clear existing options

            // Add default option
            const defaultOption = document.createElement('option');
            defaultOption.value = '';
            defaultOption.textContent = defaultOptionText;
            selectElement.appendChild(defaultOption);

            // Determine data key
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

            // Add options recursively for subcategories
            items.forEach(item => addOption(selectElement, item));
        })
        .catch(error => {
            console.error('Error fetching data:', error);
        });
}

// Recursive function to add options
function addOption(selectElement, item, depth = 0) {
    const option = document.createElement('option');
    option.value = item.id;
    option.textContent = `${'--'.repeat(depth)} ${item.name}`;
    selectElement.appendChild(option);

    if (item.subcategories && item.subcategories.length > 0) {
        item.subcategories.forEach(sub => addOption(selectElement, sub, depth + 1));
    }
}

// Function to handle Bootstrap Accordion events
function setupAccordionEvents() {
    const accordion = document.getElementById('toolFormAccordion');

    // Triggered when an accordion section expands
    accordion.addEventListener('shown.bs.collapse', (event) => {
        console.log(`Section Expanded: ${event.target.id}`);
    });

    // Triggered when an accordion section collapses
    accordion.addEventListener('hidden.bs.collapse', (event) => {
        console.log(`Section Collapsed: ${event.target.id}`);
    });
}
