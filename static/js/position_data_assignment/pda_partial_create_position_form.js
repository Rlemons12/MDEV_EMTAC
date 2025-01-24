// static/js/position_data_assignment/pda_partial_create_position_form.js

document.addEventListener('DOMContentLoaded', function () {
    /**
     * Utility Functions
     */

    /**
     * Toggle the visibility of a specific section based on the selected value.
     * @param {HTMLElement} selectElement - The dropdown element.
     * @param {HTMLElement} targetElement - The element to show/hide.
     */
    function toggleSection(selectElement, targetElement) {
        if (selectElement.value === 'new') {
            targetElement.style.display = 'block';
        } else {
            targetElement.style.display = 'none';
            // Clear input fields when hiding
            const inputs = targetElement.querySelectorAll('input, textarea');
            inputs.forEach(input => input.value = '');
        }
    }

    /**
     * Remove a parent element (used for removing dynamically added fields).
     * @param {HTMLElement} button - The remove button that was clicked.
     */
    function removeParentElement(button) {
        const parent = button.closest('.form-group');
        if (parent) {
            parent.remove();
        }
    }

    /**
     * Initialize event listeners for "Add New..." dropdowns.
     */
    function initializeAddNewDropdowns() {
        const addNewDropdowns = document.querySelectorAll('select[id$="Dropdown"]');

        addNewDropdowns.forEach(dropdown => {
            dropdown.addEventListener('change', function () {
                const targetId = dropdown.id.replace('Dropdown', 'Fields');
                const targetElement = document.getElementById(targetId);
                toggleSection(dropdown, targetElement);
            });
        });
    }

    /**
     * Initialize event listeners for dynamically added remove buttons.
     */
    function initializeRemoveButtons(containerId) {
        const container = document.getElementById(containerId);
        container.addEventListener('click', function (e) {
            if (e.target && e.target.matches('.remove-field-btn')) {
                removeParentElement(e.target);
            }
        });
    }

    /**
     * Handle form submission via AJAX.
     * @param {HTMLFormElement} form - The form element to submit.
     */
    function handleFormSubmission(form) {
        form.addEventListener('submit', async function (event) {
            event.preventDefault(); // Prevent default form submission

            const formData = new FormData(form);
            const actionUrl = form.getAttribute('action');

            try {
                // Show loading indicator
                toggleLoadingSpinner(true);

                const response = await fetch(actionUrl, {
                    method: 'POST',
                    body: formData,
                    headers: {
                        'X-CSRFToken': formData.get('csrf_token') // CSRF token for security
                    }
                });

                const result = await response.json();

                if (response.ok) {
                    if (result.success) {
                        toastr.success(result.message || 'Position created successfully!');
                        form.reset(); // Reset the form after success

                        // Hide all new entity input fields
                        const newFields = form.querySelectorAll('[id$="Fields"]');
                        newFields.forEach(field => field.style.display = 'none');

                        // Optionally, switch to the Search Positions tab to view the new entry
                        // Assuming Bootstrap is used for tabs
                        $('#positionTabs a[href="#search-positions"]').tab('show');
                    } else {
                        toastr.error(result.message || 'Failed to create position.');
                    }
                } else {
                    toastr.error(result.message || 'An error occurred while creating the position.');
                }
            } catch (err) {
                console.error('Error submitting form:', err);
                toastr.error('An unexpected error occurred. Please try again.');
            } finally {
                // Hide loading indicator
                toggleLoadingSpinner(false);
            }
        });
    }

    /**
     * Initialize all functionalities.
     */
    function initialize() {
        initializeAddNewDropdowns();

        // Initialize remove buttons for all entity containers
        initializeRemoveButtons('newAreaFields');
        initializeRemoveButtons('newEquipmentGroupFields');
        initializeRemoveButtons('newModelFields');
        initializeRemoveButtons('newAssetNumberFields');
        initializeRemoveButtons('newLocationFields');
        initializeRemoveButtons('newAssemblyFields');
        initializeRemoveButtons('newSubassemblyFields');
        initializeRemoveButtons('newAssemblyViewFields');
        initializeRemoveButtons('newSiteLocationFields');

        // Initialize form submission handler
        const createPositionForm = document.getElementById('positionForm');
        if (createPositionForm) {
            handleFormSubmission(createPositionForm);
        }

        // Initialize event listeners for dynamically added remove buttons (if cloning is implemented)
        // For example, if using cloned templates:
        /*
        const areaFieldsWrapper = document.getElementById('areaFieldsWrapper');
        areaFieldsWrapper.addEventListener('click', function (e) {
            if (e.target && e.target.matches('.remove-field-btn')) {
                removeParentElement(e.target);
            }
        });
        */
    }

    /**
     * Function to toggle loading spinner visibility.
     * Assumes there's an element with ID 'loadingSpinner' in your template.
     * @param {boolean} show - Whether to show or hide the spinner.
     */
    function toggleLoadingSpinner(show) {
        const spinner = document.getElementById('loadingSpinner');
        if (spinner) {
            spinner.style.display = show ? 'block' : 'none';
        }
    }

    // Initialize all components
    initialize();
});
