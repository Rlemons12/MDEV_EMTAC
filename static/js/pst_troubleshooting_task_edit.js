document.addEventListener('DOMContentLoaded', () => {
    'use strict';

    // Initialize arrays to hold selected items for images, parts, and drawings
    const selectedImages = [];
    const selectedParts = [];
    const selectedDrawings = [];

    /**
     * Function to update the display box for selected items
     */
    function updateSelectedDisplay(displayBoxId, items, itemType) {
    const displayBox = document.getElementById(displayBoxId);
    displayBox.innerHTML = ''; // Clear existing items

    items.forEach(item => {
        const itemDiv = document.createElement('div');
        itemDiv.classList.add('selected-item', 'd-flex', 'align-items-center', 'mb-1');

        // Display different properties based on item type
        if (itemType === 'part') {
            itemDiv.textContent = `${item.part_number || 'N/A'} - ${item.name || 'N/A'}`;
        } else if (itemType === 'drawing') {
            itemDiv.textContent = `${item.drw_number || 'N/A'} - ${item.drw_name || 'N/A'}`;
        } else if (itemType === 'image') {
            itemDiv.textContent = `${item.title || 'N/A'} - ${item.description || 'N/A'}`;
        }

        const removeButton = document.createElement('button');
        removeButton.classList.add('btn', 'btn-sm', 'btn-outline-danger', 'ms-2');
        removeButton.textContent = 'Remove';
        removeButton.onclick = () => {
            const index = items.findIndex(selectedItem => selectedItem.id === item.id);
            if (index > -1) items.splice(index, 1);
            updateSelectedDisplay(displayBoxId, items, itemType);
        };

        itemDiv.appendChild(removeButton);
        displayBox.appendChild(itemDiv);
    });
}


    /**
     * Populate dropdowns dynamically based on selected Area and Equipment Group
     */
    document.getElementById('pst_task_edit_areaDropdown').addEventListener('change', event => {
        const areaId = event.target.value;
        fetch(`/get_equipment_groups?area_id=${areaId}`)
            .then(response => response.json())
            .then(data => populateDropdown('pst_task_edit_equipmentGroupDropdown', data, 'Select Equipment Group'))
            .catch(error => console.error('Error loading equipment groups:', error));
    });

    document.getElementById('pst_task_edit_equipmentGroupDropdown').addEventListener('change', event => {
        const equipmentGroupId = event.target.value;
        fetch(`/get_models?equipment_group_id=${equipmentGroupId}`)
            .then(response => response.json())
            .then(data => populateDropdown('pst_task_edit_modelDropdown', data, 'Select Model'))
            .catch(error => console.error('Error loading models:', error));
    });

    document.getElementById('pst_task_edit_modelDropdown').addEventListener('change', event => {
        const modelId = event.target.value;
        fetch(`/get_asset_numbers?model_id=${modelId}`)
            .then(response => response.json())
            .then(data => populateDropdown('pst_task_edit_assetNumberInput', data, 'Select Asset Number'))
            .catch(error => console.error('Error loading asset numbers:', error));

        fetch(`/get_locations?model_id=${modelId}`)
            .then(response => response.json())
            .then(data => populateDropdown('pst_task_edit_locationInput', data, 'Select Location'))
            .catch(error => console.error('Error loading locations:', error));
    });

    document.getElementById('pst_task_edit_locationInput').addEventListener('change', () => {
        const modelId = document.getElementById('pst_task_edit_modelDropdown').value;
        const assetNumberId = document.getElementById('pst_task_edit_assetNumberInput').value;
        const locationId = document.getElementById('pst_task_edit_locationInput').value;

        fetch(`/get_site_locations?model_id=${modelId}&asset_number_id=${assetNumberId}&location_id=${locationId}`)
            .then(response => response.json())
            .then(data => populateDropdown('pst_task_edit_siteLocationDropdown', data, 'Select Site Location'))
            .catch(error => console.error('Error loading site locations:', error));
    });

    /**
     * Utility function to populate dropdowns with data
     */
    function populateDropdown(dropdownId, data, placeholder) {
        const dropdown = document.getElementById(dropdownId);
        dropdown.innerHTML = `<option value="">${placeholder}</option>`;
        data.forEach(item => {
            const option = document.createElement('option');
            option.value = item.id;
            option.text = item.name || `${item.title} - Room ${item.room_number}`;
            dropdown.appendChild(option);
        });
        dropdown.disabled = data.length === 0;
    }

    /**
     * Event handlers for selecting images, parts, and drawings
     */
    function handleSelection(selectElementId, selectedArray, displayBoxId, itemType) {
        document.getElementById(selectElementId).addEventListener('change', event => {
            const selectedOptions = Array.from(event.target.selectedOptions);
            selectedOptions.forEach(option => {
                const itemExists = selectedArray.some(item => item.id === option.value);
                if (!itemExists) {
                    // Fetch full item details if necessary
                    const newItem = { id: option.value, name: option.text };
                    selectedArray.push(newItem);
                }
            });
            updateSelectedDisplay(displayBoxId, selectedArray, itemType);
        });
    }

    handleSelection('pst_task_edit_task_images', selectedImages, 'pst_task_edit_selected_images', 'image');
    handleSelection('pst_task_edit_task_parts', selectedParts, 'pst_task_edit_selected_parts', 'part');
    handleSelection('pst_task_edit_task_drawings', selectedDrawings, 'pst_task_edit_selected_drawings', 'drawing');

    /**
     * Clear all input fields and selected items in the form
     */
    function clearEditTaskForm() {
        document.getElementById('pst_task_edit_task_name').value = '';
        document.getElementById('pst_task_edit_task_description').value = '';
        document.getElementById('pst_task_edit_areaDropdown').selectedIndex = 0;
        document.getElementById('pst_task_edit_equipmentGroupDropdown').innerHTML = '<option value="">Select Equipment Group</option>';
        document.getElementById('pst_task_edit_modelDropdown').innerHTML = '<option value="">Select Model</option>';
        document.getElementById('pst_task_edit_assetNumberInput').value = '';
        document.getElementById('pst_task_edit_locationInput').value = '';
        document.getElementById('pst_task_edit_siteLocationDropdown').selectedIndex = 0;

        selectedImages.length = 0;
        selectedParts.length = 0;
        selectedDrawings.length = 0;

        updateSelectedDisplay('pst_task_edit_selected_images', selectedImages, 'image');
        updateSelectedDisplay('pst_task_edit_selected_parts', selectedParts, 'part');
        updateSelectedDisplay('pst_task_edit_selected_drawings', selectedDrawings, 'drawing');
    }

    /**
     * Fetch and populate existing task data into the edit form
     */
    function loadTaskDetails(taskId) {
        fetch(`/pst_troubleshooting_task/get_task_details/${taskId}`)
            .then(response => response.ok ? response.json() : Promise.reject(response.json()))
            .then(data => {
                if (data.task) {
                    document.getElementById('pst_task_edit_task_name').value = data.task.name || '';
                    document.getElementById('pst_task_edit_task_description').value = data.task.description || '';
                    document.getElementById('pst_task_edit_areaDropdown').value = data.task.area_id || '';
                    document.getElementById('pst_task_edit_equipmentGroupDropdown').value = data.task.equipment_group_id || '';
                    document.getElementById('pst_task_edit_modelDropdown').value = data.task.model_id || '';
                    document.getElementById('pst_task_edit_assetNumberInput').value = data.task.asset_number || '';
                    document.getElementById('pst_task_edit_locationInput').value = data.task.location || '';
                    document.getElementById('pst_task_edit_siteLocationDropdown').value = data.task.site_location_id || '';

                    // Populate selected items with logging for each association type
                    console.log("Populating images:", data.task.associations.images);
                    data.task.associations.images.forEach(image => {
                        console.log("Adding image:", image); // Log each image item
                        selectedImages.push(image);
                    });

                    console.log("Populating parts:", data.task.associations.parts);
                    data.task.associations.parts.forEach(part => {
                        console.log("Adding part:", part); // Log each part item
                        selectedParts.push(part);
                    });

                    console.log("Populating drawings:", data.task.associations.drawings);
                    data.task.associations.drawings.forEach(drawing => {
                        console.log("Adding drawing:", drawing); // Log each drawing item
                        selectedDrawings.push(drawing);
                    });

                    // Check final arrays after population
                    console.log("Final selected images:", selectedImages);
                    console.log("Final selected parts:", selectedParts);
                    console.log("Final selected drawings:", selectedDrawings);


                    updateSelectedDisplay('pst_task_edit_selected_images', selectedImages, 'image');
                    updateSelectedDisplay('pst_task_edit_selected_parts', selectedParts, 'part');
                    updateSelectedDisplay('pst_task_edit_selected_drawings', selectedDrawings, 'drawing');
                } else {
                    showAlert('Task details not found', 'warning');
                }
            })
            .catch(error => {
                console.error('Error loading task details:', error);
                showAlert('Error loading task details: ' + (error.message || 'Unknown error'), 'danger');
            });
    }

    /**
     * Display an alert message
     */
    function showAlert(message, category) {
        const alertContainer = document.getElementById('alertContainer');
        if (alertContainer) {
            const alertDiv = document.createElement('div');
            alertDiv.className = `alert alert-${category} alert-dismissible fade show`;
            alertDiv.innerHTML = `
                ${message}
                <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
            `;
            alertContainer.appendChild(alertDiv);

            setTimeout(() => {
                alertDiv.remove();
            }, 5000);
        }
    }

    // Example usage: loadTaskDetails(1);
});
