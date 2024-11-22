// pst_troubleshooting_task_edit.js
document.addEventListener('DOMContentLoaded', () => {
    'use strict';

    // === 1. Define Backend Endpoint URLs ===
    const ENDPOINTS = {
        tasks: {
            details: '/pst_troubleshooting_guide_edit_update/task_details/',
            update: '/pst_troubleshooting_guide_edit_update/update_task',
            savePosition: '/pst_troubleshooting_guide_edit_update/save_position',
            searchDocuments: '/pst_troubleshooting_guide_edit_update/search_documents',
            saveDocuments: '/pst_troubleshooting_guide_edit_update/save_task_documents',
            searchDrawings: '/pst_troubleshooting_guide_edit_update/search_drawings',
            saveDrawings: '/pst_troubleshooting_guide_edit_update/save_task_drawings',
            searchParts: '/pst_troubleshooting_guide_edit_update/search_parts', // New endpoint for part search
            saveParts: '/pst_troubleshooting_guide_edit_update/save_task_parts', // Placeholder for saving parts
            searchImages: '/pst_troubleshooting_guide_edit_update/search_images', // Add this line
            saveImages: '/pst_troubleshooting_guide_edit_update/save_task_images',
            removePosition: '/pst_troubleshooting_guide_edit_update/remove_position' // Corrected line
        }
    };

    // === 2. Initialize Global State Object ===
    window.AppState = window.AppState || {};
    window.AppState.currentTaskId = null;
    window.AppState.currentSolutionId = null;

    // === 3. Event Delegation for Save and Remove Position Buttons ===
const positionsContainer = document.getElementById('pst_task_edit_positions_container');
if (positionsContainer) {
    positionsContainer.addEventListener('click', (event) => {
        if (event.target && event.target.matches('.savePositionBtn')) {
            const positionSection = event.target.closest('.position-section');
            const index = Array.from(positionsContainer.children).indexOf(positionSection);
            console.log(`Delegated Save Position button clicked for index ${index}`);
            savePosition(positionSection, index);
        }

        if (event.target && event.target.matches('.removePositionBtn')) {
            const positionSection = event.target.closest('.position-section');
            const index = Array.from(positionsContainer.children).indexOf(positionSection);
            console.log("Delegated Remove Position button clicked for index", index);
            handleRemovePosition(positionSection, index);
        }
    });
    console.log("Attached delegated event listener to positions container.");
} else {
    console.warn("Positions container with ID 'pst_task_edit_positions_container' not found.");
}



    // === 3. Initialize Select2 for Document Search ===
// Initialize Select2 for Document Search with empty placeholder for selected items
$('#pst_task_edit_task_documents').select2({
    placeholder: 'Select or search for documents',
    allowClear: true,
    ajax: {
        url: ENDPOINTS.tasks.searchDocuments, // Ensure this is defined in ENDPOINTS
        dataType: 'json',
        delay: 250,
        data: params => ({ query: params.term }),
        processResults: data => ({
            results: data.map(doc => ({
                id: doc.id,
                text: doc.text // Ensure backend provides `id` and `text` fields
            }))
        }),
        cache: true
    }
});

// Function to update custom container with selected documents
function updateSelectedDocumentsDisplay() {
    const selectedDocumentIds = $('#pst_task_edit_task_documents').val();
    const selectedDocumentsContainer = $('#pst_task_edit_selected_documents');

    // Clear the container first
    selectedDocumentsContainer.empty();

    // Fetch selected options and add them to the custom display container
    selectedDocumentIds.forEach(id => {
        const documentText = $('#pst_task_edit_task_documents option[value="' + id + '"]').text();
        const documentDiv = $('<div class="selected-item"></div>')
            .text(documentText)
            .append(
                $('<button type="button" class="btn btn-sm btn-danger ms-2">Remove</button>')
                    .on('click', function () {
                        // Remove the item from select2 and update the display
                        const updatedSelection = $('#pst_task_edit_task_documents').val().filter(val => val !== id);
                        $('#pst_task_edit_task_documents').val(updatedSelection).trigger('change');
                        updateSelectedDocumentsDisplay();
                    })
            );
        selectedDocumentsContainer.append(documentDiv);
    });
}

// Listen for change events on the select2 element
$('#pst_task_edit_task_documents').on('change', updateSelectedDocumentsDisplay);

// === 4. Save Selected Documents ===
async function saveSelectedDocuments() {
    const selectedDocumentIds = $('#pst_task_edit_task_documents').val();
    const taskId = window.AppState.currentTaskId;

    if (!taskId) {
        SolutionTaskCommon.showAlert('No task selected to save documents.', 'warning');
        return;
    }

    const payload = {
        task_id: taskId,
        document_ids: selectedDocumentIds
    };

    try {
        const response = await fetch(ENDPOINTS.tasks.saveDocuments, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        });

        const data = await response.json();
        if (data.status === 'success') {
            SolutionTaskCommon.showAlert('Documents saved successfully.', 'success');
        } else {
            SolutionTaskCommon.showAlert(data.message || 'Failed to save documents.', 'danger');
        }
    } catch (error) {
        console.error('Error saving documents:', error);
        SolutionTaskCommon.showAlert('An error occurred while saving documents.', 'danger');
    }
}

const saveDocumentsBtn = document.getElementById('saveDocumentsBtn');
if (saveDocumentsBtn) {
    saveDocumentsBtn.addEventListener('click', saveSelectedDocuments);
} else {
    console.warn("Save Documents button with ID 'saveDocumentsBtn' not found.");
}
/**
    // === 5. Position Handling Functions ===
    async function addPosition(positionData = null, index = 0) {
        const container = document.getElementById('pst_task_edit_positions_container');
        const template = document.getElementById('position-template');
        if (!container || !template) {
            console.warn("Positions container or template not found.");
            return;
        }

        const clone = template.content.cloneNode(true);
        const positionSection = clone.querySelector('.position-section');

        // Assign unique IDs based on index for all form elements
        const elementsToId = ['areaDropdown', 'equipmentGroupDropdown', 'modelDropdown', 'assetNumberInput', 'locationInput', 'siteLocationDropdown'];
        elementsToId.forEach(elementClass => {
            const element = positionSection.querySelector(`.${elementClass}`);
            if (element) {
                element.id = `${elementClass}_${index}`;
                console.log(`Assigned ID ${element.id} to ${elementClass}`);
            }
        });

        if (positionData) {
            await populatePositionFields(positionSection, positionData, index);
        } else {
            await initializeNewPosition(positionSection, index);
        }
        container.appendChild(clone);
        console.log(`Added new position section with index ${index}`);
    }

    async function populatePositionFields(positionSection, positionData, index) {
        const areaDropdown = positionSection.querySelector('.areaDropdown');
        if (areaDropdown) {
            const areas = await SolutionTaskCommon.fetchInitialAreas();
            window.SolutionTaskCommon.populateDropdown(areaDropdown, areas, 'Select Area');
            areaDropdown.value = positionData.area_id || '';
            areaDropdown.disabled = false;
            console.log(`Populated Area Dropdown with ID ${positionData.area_id}`);
        }

        const equipmentGroupDropdown = positionSection.querySelector('.equipmentGroupDropdown');
        if (equipmentGroupDropdown && positionData.area_id) {
            const equipmentGroups = await SolutionTaskCommon.fetchInitialEquipmentGroups(positionData.area_id);
            SolutionTaskCommon.populateDropdown(equipmentGroupDropdown, equipmentGroups, 'Select Equipment Group');
            equipmentGroupDropdown.value = positionData.equipment_group_id || '';
            equipmentGroupDropdown.disabled = false;
            console.log(`Populated Equipment Group Dropdown with ID ${positionData.equipment_group_id}`);
        }

        const modelDropdown = positionSection.querySelector('.modelDropdown');
        if (modelDropdown && positionData.equipment_group_id) {
            const models = await SolutionTaskCommon.fetchInitialModels(positionData.equipment_group_id);
            SolutionTaskCommon.populateDropdown(modelDropdown, models, 'Select Model');
            modelDropdown.value = positionData.model_id || '';
            modelDropdown.disabled = false;
            console.log(`Populated Model Dropdown with ID ${positionData.model_id}`);
        }

        const assetNumberInput = positionSection.querySelector('.assetNumberInput');
        if (assetNumberInput) {
            assetNumberInput.value = positionData.asset_number || '';
            assetNumberInput.disabled = false;
            console.log(`Set Asset Number to: ${positionData.asset_number}`);
        }

        const locationInput = positionSection.querySelector('.locationInput');
        if (locationInput) {
            locationInput.value = positionData.location || '';
            locationInput.disabled = false;
            console.log(`Set Location to: ${positionData.location}`);
        }

        const siteLocationDropdown = positionSection.querySelector('.siteLocationDropdown');
        if (siteLocationDropdown) {
            const siteLocations = await SolutionTaskCommon.fetchInitialSiteLocations();
            SolutionTaskCommon.populateDropdown(siteLocationDropdown, siteLocations, 'Select Site Location');
            siteLocationDropdown.value = positionData.site_location_id || '';
            siteLocationDropdown.disabled = false;
            console.log(`Populated Site Location Dropdown independently with ID ${positionData.site_location_id}`);
        }
    }

    async function initializeNewPosition(positionSection, index) {
        const areaDropdown = positionSection.querySelector('.areaDropdown');
        if (areaDropdown) {
            const areas = await SolutionTaskCommon.fetchInitialAreas();
            SolutionTaskCommon.populateDropdown(areaDropdown, areas, 'Select Area');
            areaDropdown.disabled = false;
            console.log("Initialized Area Dropdown for new position.");
        }

        const siteLocationDropdown = positionSection.querySelector('.siteLocationDropdown');
        if (siteLocationDropdown) {
            const siteLocations = await SolutionTaskCommon.fetchInitialSiteLocations();
            SolutionTaskCommon.populateDropdown(siteLocationDropdown, siteLocations, 'Select Site Location');
            siteLocationDropdown.disabled = false;
            console.log("Initialized Site Location Dropdown independently.");
        }

        ['equipmentGroupDropdown', 'modelDropdown', 'assetNumberInput', 'locationInput'].forEach(className => {
            const element = positionSection.querySelector(`.${className}`);
            if (element) {
                if (element.tagName.toLowerCase() === 'select') {
                    SolutionTaskCommon.populateDropdown(element, [], `Select ${SolutionTaskCommon.capitalizeFirstLetter(className.replace('Dropdown', '').replace('Input', ''))}`);
                } else if (element.tagName.toLowerCase() === 'input') {
                    element.value = '';
                }
                element.disabled = true;
                console.log(`Initialized and disabled ${className} (ID: ${element.id})`);
            }
        });
        console.log("Initialized new position section with default states.");
    }
*/
async function savePosition(positionSection, index) {
    const areaDropdown = positionSection.querySelector('.areaDropdown');
    const equipmentGroupDropdown = positionSection.querySelector('.equipmentGroupDropdown');
    const modelDropdown = positionSection.querySelector('.modelDropdown');
    const assetNumberInput = positionSection.querySelector('.assetNumberInput');
    const locationInput = positionSection.querySelector('.locationInput');
    const siteLocationDropdown = positionSection.querySelector('.siteLocationDropdown');

    if (!areaDropdown.value || !equipmentGroupDropdown.value || !modelDropdown.value) {
        SolutionTaskCommon.showAlert('Please fill in all required fields before saving.', 'warning');
        return;
    }

    if (!window.AppState.currentTaskId || !window.AppState.currentSolutionId) {
        SolutionTaskCommon.showAlert('Task and Solution must be selected before saving a position.', 'warning');
        return;
    }

    const positionData = {
        area_id: parseInt(areaDropdown.value, 10),
        equipment_group_id: parseInt(equipmentGroupDropdown.value, 10),
        model_id: parseInt(modelDropdown.value, 10),
        asset_number_id: assetNumberInput.value.trim() || null, // Adjusted key to match backend
        location_id: locationInput.value.trim() || null,       // Adjusted key to match backend
        site_location_id: parseInt(siteLocationDropdown.value, 10) || null
    };

    const saveBtn = positionSection.querySelector('.savePositionBtn');
    let originalBtnText = '';
    if (saveBtn) {
        saveBtn.disabled = true;
        originalBtnText = saveBtn.textContent;
        saveBtn.textContent = 'Saving...';
    }

    try {
        const response = await fetch(ENDPOINTS.tasks.savePosition, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                task_id: window.AppState.currentTaskId,
                solution_id: window.AppState.currentSolutionId,
                position_data: positionData
            })
        });

        const data = await response.json();

        if (response.ok && data.status === 'success') {
            // Update the data-position-id attribute with the new position_id
            const newPositionId = data.position_id;
            positionSection.setAttribute('data-position-id', newPositionId);

            SolutionTaskCommon.showAlert('Position saved successfully.', 'success');
            console.log(`Position saved with ID: ${newPositionId}`);
        } else {
            SolutionTaskCommon.showAlert(data.error || 'Failed to save position.', 'danger');
            console.error('Failed to save position:', data.error || data.message);
        }
    } catch (error) {
        console.error('Error saving position:', error);
        SolutionTaskCommon.showAlert('An error occurred while saving the position.', 'danger');
    } finally {
        if (saveBtn) {
            saveBtn.disabled = false;
            saveBtn.textContent = originalBtnText;
        }
    }
}

    // === 6. Task Handling Functions ===
    async function saveTaskDetails() {
        const taskId = window.AppState.currentTaskId;
        const taskNameInput = document.getElementById('pst_task_edit_task_name');
        const taskDescriptionTextarea = document.getElementById('pst_task_edit_task_description');
        const positionsData = collectPositionsData();

        const updatedTaskData = {
            task_id: taskId,
            name: taskNameInput.value.trim(),
            description: taskDescriptionTextarea.value.trim(),
            positions: positionsData
        };

        const saveTaskBtn = document.getElementById('saveTaskBtn');
        if (saveTaskBtn) {
            saveTaskBtn.disabled = true;
            saveTaskBtn.textContent = 'Saving...';
        }

        const response = await fetchWithHandling(ENDPOINTS.tasks.update, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(updatedTaskData)
        });

        if (response.status === 'success') {
            SolutionTaskCommon.showAlert('Task updated successfully.', 'success');
        } else {
            SolutionTaskCommon.showAlert('Failed to update task.', 'danger');
        }

        if (saveTaskBtn) {
            saveTaskBtn.disabled = false;
            saveTaskBtn.textContent = 'Save Task';
        }
    }

    function collectPositionsData() {
        const positionsContainer = document.getElementById('pst_task_edit_positions_container');
        const positionsData = [];
        positionsContainer.querySelectorAll('.position-section').forEach(section => {
            positionsData.push({
                area_id: parseInt(section.querySelector('.areaDropdown').value, 10),
                equipment_group_id: parseInt(section.querySelector('.equipmentGroupDropdown').value, 10),
                model_id: parseInt(section.querySelector('.modelDropdown').value, 10),
                asset_number: section.querySelector('.assetNumberInput').value.trim() || null,
                location: section.querySelector('.locationInput').value.trim() || null,
                site_location_id: parseInt(section.querySelector('.siteLocationDropdown').value, 10) || null
            });
        });
        return positionsData;
    }

    async function fetchWithHandling(url, options = {}) {
        try {
            const response = await fetch(url, options);
            if (!response.ok) throw new Error('Network response was not ok');
            return await response.json();
        } catch (error) {
            console.error('Fetch error:', error);
            throw error;
        }
    }

    // === 7. Initialize Event Listeners ===
function initializeEventListeners() {
    const saveTaskBtn = document.getElementById('saveTaskBtn');
    if (saveTaskBtn) {
        saveTaskBtn.addEventListener('click', saveTaskDetails);
    }

    // Add event listener for Update Task Details button
    const updateTaskDetailsBtn = document.getElementById('updateTaskDetailsBtn');
    if (updateTaskDetailsBtn) {
        updateTaskDetailsBtn.addEventListener('click', updateTaskDetails);
    } else {
        console.warn("Update Task Details button with ID 'updateTaskDetailsBtn' not found.");
    }

    async function updateTaskDetails() {
    const taskId = window.AppState.currentTaskId;
    const taskNameInput = document.getElementById('pst_task_edit_task_name');
    const taskDescriptionTextarea = document.getElementById('pst_task_edit_task_description');
    const taskName = taskNameInput.value.trim();
    const taskDescription = taskDescriptionTextarea.value.trim();

    if (!taskId) {
        SolutionTaskCommon.showAlert('No task selected to update.', 'warning');
        return;
    }

    if (!taskName) {
        SolutionTaskCommon.showAlert('Task name cannot be empty.', 'warning');
        return;
    }

    const payload = {
        task_id: taskId,
        name: taskName,
        description: taskDescription
    };

    const updateBtn = document.getElementById('updateTaskDetailsBtn');
    let originalBtnText = '';
    if (updateBtn) {
        updateBtn.disabled = true;
        originalBtnText = updateBtn.textContent;
        updateBtn.textContent = 'Updating...';
    }

    try {
        const response = await fetch(ENDPOINTS.tasks.update, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const data = await response.json();

        if (response.ok && data.status === 'success') {
            SolutionTaskCommon.showAlert('Task updated successfully.', 'success');
        } else {
            SolutionTaskCommon.showAlert(data.error || 'Failed to update task.', 'danger');
            console.error('Failed to update task:', data.error || data.message);
        }
    } catch (error) {
        console.error('Error updating task:', error);
        SolutionTaskCommon.showAlert('An error occurred while updating the task.', 'danger');
    } finally {
        if (updateBtn) {
            updateBtn.disabled = false;
            updateBtn.textContent = originalBtnText;
        }
    }
}


/**
        const addPositionBtn = document.getElementById('addPositionBtn');
        if (addPositionBtn) {
            addPositionBtn.addEventListener('click', () => {
                const currentIndex = document.querySelectorAll('.position-section').length;
                addPosition(null, currentIndex);
            });
        }

        */

        // Ensure the "Save Documents" button has the correct event listener
        const saveDocumentsBtn = document.getElementById('saveDocumentsBtn');
        if (saveDocumentsBtn) {
            saveDocumentsBtn.addEventListener('click', saveSelectedDocuments);
        } else {
            console.warn("Save Documents button with ID 'saveDocumentsBtn' not found.");
        }
    }

    // Initialize Select2 for Drawing Search
    $('#pst_task_edit_task_drawings').select2({
        placeholder: 'Select or search for drawings',
        allowClear: true,
        ajax: {
            url: ENDPOINTS.tasks.searchDrawings, // Define this endpoint in the ENDPOINTS object
            dataType: 'json',
            delay: 250,
            data: params => ({ q: params.term }), // Use 'q' as defined in the route
            processResults: data => ({
                results: data // Directly use the array of drawing objects with 'id' and 'text'
            }),
            cache: true
        }
    });

    async function saveSelectedDrawings() {
        const selectedDrawingIds = $('#pst_task_edit_task_drawings').val();
        const taskId = window.AppState.currentTaskId;

        if (!taskId) {
            SolutionTaskCommon.showAlert('No task selected to save drawings.', 'warning');
            return;
        }

        const payload = {
            task_id: taskId,
            drawing_ids: selectedDrawingIds
        };

        try {
            const response = await fetch(ENDPOINTS.tasks.saveDrawings, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(payload)
            });

            const data = await response.json();
            if (data.status === 'success') {
                SolutionTaskCommon.showAlert('Drawings saved successfully.', 'success');
            } else {
                SolutionTaskCommon.showAlert(data.message || 'Failed to save drawings.', 'danger');
            }
        } catch (error) {
            console.error('Error saving drawings:', error);
            SolutionTaskCommon.showAlert('An error occurred while saving drawings.', 'danger');
        }
    }

    // Ensure the "Save Drawings" button has the correct event listener
    const saveDrawingsBtn = document.getElementById('saveDrawingsBtn');
    if (saveDrawingsBtn) {
        saveDrawingsBtn.addEventListener('click', saveSelectedDrawings);
    } else {
        console.warn("Save Drawings button with ID 'saveDrawingsBtn' not found.");
    }


    // Initialize Select2 for Part Search
    $('#pst_task_edit_task_parts').select2({
        placeholder: 'Select or search for parts',
        allowClear: true,
        ajax: {
            url: ENDPOINTS.tasks.searchParts,
            dataType: 'json',
            delay: 250,
            data: params => ({ q: params.term }),
            processResults: data => {
                console.log("Received part data:", data);
                return { results: data };
            },
            cache: true
        }
    });


    async function saveSelectedParts() {
        const selectedPartIds = $('#pst_task_edit_task_parts').val();
        const taskId = window.AppState.currentTaskId;

        if (!taskId) {
            SolutionTaskCommon.showAlert('No task selected to save parts.', 'warning');
            return;
        }

        const payload = {
            task_id: taskId,
            part_ids: selectedPartIds
        };

        try {
            const response = await fetch(ENDPOINTS.tasks.saveParts, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(payload)
            });

            const data = await response.json();
            if (data.status === 'success') {
                SolutionTaskCommon.showAlert('Parts saved successfully.', 'success');
            } else {
                SolutionTaskCommon.showAlert(data.message || 'Failed to save parts.', 'danger');
            }
        } catch (error) {
            console.error('Error saving parts:', error);
            SolutionTaskCommon.showAlert('An error occurred while saving parts.', 'danger');
        }
    }

    const savePartsBtn = document.getElementById('savePartsBtn');
if (savePartsBtn) {
    console.log('Save Parts button found.'); // Confirm button existence
    savePartsBtn.addEventListener('click', saveSelectedParts);
} else {
    console.warn('Save Parts button with ID "savePartsBtn" not found.');
}


    $('#pst_task_edit_task_images').select2({
    placeholder: 'Select or search for images',
    allowClear: true,
    ajax: {
        url: ENDPOINTS.tasks.searchImages, // Ensure this is defined in the ENDPOINTS object
        dataType: 'json',
        delay: 250,
        data: params => ({ q: params.term }), // Use 'q' as defined in the route
        processResults: data => ({
            results: data.map(image => ({
                id: image.id,
                text: `${image.title} - ${image.description}`
            }))
        }),
        cache: true
    }
});

    async function saveSelectedImages() {
        const selectedImageIds = $('#pst_task_edit_task_images').val();
        const taskId = window.AppState.currentTaskId;

        if (!taskId) {
            SolutionTaskCommon.showAlert('No task selected to save images.', 'warning');
            return;
        }

        const payload = {
            task_id: taskId,
            image_ids: selectedImageIds
        };

        try {
            const response = await fetch(ENDPOINTS.tasks.saveImages, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(payload)
            });

            const data = await response.json();
            if (data.status === 'success') {
                SolutionTaskCommon.showAlert('Images saved successfully.', 'success');
            } else {
                SolutionTaskCommon.showAlert(data.message || 'Failed to save images.', 'danger');
            }
        } catch (error) {
            console.error('Error saving images:', error);
            SolutionTaskCommon.showAlert('An error occurred while saving images.', 'danger');
        }
    }

    // Inside your initializeEventListeners function
    const saveImagesBtn = document.getElementById('saveImagesBtn');
    if (saveImagesBtn) {
        saveImagesBtn.addEventListener('click', saveSelectedImages);
    } else {
        console.warn("Save Images button with ID 'saveImagesBtn' not found.");
    }
// === 13. Remove Position Function ===
async function handleRemovePosition(positionSection, index) {
    const positionId = positionSection.getAttribute('data-position-id');
    const removeBtn = positionSection.querySelector('.removePositionBtn');

    // Confirm deletion with the user
    const confirmDeletion = confirm('Are you sure you want to remove this position?');
    if (!confirmDeletion) return;

    let originalBtnText = ''; // Ensure originalBtnText is accessible
    if (removeBtn) {
        // Disable the Remove button to prevent multiple clicks
        removeBtn.disabled = true;
        originalBtnText = removeBtn.textContent;
        removeBtn.textContent = 'Removing...';
    }

    // Check if positionId is a temporary ID
    if (positionId && !positionId.startsWith('temp-')) {
        // The position exists in the backend; proceed to delete from backend
        try {
            // Make a POST request to remove the position
            const response = await fetch(ENDPOINTS.tasks.removePosition, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ position_id: positionId })
            });

            const data = await response.json();

            if (response.ok && data.status === 'success') {
                // Remove the position from the DOM
                positionSection.remove();
                SolutionTaskCommon.showAlert('Position removed successfully.', 'success');
                console.log(`Removed position section with index ${index} and position_id ${positionId}`);
            } else {
                // Handle errors returned by the backend
                SolutionTaskCommon.showAlert(data.error || 'Failed to remove position.', 'danger');
                console.error('Failed to remove position:', data.error || data.message);
                if (removeBtn) {
                    // Re-enable the Remove button and restore original text
                    removeBtn.disabled = false;
                    removeBtn.textContent = originalBtnText;
                }
            }
        } catch (error) {
            // Handle network or unexpected errors
            console.error('Error removing position:', error);
            SolutionTaskCommon.showAlert('An error occurred while removing the position.', 'danger');
            if (removeBtn) {
                // Re-enable the Remove button and restore original text
                removeBtn.disabled = false;
                removeBtn.textContent = originalBtnText;
            }
        }
    } else {
        // If position_id is a temporary ID, remove from DOM without backend call
        positionSection.remove();
        SolutionTaskCommon.showAlert('Position removed successfully.', 'info');
        console.log(`Removed position section with index ${index} without backend association.`);
        if (removeBtn) {
            // Re-enable the Remove button and restore original text
            removeBtn.disabled = false;
            removeBtn.textContent = originalBtnText;
        }
    }
}

    initializeEventListeners();
});
