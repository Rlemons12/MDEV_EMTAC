// pst_troubleshooting_solution_task_edit.js

document.addEventListener('DOMContentLoaded', () => {
    'use strict';

    // === 1. Define Backend Endpoint URLs ===
    const ENDPOINTS = {
        solutions: {
            get: '/pst_troubleshooting_solution/get_solutions/',
            add: '/pst_troubleshooting_solution/add_solution/',
            remove: '/pst_troubleshooting_solution/remove_solutions/',
        },
        tasks: {
            get: '/pst_troubleshooting_solution/get_tasks/',
            add: '/pst_troubleshooting_solution/add_task/',
            remove: '/pst_troubleshooting_solution/remove_tasks/',
            details: '/pst_troubleshooting_task/get_task_details/',
            update: '/pst_troubleshooting_task/update_task_details/',
        },
        initialData: {
            areas: '/pst_troubleshooting_guide_edit_update/get_areas',
            equipmentGroups: '/pst_troubleshooting_guide_edit_update/get_equipment_groups',
            models: '/pst_troubleshooting_guide_edit_update/get_models',
            assetNumbers: '/pst_troubleshooting_guide_edit_update/get_asset_numbers',
            locations: '/pst_troubleshooting_guide_edit_update/get_locations',
            siteLocations: '/pst_troubleshooting_guide_edit_update/get_site_locations',
        }
    };

    // Placeholder for current problem and solution IDs
    let currentProblemId = null; // If needed, otherwise remove
    let currentSolutionId = null; // Holds the ID of the currently selected solution
    let selectedTaskId = null; // Holds the ID of the currently selected task

    console.log('Current Problem ID:', currentProblemId);
    console.log('Current Solution ID:', currentSolutionId);

    // Set values
    window.currentSolutionId = currentSolutionId;
    window.selectedTaskId = selectedTaskId;

    // === 2. Namespace for Common Functions ===
    window.SolutionTaskCommon = {
        /**
         * Show alert messages to the user
         * @param {string} message - The message to display
         * @param {string} category - Bootstrap alert category (e.g., 'success', 'danger')
         */
        showAlert(message, category) {
            const alertContainer = document.getElementById('alertContainer');
            if (alertContainer) {
                const alertDiv = document.createElement('div');
                alertDiv.className = `alert alert-${category} alert-dismissible fade show`;
                alertDiv.setAttribute('role', 'alert');
                alertDiv.innerHTML = `
                    ${message}
                    <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
                `;
                alertContainer.appendChild(alertDiv);
                // Automatically remove the alert after 5 seconds
                setTimeout(() => {
                    alertDiv.remove();
                }, 5000);
            } else {
                console.warn("Alert container with ID 'alertContainer' not found.");
            }
        },

        /**
         * Populate a dropdown (select element) with data
         * @param {HTMLElement} dropdown - The select element
         * @param {Array} data - The array of data objects to populate
         * @param {string} placeholder - The placeholder text for the first option
         */
        populateDropdown(dropdown, data, placeholder) {
            if (!dropdown) {
                console.warn(`Dropdown element not provided.`);
                return;
            }

            // Clear existing options
            dropdown.innerHTML = `<option value="">${placeholder}</option>`;

            if (!Array.isArray(data)) {
                console.error(`Expected an array for dropdown, but received:`, data);
                this.showAlert(`Invalid data format for dropdown.`, 'danger');
                dropdown.disabled = true;
                return;
            }

            // Populate new options
            data.forEach(item => {
                const option = document.createElement('option');
                option.value = item.id;
                option.textContent = this.getDropdownText(item);
                dropdown.appendChild(option);
            });

            // Disable the dropdown if no data is available
            dropdown.disabled = data.length === 0;
        },

        /**
         * Determine the display text for a dropdown option based on available properties
         * @param {Object} item - The data object
         * @returns {string} - The display text
         */
        getDropdownText(item) {
            if (item.name) {
                return item.name;
            } else if (item.title && item.room_number) {
                return `${item.title} - Room ${item.room_number}`;
            } else if (item.number) { // For Asset Numbers
                return item.number;
            } else {
                return `ID: ${item.id}`;
            }
        },

        /**
         * Update the display box for selected items (images, parts, drawings)
         * @param {string} displayBoxId - The ID of the display container
         * @param {Array} items - The array of selected items
         * @param {string} itemType - The type of items ('image', 'part', 'drawing')
         */
        updateSelectedDisplay(displayBoxId, items, itemType) {
            const displayBox = document.getElementById(displayBoxId);
            if (!displayBox) {
                console.warn(`Element with ID '${displayBoxId}' not found.`);
                return;
            }
            displayBox.innerHTML = ''; // Clear existing items

            if (!Array.isArray(items) || items.length === 0) {
                displayBox.textContent = 'No items selected.';
                return;
            }

            items.forEach(item => {
                const itemDiv = document.createElement('div');
                itemDiv.classList.add('selected-item', 'd-flex', 'align-items-center', 'mb-1');

                // Display different properties based on item type
                switch (itemType) {
                    case 'part':
                        itemDiv.textContent = `${item.part_number || 'N/A'} - ${item.name || 'N/A'}`;
                        break;
                    case 'drawing':
                        itemDiv.textContent = `${item.drw_number || 'N/A'} - ${item.drw_name || 'N/A'}`;
                        break;
                    case 'image':
                        itemDiv.textContent = `${item.title || 'N/A'} - ${item.description || 'N/A'}`;
                        break;
                    default:
                        itemDiv.textContent = `Unknown item type: ${itemType}`;
                }

                const removeButton = document.createElement('button');
                removeButton.classList.add('btn', 'btn-sm', 'btn-outline-danger', 'ms-2');
                removeButton.textContent = 'Remove';
                removeButton.onclick = () => {
                    const index = items.findIndex(selectedItem => selectedItem.id === item.id);
                    if (index > -1) items.splice(index, 1);
                    this.updateSelectedDisplay(displayBoxId, items, itemType);
                };

                itemDiv.appendChild(removeButton);
                displayBox.appendChild(itemDiv);
            });
        },

        /**
         * Clear input fields by their element references
         * @param  {...HTMLElement} elements - The input/select elements to clear
         */
        clearInputFields(...elements) {
            elements.forEach(element => {
                if (element) {
                    if (element.tagName === 'SELECT') {
                        element.selectedIndex = 0;
                    } else {
                        element.value = '';
                    }
                } else {
                    console.warn(`One of the provided elements is undefined.`);
                }
            });
        },

        /**
         * Clear the Edit Task form
         */
        clearEditTaskForm() {
            // Clear Task Name and Description
            const taskNameInput = document.getElementById('pst_task_edit_task_name');
            const taskDescriptionTextarea = document.getElementById('pst_task_edit_task_description');
            this.clearInputFields(taskNameInput, taskDescriptionTextarea);

            // Clear Positions Container
            const positionsContainer = document.getElementById('pst_task_edit_positions_container');
            if (positionsContainer) {
                positionsContainer.innerHTML = '';
            } else {
                console.warn("Element with ID 'pst_task_edit_positions_container' not found.");
            }

            // Clear selected items displays
            ['pst_task_edit_selected_images', 'pst_task_edit_selected_parts', 'pst_task_edit_selected_drawings'].forEach(id => {
                const displayBox = document.getElementById(id);
                if (displayBox) {
                    displayBox.innerHTML = '';
                } else {
                    console.warn(`Element with ID '${id}' not found.`);
                }
            });
        },

        /**
         * Capitalize the first letter of a string
         * @param {string} string - The string to capitalize
         * @returns {string} - The capitalized string
         */
        capitalizeFirstLetter(string) {
            return string.charAt(0).toUpperCase() + string.slice(1);
        }
    };

    // === 3. Helper Function for Fetch with Error Handling ===
    async function fetchWithHandling(url, options = {}) {
        try {
            console.log(`Fetching URL: ${url}`);
            const response = await fetch(url, options);
            const data = await response.json();
            console.log(`Received data from ${url}:`, data);
            if (!response.ok) {
                throw new Error(data.error || 'Unknown error');
            }
            return data;
        } catch (error) {
            SolutionTaskCommon.showAlert(error.message || 'Unknown error', 'danger');
            console.error(`Error fetching ${url}:`, error);
            throw error;
        }
    }

    /**
     * Fetch and display solutions for a specific problem
     * @param {number|string} problemId - The ID of the current problem
     */
    async function fetchSolutions(problemId) {
        SolutionTaskCommon.showAlert('Loading solutions...', 'info');
        try {
            const data = await fetchWithHandling(`${ENDPOINTS.solutions.get}${encodeURIComponent(problemId)}`);

            // Verify data structure and solutions array
            const solutions = data.solutions || data; // Check if solutions are under data.solutions or directly under data
            if (!Array.isArray(solutions)) {
                console.error("Expected an array for solutions, but received:", solutions);
                return; // Exit if solutions is not an array
            }

            // Populate solutions dropdown
            populateSolutionsDropdown(solutions);

            // Update problem name if available
            if (data.problem_name) {
                updateProblemName(data.problem_name);
            }

            // Activate the solution tab
            activateTab('solution-tab');

            console.log("Data received from fetchSolutions:", solutions);
        } catch (error) {
            console.error("Error fetching solutions:", error);
            // Error handling is already managed in fetchWithHandling
        }
    }




    /**
 * Populate solutions dropdown
 * @param {Array} solutions - Array of solution objects
 */
function populateSolutionsDropdown(solutions) {
    const solutionsDropdown = document.getElementById('existing_solutions');

    // Validate that solutions is an array
    if (!Array.isArray(solutions)) {
        console.error("Expected an array for solutions, but received:", solutions);
        return; // Exit function if data is not in the expected format
    }

    // Use the SolutionTaskCommon helper function to populate the dropdown
    SolutionTaskCommon.populateDropdown(solutionsDropdown, solutions, 'Select Solution');
    console.log("Solutions dropdown populated with:", solutions);
}


    /**
     * Update problem name header
     * @param {string} problemName - The name of the problem
     */
    function updateProblemName(problemName) {
        const header = document.getElementById('selected-problem-name');
        if (header) {
            header.textContent = `Problem Solutions for: ${problemName}`;
        } else {
            console.warn("Element with ID 'selected-problem-name' not found.");
        }
    }

    /**
     * Activate a specific tab
     * @param {string} tabId - The ID of the tab to activate
     */
    function activateTab(tabId) {
        const tabLink = document.getElementById(tabId);
        if (tabLink) {
            new bootstrap.Tab(tabLink).show();
        } else {
            console.warn(`Tab with ID '${tabId}' not found.`);
        }
    }

    /**
     * Add a new solution for a problem
     * @param {number|string} problemId - The ID of the current problem
     * @param {string} solutionName - The name of the new solution
     * @param {string} solutionDescription - The description of the new solution
     */
    async function addNewSolution(problemId, solutionName, solutionDescription) {
        if (!problemId) {
            SolutionTaskCommon.showAlert('No problem selected.', 'warning');
            return;
        }

        try {
            // Wait for the server to confirm the solution is added
            const response = await fetchWithHandling(ENDPOINTS.solutions.add, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ problem_id: problemId, name: solutionName, description: solutionDescription })
            });

            // Confirm that the response is successful and contains the expected data
            if (response && response.message === 'Solution added successfully.' && response.solution) {
                SolutionTaskCommon.showAlert('Solution added successfully.', 'success');

                // After confirming the solution is saved, fetch the updated solutions list
                await fetchSolutions(problemId);

                // Clear input fields after successful addition
                const nameInput = document.getElementById('new_solution_name');
                const descInput = document.getElementById('new_solution_description');
                SolutionTaskCommon.clearInputFields(nameInput, descInput);
            } else {
                console.error("Unexpected response format:", response);
                SolutionTaskCommon.showAlert('Failed to add the solution. Unexpected response from the server.', 'warning');
            }

        } catch (error) {
            console.error("Error in addNewSolution:", error);
            SolutionTaskCommon.showAlert('An error occurred while adding the solution.', 'danger');
        }
    }




    /**
     * Remove selected solutions from a problem
     * @param {number|string} problemId - The ID of the current problem
     * @param {Array} solutionIds - Array of solution IDs to remove
     */
    async function removeSolutions(problemId, solutionIds) {
        if (!problemId || !Array.isArray(solutionIds) || solutionIds.length === 0) {
            SolutionTaskCommon.showAlert('No solutions selected for removal.', 'warning');
            return;
        }
        try {
            await fetchWithHandling(ENDPOINTS.solutions.remove, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ problem_id: problemId, solution_ids: solutionIds })
            });
            SolutionTaskCommon.showAlert('Selected solutions removed successfully.', 'success');
            await fetchSolutions(problemId);
        } catch (error) {
            // Error handling is already managed in fetchWithHandling
        }
    }

    /**
     * Fetch tasks for a selected solution and populate the task dropdown
     * @param {number|string} solutionId - The ID of the selected solution
     */
    async function fetchTasksForSolution(solutionId) {
        if (!solutionId) {
            console.warn("Invalid solutionId passed to fetchTasksForSolution.");
            return;
        }
        SolutionTaskCommon.showAlert('Loading tasks...', 'info');
        try {
            const data = await fetchWithHandling(`${ENDPOINTS.tasks.get}${encodeURIComponent(solutionId)}`);
            populateTasksDropdown(data.tasks);
            activateTab('task-tab');
        } catch (error) {
            // Error handling is already managed in fetchWithHandling
        }
    }

    /**
     * Populate tasks in a dropdown list
     * @param {Array} tasks - Array of task objects
     */
    function populateTasksDropdown(tasks) {
        const tasksDropdown = document.getElementById('existing_tasks');
        SolutionTaskCommon.populateDropdown(tasksDropdown, tasks, 'Select Task');


    }


    // Single-click event listener for selecting a task
    document.getElementById('existing_tasks').addEventListener('click', (event) => {
        selectedTaskId = event.target.value; // Set the selected task ID for possible removal
        console.log(`Task selected with ID: ${selectedTaskId}`);
    });

    // Double-click event listener for editing a task
    document.getElementById('existing_tasks').addEventListener('dblclick', (event) => {
        const taskId = event.target.value; // Get the ID of the double-clicked task
        if (taskId) {
            openTaskDetails(taskId); // Call function to open the edit form
        }
    });

    async function openTaskDetails(taskId) {
        // Assign the task ID to the global AppState
        window.AppState.selectedTaskId = taskId;
        console.log(`openTaskDetails called with taskId: ${window.AppState.selectedTaskId}, solutionId: ${window.AppState.currentSolutionId}`);
        try {
            const data = await fetchWithHandling(`${ENDPOINTS.tasks.details}${encodeURIComponent(taskId)}`);

            // Populate the form with task data if found
            if (data && data.task) {
                await populateEditTaskForm(data.task);
                activateTab('edit-task-tab'); // Switch to the edit tab
                console.log(`Editing task with ID: ${taskId}`);
            } else {
                SolutionTaskCommon.clearEditTaskForm(); // Clear form if task is not found
                SolutionTaskCommon.showAlert('Task not found.', 'warning');
            }
        } catch (error) {
            console.error(`Error opening task details for task ID ${taskId}:`, error);
        }
    }

    /**
     * Populate the Edit Task form with task data
     * @param {Object} task - The task data object
     */
    async function populateEditTaskForm(task) {
        // Populate main Task Name and Description
        const taskNameInput = document.getElementById('pst_task_edit_task_name');
        const taskDescriptionTextarea = document.getElementById('pst_task_edit_task_description');

        if (taskNameInput) taskNameInput.value = task.name || '';
        if (taskDescriptionTextarea) taskDescriptionTextarea.value = task.description || '';

        // Populate Positions
        const positionsContainer = document.getElementById('pst_task_edit_positions_container');
        if (positionsContainer) {
            positionsContainer.innerHTML = ''; // Clear existing positions
            if (task.positions?.length) {
                for (let i = 0; i < task.positions.length; i++) {
                    await addPosition(task.positions[i], i);
                }
            } else {
                SolutionTaskCommon.showAlert('No positions associated with this task.', 'info');
            }
        } else {
            console.warn("Element with ID 'pst_task_edit_positions_container' not found.");
        }

        // Populate associated images, parts, and drawings
        SolutionTaskCommon.updateSelectedDisplay('pst_task_edit_selected_images', task.associations?.images || [], 'image');
        SolutionTaskCommon.updateSelectedDisplay('pst_task_edit_selected_parts', task.associations?.parts || [], 'part');
        SolutionTaskCommon.updateSelectedDisplay('pst_task_edit_selected_drawings', task.associations?.drawings || [], 'drawing');
    }

    /**
     * Add a new position section
     * @param {Object} positionData - The data for the position (optional)
     * @param {number} index - The index of the position (for unique IDs)
     */
    async function addPosition(positionData = null, index = 0) {
        const container = document.getElementById('pst_task_edit_positions_container');
        const template = document.getElementById('position-template');
        if (!container || !template) {
            console.warn("Positions container or template not found.");
            return;
        }

        const clone = template.content.cloneNode(true);
        const positionSection = clone.querySelector('.position-section');

        // Assign unique IDs based on index for all select elements
        const elementsToId = ['areaDropdown', 'equipmentGroupDropdown', 'modelDropdown', 'assetNumberInput', 'locationInput', 'siteLocationDropdown'];
        elementsToId.forEach(elementClass => {
            const element = positionSection.querySelector(`.${elementClass}`);
            if (element) {
                element.id = `${elementClass}_${index}`;
                console.log(`Assigned ID ${element.id} to ${elementClass}`);
            }
        });

        // If positionData is provided, populate the fields
        if (positionData) {
            await populatePositionFields(positionSection, positionData, index);
        } else {
            // Initialize dropdowns for a new position
            await initializeNewPosition(positionSection, index);
        }

        // Append the cloned and populated position section to the container
        container.appendChild(clone);
        console.log(`Added new position section with index ${index}`);
    }

    /**
     * Populate position fields based on provided data
     * @param {HTMLElement} positionSection - The position section element
     * @param {Object} positionData - The data for the position
     * @param {number} index - The index of the position
     */
    async function populatePositionFields(positionSection, positionData, index) {
        // Populate Area Dropdown
        const areaDropdown = positionSection.querySelector('.areaDropdown');
        if (areaDropdown) {
            const areas = await fetchInitialAreas();
            SolutionTaskCommon.populateDropdown(areaDropdown, areas, 'Select Area');
            areaDropdown.value = positionData.area_id || '';
            areaDropdown.disabled = false;
            console.log(`Set Area Dropdown (ID: ${areaDropdown.id}) to value: ${positionData.area_id}`);
        }

        // Populate Equipment Group Dropdown based on Area
        const equipmentGroupDropdown = positionSection.querySelector('.equipmentGroupDropdown');
        if (equipmentGroupDropdown && positionData.area_id) {
            const equipmentGroups = await fetchInitialEquipmentGroups(positionData.area_id);
            SolutionTaskCommon.populateDropdown(equipmentGroupDropdown, equipmentGroups, 'Select Equipment Group');

            // Validate equipment_group_id exists in fetched equipmentGroups
            const isValidId = equipmentGroups.some(group => String(group.id) === String(positionData.equipment_group_id));
            if (isValidId) {
                equipmentGroupDropdown.value = positionData.equipment_group_id;
                console.log(`Set Equipment Group Dropdown (ID: ${equipmentGroupDropdown.id}) to value: ${positionData.equipment_group_id}`);
            } else {
                console.warn(`Invalid Equipment Group ID: ${positionData.equipment_group_id} for Equipment Group Dropdown (ID: ${equipmentGroupDropdown.id})`);
                equipmentGroupDropdown.value = '';
            }
            equipmentGroupDropdown.disabled = false;
        }

        // Populate Model Dropdown based on Equipment Group
        const modelDropdown = positionSection.querySelector('.modelDropdown');
        if (modelDropdown && positionData.equipment_group_id) {
            const models = await fetchInitialModels(positionData.equipment_group_id);
            SolutionTaskCommon.populateDropdown(modelDropdown, models, 'Select Model');

            const isValidModelId = models.some(model => String(model.id) === String(positionData.model_id));
            if (isValidModelId) {
                modelDropdown.value = positionData.model_id;
                console.log(`Set Model Dropdown (ID: ${modelDropdown.id}) to value: ${positionData.model_id}`);
            } else {
                console.warn(`Invalid Model ID: ${positionData.model_id} for Model Dropdown (ID: ${modelDropdown.id})`);
                modelDropdown.value = '';
            }
            modelDropdown.disabled = false;
        }

        // Populate Asset Number based on Model
        const assetNumberInput = positionSection.querySelector('.assetNumberInput');
        if (assetNumberInput && positionData.model_id) {
            const assetNumbers = await fetchInitialAssetNumbers(positionData.model_id);
            SolutionTaskCommon.populateDropdown(assetNumberInput, assetNumbers, 'Select Asset Number');

            const isValidAssetNumber = assetNumbers.some(asset => String(asset.id) === String(positionData.asset_number));
            if (isValidAssetNumber) {
                assetNumberInput.value = positionData.asset_number;
                console.log(`Set Asset Number Dropdown (ID: ${assetNumberInput.id}) to value: ${positionData.asset_number}`);
            } else {
                console.warn(`Invalid Asset Number: ${positionData.asset_number} for Asset Number Dropdown (ID: ${assetNumberInput.id})`);
                assetNumberInput.value = '';
            }
            assetNumberInput.disabled = false;
        }

        // Populate Location based on Model
        const locationInput = positionSection.querySelector('.locationInput');
        if (locationInput && positionData.model_id) {
            const locations = await fetchInitialLocations(positionData.model_id);
            SolutionTaskCommon.populateDropdown(locationInput, locations, 'Select Location');

            const isValidLocation = locations.some(location => String(location.id) === String(positionData.location));
            if (isValidLocation) {
                locationInput.value = positionData.location;
                console.log(`Set Location Dropdown (ID: ${locationInput.id}) to value: ${positionData.location}`);
            } else {
                console.warn(`Invalid Location ID: ${positionData.location} for Location Dropdown (ID: ${locationInput.id})`);
                locationInput.value = '';
            }
            locationInput.disabled = false;
        }

        // Populate Site Location based on Model, Asset Number, and Location
        const siteLocationDropdown = positionSection.querySelector('.siteLocationDropdown');
        if (siteLocationDropdown && positionData.model_id && positionData.asset_number && positionData.location) {
            const siteLocations = await fetchInitialSiteLocations(positionData.model_id, positionData.asset_number, positionData.location);
            SolutionTaskCommon.populateDropdown(siteLocationDropdown, siteLocations, 'Select Site Location');

            const isValidSiteLocationId = siteLocations.some(siteLoc => String(siteLoc.id) === String(positionData.site_location_id));
            if (isValidSiteLocationId) {
                siteLocationDropdown.value = positionData.site_location_id;
                console.log(`Set Site Location Dropdown (ID: ${siteLocationDropdown.id}) to value: ${positionData.site_location_id}`);
            } else {
                console.warn(`Invalid Site Location ID: ${positionData.site_location_id} for Site Location Dropdown (ID: ${siteLocationDropdown.id})`);
                siteLocationDropdown.value = '';
            }
            siteLocationDropdown.disabled = false;
        }

        // Add event listeners for dynamic dropdown dependencies
        addPositionEventListeners(positionSection, index);
    }

    /**
     * Initialize a new position with default dropdown states
     * @param {HTMLElement} positionSection - The position section element
     * @param {number} index - The index of the position
     */
    async function initializeNewPosition(positionSection, index) {
        // Populate Area Dropdown
        const areaDropdown = positionSection.querySelector('.areaDropdown');
        if (areaDropdown) {
            const areas = await fetchInitialAreas();
            SolutionTaskCommon.populateDropdown(areaDropdown, areas, 'Select Area');
            areaDropdown.disabled = false;
            console.log(`Initialized Area Dropdown (ID: ${areaDropdown.id})`);
        }

        // Disable subsequent dropdowns until selections are made
        ['equipmentGroupDropdown', 'modelDropdown', 'assetNumberInput', 'locationInput', 'siteLocationDropdown'].forEach(className => {
            const element = positionSection.querySelector(`.${className}`);
            if (element) {
                SolutionTaskCommon.populateDropdown(element, [], `Select ${SolutionTaskCommon.capitalizeFirstLetter(className.replace('Dropdown', '').replace('Input', ''))}`);
                element.disabled = true;
                console.log(`Initialized and disabled ${className} (ID: ${element.id})`);
            }
        });

        // Add event listeners for dynamic dropdown dependencies
        addPositionEventListeners(positionSection, index);
    }

    /**
     * Add event listeners for dynamic dropdown dependencies within a position section
     * @param {HTMLElement} positionSection - The position section element
     * @param {number} index - The index of the position
     */
    function addPositionEventListeners(positionSection, index) {
        const areaDropdown = positionSection.querySelector('.areaDropdown');
        const equipmentGroupDropdown = positionSection.querySelector('.equipmentGroupDropdown');
        const modelDropdown = positionSection.querySelector('.modelDropdown');
        const assetNumberInput = positionSection.querySelector('.assetNumberInput');
        const locationInput = positionSection.querySelector('.locationInput');
        const siteLocationDropdown = positionSection.querySelector('.siteLocationDropdown');

        if (areaDropdown) {
            areaDropdown.addEventListener('change', async () => {
                const selectedAreaId = areaDropdown.value;
                console.log(`Area Dropdown (ID: ${areaDropdown.id}) changed to: ${selectedAreaId}`);
                if (selectedAreaId) {
                    const equipmentGroups = await fetchInitialEquipmentGroups(selectedAreaId);
                    SolutionTaskCommon.populateDropdown(equipmentGroupDropdown, equipmentGroups, 'Select Equipment Group');
                    equipmentGroupDropdown.disabled = false;
                    console.log(`Populated Equipment Group Dropdown (ID: ${equipmentGroupDropdown.id}) with Equipment Groups for Area ID ${selectedAreaId}`);

                    // Reset and disable subsequent dropdowns
                    resetDropdowns([modelDropdown, assetNumberInput, locationInput, siteLocationDropdown]);
                } else {
                    // Reset and disable all dependent dropdowns
                    resetDropdowns([equipmentGroupDropdown, modelDropdown, assetNumberInput, locationInput, siteLocationDropdown]);
                }
            });
        }

        if (equipmentGroupDropdown) {
            equipmentGroupDropdown.addEventListener('change', async () => {
                const selectedEquipmentGroupId = equipmentGroupDropdown.value;
                console.log(`Equipment Group Dropdown (ID: ${equipmentGroupDropdown.id}) changed to: ${selectedEquipmentGroupId}`);
                if (selectedEquipmentGroupId) {
                    const models = await fetchInitialModels(selectedEquipmentGroupId);
                    SolutionTaskCommon.populateDropdown(modelDropdown, models, 'Select Model');
                    modelDropdown.disabled = false;
                    console.log(`Populated Model Dropdown (ID: ${modelDropdown.id}) with Models for Equipment Group ID ${selectedEquipmentGroupId}`);

                    // Reset and disable subsequent dropdowns
                    resetDropdowns([assetNumberInput, locationInput, siteLocationDropdown]);
                } else {
                    // Reset and disable all dependent dropdowns
                    resetDropdowns([modelDropdown, assetNumberInput, locationInput, siteLocationDropdown]);
                }
            });
        }

        if (modelDropdown) {
            modelDropdown.addEventListener('change', async () => {
                const selectedModelId = modelDropdown.value;
                console.log(`Model Dropdown (ID: ${modelDropdown.id}) changed to: ${selectedModelId}`);
                if (selectedModelId) {
                    SolutionTaskCommon.showAlert('Loading asset numbers and locations...', 'info');
                    const [assetNumbers, locations] = await Promise.all([
                        fetchInitialAssetNumbers(selectedModelId),
                        fetchInitialLocations(selectedModelId)
                    ]);
                    SolutionTaskCommon.populateDropdown(assetNumberInput, assetNumbers, 'Select Asset Number');
                    assetNumberInput.disabled = false;
                    console.log(`Populated Asset Number Dropdown (ID: ${assetNumberInput.id}) with Asset Numbers for Model ID ${selectedModelId}`);

                    SolutionTaskCommon.populateDropdown(locationInput, locations, 'Select Location');
                    locationInput.disabled = false;
                    console.log(`Populated Location Dropdown (ID: ${locationInput.id}) with Locations for Model ID ${selectedModelId}`);

                    // Reset and disable Site Location Dropdown
                    SolutionTaskCommon.populateDropdown(siteLocationDropdown, [], 'Select Site Location');
                    siteLocationDropdown.disabled = true;
                    console.log(`Reset and disabled Site Location Dropdown (ID: ${siteLocationDropdown.id})`);
                } else {
                    // Reset and disable all dependent dropdowns
                    resetDropdowns([assetNumberInput, locationInput, siteLocationDropdown]);
                }
            });
        }

        if (assetNumberInput) {
            assetNumberInput.addEventListener('change', async () => {
                const selectedAssetNumber = assetNumberInput.value;
                const selectedModelId = modelDropdown.value;
                const selectedLocation = locationInput.value;
                console.log(`Asset Number Dropdown (ID: ${assetNumberInput.id}) changed to: ${selectedAssetNumber}`);

                if (selectedAssetNumber && selectedModelId && selectedLocation) {
                    SolutionTaskCommon.showAlert('Loading site locations...', 'info');
                    const siteLocations = await fetchInitialSiteLocations(selectedModelId, selectedAssetNumber, selectedLocation);
                    SolutionTaskCommon.populateDropdown(siteLocationDropdown, siteLocations, 'Select Site Location');
                    siteLocationDropdown.disabled = false;
                    console.log(`Populated Site Location Dropdown (ID: ${siteLocationDropdown.id}) with Site Locations for Model ID ${selectedModelId}, Asset Number ${selectedAssetNumber}, Location ${selectedLocation}`);
                } else {
                    SolutionTaskCommon.populateDropdown(siteLocationDropdown, [], 'Select Site Location');
                    siteLocationDropdown.disabled = true;
                    console.log(`Reset and disabled Site Location Dropdown (ID: ${siteLocationDropdown.id}) due to incomplete selections`);
                }
            });
        }

        // Add event listener for remove button
        const removeBtn = positionSection.querySelector('.removePositionBtn');
        if (removeBtn) {
            removeBtn.addEventListener('click', () => {
                positionSection.remove();
                console.log(`Removed position section with index ${index}`);
            });
        }
    }

    /**
     * Reset and disable multiple dropdowns
     * @param {Array} elements - Array of dropdown/select elements to reset
     */
    function resetDropdowns(elements) {
        elements.forEach(element => {
            if (element) {
                const placeholderText = `Select ${SolutionTaskCommon.capitalizeFirstLetter(element.className.replace('Dropdown', '').replace('Input', ''))}`;
                SolutionTaskCommon.populateDropdown(element, [], placeholderText);
                element.disabled = true;
                console.log(`Reset and disabled ${element.className} (ID: ${element.id})`);
            }
        });
    }

    // === 5. Fetch Initial Data Functions ===

    async function fetchInitialAreas() {
        try {
            const data = await fetchWithHandling(ENDPOINTS.initialData.areas);
            return data.areas || [];
        } catch (error) {
            return [];
        }
    }


    async function fetchInitialEquipmentGroups(areaId) {
        try {
            const url = `${ENDPOINTS.initialData.equipmentGroups}?area_id=${encodeURIComponent(areaId)}`;
            const data = await fetchWithHandling(url);
            console.log(`Received Equipment Groups:`, data);
            return Array.isArray(data) ? data : [];
        } catch (error) {
            console.error(`Error fetching Equipment Groups for Area ID ${areaId}:`, error);
            return [];
        }
    }

    async function fetchInitialModels(equipmentGroupId) {
        try {
            const url = `${ENDPOINTS.initialData.models}?equipment_group_id=${encodeURIComponent(equipmentGroupId)}`;
            const data = await fetchWithHandling(url);
            console.log(`Received Models:`, data);
            return Array.isArray(data) ? data : [];
        } catch (error) {
            console.error(`Error fetching Models for Equipment Group ID ${equipmentGroupId}:`, error);
            return [];
        }
    }

    async function fetchInitialAssetNumbers(modelId) {
        try {
            const url = `${ENDPOINTS.initialData.assetNumbers}?model_id=${encodeURIComponent(modelId)}`;
            const data = await fetchWithHandling(url);
            console.log(`Received Asset Numbers:`, data);
            return Array.isArray(data) ? data : [];
        } catch (error) {
            console.error(`Error fetching Asset Numbers for Model ID ${modelId}:`, error);
            return [];
        }
    }

    async function fetchInitialLocations(modelId) {
        try {
            const url = `${ENDPOINTS.initialData.locations}?model_id=${encodeURIComponent(modelId)}`;
            const data = await fetchWithHandling(url);
            console.log(`Received Locations:`, data);
            return Array.isArray(data) ? data : [];
        } catch (error) {
            console.error(`Error fetching Locations for Model ID ${modelId}:`, error);
            return [];
        }
    }

    async function fetchInitialSiteLocations(modelId, assetNumber, location) {
        try {
            const queryParams = new URLSearchParams({ model_id: modelId, asset_number: assetNumber, location: location });
            const url = `${ENDPOINTS.initialData.siteLocations}?${queryParams.toString()}`;
            const data = await fetchWithHandling(url);
            console.log(`Received Site Locations:`, data);
            return Array.isArray(data) ? data : [];
        } catch (error) {
            console.error(`Error fetching Site Locations for Model ID ${modelId}, Asset Number ${assetNumber}, Location ${location}:`, error);
            return [];
        }
    }

    // === 6. Event Listeners ===

    /**
     * Initialize event listeners for the form
     */
    function initializeEventListeners() {
    const addSolutionBtn = document.getElementById('addSolutionBtn');
    if (addSolutionBtn) {
        addSolutionBtn.addEventListener('click', () => {
            const nameInput = document.getElementById('new_solution_name');
            const descInput = document.getElementById('new_solution_description');
            const name = nameInput.value.trim();
            const description = descInput?.value.trim();

            // Retrieve currentProblemId from sessionStorage
            const problemId = parseInt(sessionStorage.getItem('currentProblemId'), 10);
            console.log('Add Solution Clicked. name:', name, 'currentProblemId:', problemId);

            if (name && problemId) {
                addNewSolution(problemId, name, description);
            } else {
                SolutionTaskCommon.showAlert('Solution name cannot be empty or no problem selected.', 'warning');
            }
        });
    }

    // Adding an event listener for the solution dropdown to set window.AppState.currentSolutionId
    const existingSolutions = document.getElementById('existing_solutions');
    if (existingSolutions) {
        existingSolutions.addEventListener('change', (event) => {
            const solutionId = parseInt(event.target.value, 10);
            window.AppState.currentSolutionId = solutionId || null; // Set to integer or null if empty
            console.log(`Updated Current Solution ID: ${window.AppState.currentSolutionId}`);
        });
    }
}



        // Existing Solutions Dropdown Change
        const existingSolutions = document.getElementById('existing_solutions');
        if (existingSolutions) {
            existingSolutions.addEventListener('change', (event) => {
                const solutionId = parseInt(event.target.value, 10); // Ensure solutionId is an integer
                console.log(`Solutions Dropdown changed to: ${solutionId}`);

                if (solutionId) {
                    // Update the globally accessible currentSolutionId in window.AppState
                    window.AppState.currentSolutionId = solutionId;
                    fetchTasksForSolution(window.AppState.currentSolutionId);
                } else {
                    // Reset globally accessible currentSolutionId in window.AppState
                    window.AppState.currentSolutionId = null;
                    const tasksDropdown = document.getElementById('existing_tasks');
                    if (tasksDropdown) {
                        SolutionTaskCommon.populateDropdown(tasksDropdown, [], 'Select Task');
                        tasksDropdown.disabled = true;
                        console.log(`Cleared Tasks Dropdown`);
                    }
                    SolutionTaskCommon.clearEditTaskForm();
                }

                // Log the updated solution ID in window.AppState for verification
                console.log(`Updated Current Solution ID: ${window.AppState.currentSolutionId}`);
            });
        }




        // Add Task Button
        const addTaskBtn = document.getElementById('addTaskBtn');
        if (addTaskBtn) {
            addTaskBtn.addEventListener('click', () => {
                const nameInput = document.getElementById('new_task_name');
                const descInput = document.getElementById('new_task_description');
                const name = nameInput.value.trim();
                const description = descInput?.value.trim();

                // Use window.AppState.currentSolutionId for consistency
                const solutionId = window.AppState.currentSolutionId;
                console.log('Task Name:', name, 'Current Solution ID:', solutionId);

                if (name && solutionId) {
                    addNewTask(solutionId, name, description);
                } else {
                    SolutionTaskCommon.showAlert('Task name cannot be empty or no solution selected.', 'warning');
                }
            });
        }



        // Add Position Button
        const addPositionBtn = document.getElementById('addPositionBtn');
        if (addPositionBtn) {
            addPositionBtn.addEventListener('click', () => {
                const positionsContainer = document.getElementById('pst_task_edit_positions_container');
                const currentIndex = positionsContainer ? positionsContainer.children.length : 0;
                console.log(`Adding new position with index: ${currentIndex}`);
                addPosition(null, currentIndex);
            });
        }

        // Optional: Implement removal of selected solutions or tasks based on UI needs


    /**
     * Add a new task to a solution
     * @param {number|string} solutionId - The ID of the current solution
     * @param {string} name - The name of the new task
     * @param {string} description - The description of the new task
     */
    async function addNewTask(solutionId = window.currentSolutionId, name, description) {
        // Confirm solution ID is valid
        console.log("Adding Task:", { solutionId, name, description });

        if (!solutionId) {
            SolutionTaskCommon.showAlert('No solution selected.', 'warning');
            return;
        }
        try {
            await fetchWithHandling(ENDPOINTS.tasks.add, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ solution_id: solutionId, name, description })
            });
            SolutionTaskCommon.showAlert('Task added successfully.', 'success');

            // Fetch updated tasks list for this solution
            await fetchTasksForSolution(solutionId);

            // Clear input fields
            const nameInput = document.getElementById('new_task_name');
            const descInput = document.getElementById('new_task_description');
            SolutionTaskCommon.clearInputFields(nameInput, descInput);
        } catch (error) {
            console.error("Error in addNewTask:", error);
            SolutionTaskCommon.showAlert('An error occurred while adding the task.', 'danger');
        }
    }

    /**
     * Remove selected tasks from a solution
     * @param {number|string} solutionId - The ID of the current solution
     * @param {Array} taskIds - Array of task IDs to remove
     */
    async function removeTasks(solutionId, taskIds) {
        if (!solutionId || !Array.isArray(taskIds) || taskIds.length === 0) {
            SolutionTaskCommon.showAlert('No tasks selected for removal.', 'warning');
            return;
        }
        try {
            await fetchWithHandling(ENDPOINTS.tasks.remove, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ solution_id: solutionId, task_ids: taskIds })
            });
            SolutionTaskCommon.showAlert('Selected tasks removed successfully.', 'success');
            await fetchTasksForSolution(solutionId);
        } catch (error) {
            // Error handling is already managed in fetchWithHandling
        }
    }
    // Event listener for Remove Task button
    document.getElementById('removeTaskBtn').addEventListener('click', () => {
        const tasksDropdown = document.getElementById('existing_tasks');
        const selectedTaskId = tasksDropdown.value;

        // Check if a task is selected
        if (!selectedTaskId) {
            SolutionTaskCommon.showAlert('Please select a task to remove.', 'warning');
            return;
        }

        // Confirm removal
        const confirmDelete = confirm('Are you sure you want to remove the selected task?');
        if (!confirmDelete) return;

        // Call the remove task function
        removeTask(selectedTaskId);
    });

    // Function to remove the task
    function removeTask(taskId) {
        fetch('/pst_troubleshooting_solution/remove_task/', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ task_id: taskId })
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                SolutionTaskCommon.showAlert(data.message, 'success');
                // Optionally, refresh the tasks dropdown to reflect the change
                fetchTasksForSolution(currentSolutionId);
            } else {
                SolutionTaskCommon.showAlert(data.error || "Failed to remove task", 'danger');
            }
        })
        .catch(error => console.error('Error removing task:', error));
    }


    // === 7. Initialize Event Listeners ===
    initializeEventListeners();

    // === 8. Optional: Fetch and Display Initial Data ===
    // You can call fetchSolutions with a default problem ID or based on user selection
    // Example:
    // fetchSolutions(initialProblemId);
});