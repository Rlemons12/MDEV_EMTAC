// static/js/pst_troubleshooting_position_update.js

document.addEventListener('DOMContentLoaded', () => {
    'use strict';

    // Explicitly set each endpoint URL
    const GET_EQUIPMENT_GROUPS_URL = '/pst_troubleshooting/get_equipment_groups';
    const GET_MODELS_URL = '/pst_troubleshooting/get_models';
    const GET_ASSET_NUMBERS_URL = '/pst_troubleshooting/get_asset_numbers';
    const GET_LOCATIONS_URL = '/pst_troubleshooting/get_locations';
    const GET_SITE_LOCATIONS_URL = '/pst_troubleshooting/get_site_locations';
    const SEARCH_PROBLEMS_URL = '/pst_troubleshooting_position_update/search_problems';
    const GET_PROBLEM_DETAILS_URL = '/pst_troubleshooting_position_update/get_problem/';
    const GET_SOLUTIONS_URL = '/pst_troubleshooting_solution/get_solutions/';

    // Dropdown elements
    const dropdowns = {
        area: document.getElementById('pst_areaDropdown'),
        equipmentGroup: document.getElementById('pst_equipmentGroupDropdown'),
        model: document.getElementById('pst_modelDropdown'),
        assetNumberInput: document.getElementById('pst_assetNumberInput'),
        locationInput: document.getElementById('pst_locationInput'),
        siteLocation: document.getElementById('pst_siteLocationDropdown')
    };

    console.log("JavaScript loaded and ready");

    // Initialize Select2 for Site Location Dropdown
    if (dropdowns.siteLocation) {
        $('#pst_siteLocationDropdown').select2({
            placeholder: 'Select Site Location or type "New..."',
            allowClear: true
        });
    }

    // Event listeners for dropdowns
    if (dropdowns.area) {
        dropdowns.area.addEventListener('change', handleAreaChange);
    }

    if (dropdowns.equipmentGroup) {
        dropdowns.equipmentGroup.addEventListener('change', handleEquipmentGroupChange);
    }

    if (dropdowns.model) {
        dropdowns.model.addEventListener('change', handleModelChange);
    }

    // Search button event listener
    const searchButton = document.getElementById('searchProblemByPositionBtn');
    if (searchButton) {
        searchButton.addEventListener('click', handleSearchButtonClick);
    }

    // Event delegation for dynamically generated buttons
    const resultsList = document.getElementById('pst_positionResultsList');
    if (resultsList) {
        resultsList.addEventListener('click', handleResultsListClick);
    }

    // Function to handle Area change
    function handleAreaChange() {
        const areaId = dropdowns.area.value;
        if (areaId) {
            fetchData(`${GET_EQUIPMENT_GROUPS_URL}?area_id=${encodeURIComponent(areaId)}`)
                .then(data => populateDropdown(dropdowns.equipmentGroup, data, 'Select Equipment Group'))
                .catch(error => console.error('Error fetching equipment groups:', error));
        } else {
            resetAllDropdowns();
        }
    }

    // Function to handle Equipment Group change
    function handleEquipmentGroupChange() {
        const equipmentGroupId = dropdowns.equipmentGroup.value;
        if (equipmentGroupId) {
            fetchData(`${GET_MODELS_URL}?equipment_group_id=${encodeURIComponent(equipmentGroupId)}`)
                .then(data => populateDropdown(dropdowns.model, data, 'Select Model'))
                .catch(error => console.error('Error fetching models:', error));
        } else {
            resetDropdown(dropdowns.model, 'Select Model');
            resetDropdown(dropdowns.siteLocation, 'Select Site Location');
        }
    }

    // Function to handle Model change
    function handleModelChange() {
        const modelId = dropdowns.model.value;
        if (modelId) {
            fetchData(`${GET_ASSET_NUMBERS_URL}?model_id=${encodeURIComponent(modelId)}`)
                .then(data => {
                    // Handle asset numbers if needed
                    console.log("Asset numbers loaded:", data);
                    // If you need to populate a dropdown, call populateDropdown here
                })
                .catch(error => console.error('Error fetching asset numbers:', error));

            fetchData(`${GET_LOCATIONS_URL}?model_id=${encodeURIComponent(modelId)}`)
                .then(data => populateDropdown(dropdowns.siteLocation, data, 'Select Site Location'))
                .catch(error => console.error('Error fetching locations:', error));
        } else {
            resetDropdown(dropdowns.siteLocation, 'Select Site Location');
        }
    }

    // Function to handle Search button click
    function handleSearchButtonClick() {
        const searchParams = {
            area_id: dropdowns.area.value,
            equipment_group_id: dropdowns.equipmentGroup.value,
            model_id: dropdowns.model.value,
            asset_number: dropdowns.assetNumberInput.value.trim(),
            location: dropdowns.locationInput.value.trim(),
            site_location_id: dropdowns.siteLocation.value
        };

        // Check if at least one search criterion is provided
        const hasCriteria = Object.values(searchParams).some(value => value && value !== '');

        if (hasCriteria) {
            const queryString = new URLSearchParams(searchParams).toString();
            fetchData(`${SEARCH_PROBLEMS_URL}?${queryString}`)
                .then(data => displaySearchResults(data))
                .catch(error => console.error('Error searching for problems:', error));
        } else {
            alert('Please enter at least one search criterion.');
        }
    }

    // Function to handle click events in the search results list
    function handleResultsListClick(event) {
        if (event.target.classList.contains('update-problem-btn')) {
            const problemId = event.target.dataset.problemId;
            fetchProblemDetails(problemId);
        } else if (event.target.classList.contains('edit-solutions-btn')) {
            const problemId = event.target.dataset.problemId;
            editRelatedSolutions(problemId);
        }
    }

    // Fetch data with error handling
    function fetchData(url) {
        return fetch(url)
            .then(response => {
                if (!response.ok) {
                    return response.json().then(errorData => {
                        throw new Error(errorData.error || 'Unknown error occurred.');
                    });
                }
                return response.json();
            });
    }

    // Display search results
    function displaySearchResults(problems) {
        if (!Array.isArray(problems)) {
            console.error('Invalid data format received for search results.');
            return;
        }

        resultsList.innerHTML = '';
        document.getElementById('pst_searchResults').style.display = problems.length ? 'block' : 'none';

        problems.forEach(problem => {
            const listItem = document.createElement('li');
            listItem.classList.add('list-group-item');

            listItem.innerHTML = `
                <div>
                    <strong>${problem.name}</strong> - ${problem.description}
                </div>
                <div class="mt-2">
                    <button class="btn btn-sm btn-warning update-problem-btn" data-problem-id="${problem.id}">
                        Update Problem Position
                    </button>
                    <button class="btn btn-sm btn-info edit-solutions-btn" data-problem-id="${problem.id}">
                        Edit Related Solutions
                    </button>
                </div>
                
            `;
            resultsList.appendChild(listItem);
        });
    }

    // Fetch and display problem details in the update form
    function fetchProblemDetails(problemId) {
        if (!problemId) {
            console.error('Invalid problem ID:', problemId);
            return;
        }

        fetchData(`${GET_PROBLEM_DETAILS_URL}${encodeURIComponent(problemId)}`)
            .then(data => populateUpdateForm(data))
            .catch(error => console.error('Error fetching problem details:', error));
    }

    // Populate Update Form
    function populateUpdateForm(data) {
        // References to update form elements
        const updateDropdowns = {
            area: document.getElementById('update_pst_areaDropdown'),
            equipmentGroup: document.getElementById('update_pst_equipmentGroupDropdown'),
            model: document.getElementById('update_pst_modelDropdown'),
            assetNumberInput: document.getElementById('update_pst_assetNumberInput'),
            locationInput: document.getElementById('update_pst_locationInput'),
            siteLocation: document.getElementById('update_pst_siteLocationDropdown')
        };

        if (!data || !data.problem || !data.position) {
            console.error('Problem data is undefined or null.');
            alert('Error loading problem details.');
            return;
        }

        // Set hidden problem ID
        document.getElementById('update_problem_id').value = data.problem.id || '';
        document.getElementById('update_problem_name').value = data.problem.name || '';
        document.getElementById('update_problem_description').value = data.problem.description || '';

        // Populate and set the Area dropdown
        if (updateDropdowns.area && data.position.area_id) {
            updateDropdowns.area.value = data.position.area_id;
            updateDropdowns.area.disabled = false;

            // Fetch equipment groups based on the area
            fetchData(`${GET_EQUIPMENT_GROUPS_URL}?area_id=${encodeURIComponent(data.position.area_id)}`)
                .then(equipmentGroups => {
                    populateDropdown(updateDropdowns.equipmentGroup, equipmentGroups, 'Select Equipment Group');
                    updateDropdowns.equipmentGroup.value = data.position.equipment_group_id || '';

                    // Fetch and populate Models based on the Equipment Group
                    return fetchData(`${GET_MODELS_URL}?equipment_group_id=${encodeURIComponent(data.position.equipment_group_id)}`);
                })
                .then(models => {
                    populateDropdown(updateDropdowns.model, models, 'Select Model');
                    updateDropdowns.model.value = data.position.model_id || '';

                    // Enable the dropdowns
                    updateDropdowns.equipmentGroup.disabled = false;
                    updateDropdowns.model.disabled = false;
                })
                .catch(error => console.error('Error fetching equipment groups or models:', error));
        } else {
            console.error('Area ID is missing in position data.');
            alert('Area information is missing in problem details.');
        }

        // Set Asset Number and Location
        if (updateDropdowns.assetNumberInput) {
            updateDropdowns.assetNumberInput.value = data.position.asset_number || '';
        }
        if (updateDropdowns.locationInput) {
            updateDropdowns.locationInput.value = data.position.location || '';
        }

        // Fetch and populate Site Locations
        if (updateDropdowns.siteLocation) {
            fetchData(GET_SITE_LOCATIONS_URL)
                .then(siteLocations => {
                    if (Array.isArray(siteLocations)) {
                        populateDropdown(updateDropdowns.siteLocation, siteLocations, 'Select Site Location');
                        updateDropdowns.siteLocation.value = data.position.site_location_id || '';
                        updateDropdowns.siteLocation.disabled = false;
                    } else {
                        console.error('Invalid site locations data format.');
                    }
                })
                .catch(error => console.error('Error fetching site locations:', error));
        }

        // Show the update problem section
        const updateSection = document.getElementById('pst_updateProblemSection');
        if (updateSection) {
            updateSection.style.display = 'block';
        }
    }


    // Fetch and populate Solutions Tab with related solutions for the selected problem
    function editRelatedSolutions(problemId) {
        if (!problemId) {
            console.error('Invalid problem ID:', problemId);
            return;
        }

        fetchData(`${GET_SOLUTIONS_URL}${encodeURIComponent(problemId)}`)
            .then(data => {
                populateSolutionsTab(data);
                // Switch to the Solutions tab after populating
                const solutionTab = document.getElementById('solution-tab');
                if (solutionTab) {
                    solutionTab.click();
                }
            })
            .catch(error => console.error('Error fetching related solutions:', error));
    }

    // Populate Solutions Tab
    function populateSolutionsTab(solutions) {
        const solutionsDropdown = document.getElementById('existing_solutions');
        if (!solutionsDropdown) {
            console.error('Solutions dropdown element not found.');
            return;
        }

        solutionsDropdown.innerHTML = ''; // Clear previous options

        if (Array.isArray(solutions)) {
            solutions.forEach(solution => {
                const option = document.createElement('option');
                option.value = solution.id;
                option.textContent = `${solution.name} - ${solution.description}`;
                solutionsDropdown.appendChild(option);
            });
        } else {
            console.error('Invalid solutions data format.');
        }
    }

    // Utility functions
    function populateDropdown(dropdown, data, placeholder) {
        if (!dropdown) {
            console.error('Dropdown element is undefined.');
            return;
        }

        dropdown.innerHTML = `<option value="">${placeholder}</option>`;

        if (Array.isArray(data)) {
            data.forEach(item => {
                let displayText = item.name || `${item.title} - Room ${item.room_number}`;
                const option = document.createElement('option');
                option.value = item.id;
                option.textContent = displayText;
                dropdown.appendChild(option);
            });
            dropdown.disabled = false;
        } else {
            console.error('Data provided to populateDropdown is not an array:', data);
        }
    }

    function resetDropdown(dropdown, placeholder) {
        if (dropdown) {
            dropdown.innerHTML = `<option value="">${placeholder}</option>`;
            dropdown.disabled = true;
        }
    }

    function resetAllDropdowns() {
        resetDropdown(dropdowns.equipmentGroup, 'Select Equipment Group');
        resetDropdown(dropdowns.model, 'Select Model');
        resetDropdown(dropdowns.siteLocation, 'Select Site Location');
        if (dropdowns.assetNumberInput) {
            dropdowns.assetNumberInput.value = '';
        }
        if (dropdowns.locationInput) {
            dropdowns.locationInput.value = '';
        }
    }
});
