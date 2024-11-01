document.addEventListener('DOMContentLoaded', () => {
    // Explicitly set each endpoint URL
    const GET_EQUIPMENT_GROUPS_URL = '/pst_troubleshooting/get_equipment_groups';
    const GET_MODELS_URL = '/pst_troubleshooting/get_models';
    const GET_ASSET_NUMBERS_URL = '/pst_troubleshooting/get_asset_numbers';
    const GET_LOCATIONS_URL = '/pst_troubleshooting/get_locations';
    const GET_SITE_LOCATIONS_URL = '/pst_troubleshooting/get_site_locations';
    const SEARCH_PROBLEMS_URL = '/pst_troubleshooting_position_update/search_problems';
    const GET_PROBLEM_DETAILS_URL = '/pst_troubleshooting_position_update/get_problem_details/';
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

    // Event listeners for dropdowns
    dropdowns.area.addEventListener('change', () => {
        const areaId = dropdowns.area.value;
        if (areaId) {
            fetch(`${GET_EQUIPMENT_GROUPS_URL}?area_id=${areaId}`)
                .then(response => response.json())
                .then(data => populateDropdown(dropdowns.equipmentGroup, data, 'Select Equipment Group'))
                .catch(error => console.error('Error fetching equipment groups:', error));
        } else {
            resetAllDropdowns();
        }
    });

    dropdowns.equipmentGroup.addEventListener('change', () => {
        const equipmentGroupId = dropdowns.equipmentGroup.value;
        if (equipmentGroupId) {
            fetch(`${GET_MODELS_URL}?equipment_group_id=${equipmentGroupId}`)
                .then(response => response.json())
                .then(data => populateDropdown(dropdowns.model, data, 'Select Model'))
                .catch(error => console.error('Error fetching models:', error));
        } else {
            resetDropdown(dropdowns.model, 'Select Model');
            resetDropdown(dropdowns.siteLocation, 'Select Site Location');
        }
    });

    dropdowns.model.addEventListener('change', () => {
        const modelId = dropdowns.model.value;
        if (modelId) {
            fetch(`${GET_ASSET_NUMBERS_URL}?model_id=${modelId}`)
                .then(response => response.json())
                .then(data => console.log("Asset numbers loaded:", data))
                .catch(error => console.error('Error fetching asset numbers:', error));

            fetch(`${GET_LOCATIONS_URL}?model_id=${modelId}`)
                .then(response => response.json())
                .then(data => populateDropdown(dropdowns.siteLocation, data, 'Select Site Location'))
                .catch(error => console.error('Error fetching locations:', error));
        } else {
            resetDropdown(dropdowns.siteLocation, 'Select Site Location');
        }
    });

    // Search button event listener
    document.getElementById('searchProblemByPositionBtn').addEventListener('click', () => {
        const searchParams = {
            area_id: dropdowns.area.value,
            equipment_group_id: dropdowns.equipmentGroup.value,
            model_id: dropdowns.model.value,
            asset_number: dropdowns.assetNumberInput.value.trim(),
            location: dropdowns.locationInput.value.trim(),
            site_location_id: dropdowns.siteLocation.value
        };

        if (Object.values(searchParams).some(value => value)) {
            fetch(`${SEARCH_PROBLEMS_URL}?${new URLSearchParams(searchParams)}`)
                .then(response => response.json())
                .then(data => displaySearchResults(data))
                .catch(error => console.error('Error searching for problems:', error));
        } else {
            alert('Please enter at least one search criterion.');
        }
    });

    // Display search results with Update and Edit Solutions buttons
    function displaySearchResults(problems) {
        const resultsList = document.getElementById('pst_positionResultsList');
        resultsList.innerHTML = '';
        document.getElementById('pst_searchResults').style.display = problems.length ? 'block' : 'none';

        problems.forEach(problem => {
            const listItem = document.createElement('li');
            listItem.classList.add('list-group-item');
            listItem.innerHTML = `
                <strong>${problem.name}</strong> - ${problem.description}
                <button class="btn btn-sm btn-warning float-end ms-2 update-problem-btn" data-problem-id="${problem.id}">Update Problem Position</button>
                <button class="btn btn-sm btn-info float-end edit-solutions-btn" data-problem-id="${problem.id}">Edit Related Solutions</button>
                <br>
                <small>Area: ${problem.area}</small><br>
                <small>Equipment Group: ${problem.equipment_group}</small><br>
                <small>Model: ${problem.model}</small><br>
                <small>Asset Number: ${problem.asset_number}</small><br>
                <small>Location: ${problem.location}</small><br>
                <small>Site Location: ${problem.site_location}</small>
            `;
            resultsList.appendChild(listItem);
        });

        // Use event delegation to handle clicks on dynamically generated buttons
        resultsList.addEventListener('click', (event) => {
            if (event.target.classList.contains('update-problem-btn')) {
                const problemId = event.target.dataset.problemId;
                console.log("Update Problem Position button clicked for problem ID:", problemId);
                fetchProblemDetails(problemId);
            }
            if (event.target.classList.contains('edit-solutions-btn')) {
                const problemId = event.target.dataset.problemId;
                console.log("Edit Related Solutions button clicked for problem ID:", problemId);
                editRelatedSolutions(problemId);
            }
        });
    }

    // Fetch and display problem details in the update form
    function fetchProblemDetails(problemId) {
        fetch(`${GET_PROBLEM_DETAILS_URL}${problemId}`)
            .then(response => response.json())
            .then(data => populateUpdateForm(data))
            .catch(error => console.error('Error fetching problem details:', error));
    }

    function populateUpdateForm(data) {
    if (!data || !data.problem || !data.position) {
        console.error('Invalid data received for problem details:', data);
        alert('Error loading problem details.');
        return;
    }

    // Set problem details
    document.getElementById('update_problem_id').value = data.problem.id || '';
    document.getElementById('update_problem_name').value = data.problem.name || '';
    document.getElementById('update_problem_description').value = data.problem.description || '';

    // References to update form elements
    const updateDropdowns = {
        area: document.getElementById('update_pst_areaDropdown'),
        equipmentGroup: document.getElementById('update_pst_equipmentGroupDropdown'),
        model: document.getElementById('update_pst_modelDropdown'),
        assetNumberInput: document.getElementById('update_pst_assetNumberInput'),
        locationInput: document.getElementById('update_pst_locationInput'),
        siteLocation: document.getElementById('update_pst_siteLocationDropdown')
    };

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
            })
            .catch(error => console.error('Error fetching equipment groups or models:', error));
    } else {
        console.error('Area ID is missing in position data.');
        alert('Area information is missing in problem details.');
    }

    // Show the update problem section
    const updateSection = document.getElementById('pst_updateProblemSection');
    if (updateSection) {
        updateSection.style.display = 'block';
    }
}


    // Fetch and populate Solutions Tab with related solutions for the selected problem
    function editRelatedSolutions(problemId) {
        fetch(`${GET_SOLUTIONS_URL}${problemId}`)
            .then(response => response.json())
            .then(data => {
                populateSolutionsTab(data);
                document.getElementById('solution-tab').click(); // Activate the Solutions tab
            })
            .catch(error => console.error('Error fetching related solutions:', error));
    }

    // Populate Solutions Tab
    function populateSolutionsTab(solutions) {
        const solutionsDropdown = document.getElementById('existing_solutions');
        solutionsDropdown.innerHTML = ''; // Clear previous options
        solutions.forEach(solution => {
            solutionsDropdown.innerHTML += `<option value="${solution.id}">${solution.name} - ${solution.description}</option>`;
        });
    }

    // Utility functions
    function populateDropdown(dropdown, data, placeholder) {
        dropdown.innerHTML = `<option value="">${placeholder}</option>`;
        data.forEach(item => {
            let displayText = item.name || `${item.title} - Room ${item.room_number}`;
            dropdown.innerHTML += `<option value="${item.id}">${displayText}</option>`;
        });
        dropdown.disabled = false;
    }

    function resetDropdown(dropdown, placeholder) {
        dropdown.innerHTML = `<option value="">${placeholder}</option>`;
        dropdown.disabled = true;
    }

    function resetAllDropdowns() {
        resetDropdown(dropdowns.equipmentGroup, 'Select Equipment Group');
        resetDropdown(dropdowns.model, 'Select Model');
        resetDropdown(dropdowns.siteLocation, 'Select Site Location');
    }
});
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
