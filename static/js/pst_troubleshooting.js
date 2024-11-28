// Consolidated Script: pst_troubleshooting.js

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
        assetNumber: document.getElementById('pst_assetNumberDropdown'),
        location: document.getElementById('pst_locationDropdown'),
        siteLocation: document.getElementById('pst_siteLocationDropdown')
    };

    // Initialize AppState if it doesn't already exist
    window.AppState = window.AppState || {};

    // Ensure properties exist without overwriting existing values
    if (typeof window.AppState.currentProblemId === 'undefined') {
        window.AppState.currentProblemId = null;
    }
    if (typeof window.AppState.currentSolutionId === 'undefined') {
        window.AppState.currentSolutionId = null;
    }
    if (typeof window.AppState.selectedTaskId === 'undefined') {
        window.AppState.selectedTaskId = null;
    }

console.log('Initialized AppState in pst_troubleshooting.js:', window.AppState);


    console.log("JavaScript loaded and ready");

    // Fetch and populate Site Locations when the page loads
    fetchData(GET_SITE_LOCATIONS_URL)
        .then(data => populateDropdown(dropdowns.siteLocation, data, 'Select Site Location', item => `${item.title} - Room ${item.room_number}`))
        .catch(error => console.error('Error fetching site locations:', error));

    // Event listeners for dropdowns
    dropdowns.area.addEventListener('change', () => {
        const areaId = dropdowns.area.value;
        if (areaId) {
            fetchData(`${GET_EQUIPMENT_GROUPS_URL}?area_id=${areaId}`)
                .then(data => populateDropdown(dropdowns.equipmentGroup, data, 'Select Equipment Group', item => item.name))
                .catch(error => console.error('Error fetching equipment groups:', error));

            // Reset dependent dropdowns
            resetDropdown(dropdowns.model, 'Select Model');
            resetDropdown(dropdowns.assetNumber, 'Select Asset Number');
            resetDropdown(dropdowns.location, 'Select Location');
        } else {
            resetAllDropdowns();
        }
    });

    dropdowns.equipmentGroup.addEventListener('change', () => {
        const equipmentGroupId = dropdowns.equipmentGroup.value;
        if (equipmentGroupId) {
            fetchData(`${GET_MODELS_URL}?equipment_group_id=${equipmentGroupId}`)
                .then(data => populateDropdown(dropdowns.model, data, 'Select Model', item => item.name))
                .catch(error => console.error('Error fetching models:', error));

            // Reset dependent dropdowns
            resetDropdown(dropdowns.assetNumber, 'Select Asset Number');
            resetDropdown(dropdowns.location, 'Select Location');
        } else {
            resetDropdown(dropdowns.model, 'Select Model');
            resetDropdown(dropdowns.assetNumber, 'Select Asset Number');
            resetDropdown(dropdowns.location, 'Select Location');
        }
    });

    dropdowns.model.addEventListener('change', () => {
        const modelId = dropdowns.model.value;
        if (modelId) {
            fetchData(`${GET_ASSET_NUMBERS_URL}?model_id=${modelId}`)
                .then(data => populateDropdown(dropdowns.assetNumber, data, 'Select Asset Number', item => item.number))
                .catch(error => console.error('Error fetching asset numbers:', error));

            fetchData(`${GET_LOCATIONS_URL}?model_id=${modelId}`)
                .then(data => populateDropdown(dropdowns.location, data, 'Select Location', item => item.name))
                .catch(error => console.error('Error fetching locations:', error));

            // Do not touch the siteLocation dropdown here
        } else {
            resetDropdown(dropdowns.assetNumber, 'Select Asset Number');
            resetDropdown(dropdowns.location, 'Select Location');
        }
    });

    // Search button event listener
    document.getElementById('searchProblemByPositionBtn').addEventListener('click', () => {
        const searchParams = {
            area_id: dropdowns.area.value,
            equipment_group_id: dropdowns.equipmentGroup.value,
            model_id: dropdowns.model.value,
            asset_number: dropdowns.assetNumber.value.trim(),
            location: dropdowns.location.value.trim(),
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

    /**
     * Display search results with Update and Edit Solutions buttons
     * @param {Array} problems - Array of problem objects
     */
    function displaySearchResults(problems) {
        const resultsList = document.getElementById('pst_positionResultsList');
        resultsList.innerHTML = '';
        document.getElementById('pst_searchResults').style.display = problems.length ? 'block' : 'none';

        if (problems.length > 0) {
            window.AppState.currentProblemId = problems[0].id;
            sessionStorage.setItem('window.AppState.currentProblemId', window.AppState.currentProblemId);  // Store in sessionStorage
            console.log('Updated Current Problem ID:', window.AppState.currentProblemId);
        } else {
            window.AppState.currentProblemId = null;
            sessionStorage.removeItem('window.AppState.currentProblemId');  // Remove from sessionStorage if null
            console.log('No problems found. Resetting Current Problem ID:', window.AppState.currentProblemId);
        }

        problems.forEach(problem => {
            const listItem = document.createElement('li');
            listItem.classList.add('list-group-item');
            listItem.innerHTML = `
                <strong>${problem.name}</strong> - ${problem.description}
                <button class="btn btn-sm btn-warning float-end ms-2 update-problem-btn" data-problem-id="${problem.id}">Update Problem Position</button>
                <button class="btn btn-sm btn-info float-end edit-solutions-btn" data-problem-id="${problem.id}">Edit Related Solutions</button>
                <br>
                <!--<small>Area: ${problem.area}</small><br>
                <small>Equipment Group: ${problem.equipment_group}</small><br>
                <small>Model: ${problem.model}</small><br>
                <small>Asset Number: ${problem.asset_number}</small><br>
                <small>Location: ${problem.location}</small><br>
                <small>Site Location: ${problem.site_location}</small>-->
            `;
            resultsList.appendChild(listItem);
        });

        resultsList.addEventListener('click', (event) => {
            if (event.target.classList.contains('update-problem-btn')) {
                const problemId = event.target.dataset.problemId;
                console.log("Update Problem Position button clicked for problem ID:", problemId);
                fetchProblemDetails(problemId);
            }
            if (event.target.classList.contains('edit-solutions-btn')) {
                const problemId = event.target.dataset.problemId;
                console.log("Edit Related Solutions button clicked for problem ID:", problemId);
                window.AppState.currentProblemId = problemId;
                sessionStorage.setItem('window.AppState.currentProblemId', window.AppState.currentProblemId);  // Update sessionStorage here too
                editRelatedSolutions(problemId);
            }
        });
    }



    // Fetch and display problem details in the update form
    function fetchProblemDetails(problemId) {
        fetchData(`${GET_PROBLEM_DETAILS_URL}${problemId}`)
            .then(data => populateUpdateForm(data))
            .catch(error => console.error('Error fetching problem details:', error));
    }

    function populateUpdateForm(data) {
        if (!data || !data.problem || !data.position) {
            console.error('Invalid data received for problem details:', data);
            alert('Error loading problem details.');
            return;
        }

        document.getElementById('update_problem_id').value = data.problem.id || '';
        document.getElementById('update_problem_name').value = data.problem.name || '';
        document.getElementById('update_problem_description').value = data.problem.description || '';

        const updateDropdowns = {
            area: document.getElementById('update_pst_areaDropdown'),
            equipmentGroup: document.getElementById('update_pst_equipmentGroupDropdown'),
            model: document.getElementById('update_pst_modelDropdown'),
            assetNumber: document.getElementById('update_pst_assetNumberInput'),
            location: document.getElementById('update_pst_locationInput'),
            siteLocation: document.getElementById('update_pst_siteLocationDropdown')
        };

        if (updateDropdowns.area && data.position.area_id) {
            updateDropdowns.area.value = data.position.area_id;
            updateDropdowns.area.disabled = false;

            fetchData(`${GET_EQUIPMENT_GROUPS_URL}?area_id=${encodeURIComponent(data.position.area_id)}`)
                .then(equipmentGroups => {
                    populateDropdown(updateDropdowns.equipmentGroup, equipmentGroups, 'Select Equipment Group', item => item.name);
                    updateDropdowns.equipmentGroup.value = data.position.equipment_group_id || '';

                    return fetchData(`${GET_MODELS_URL}?equipment_group_id=${encodeURIComponent(data.position.equipment_group_id)}`);
                })
                .then(models => {
                    populateDropdown(updateDropdowns.model, models, 'Select Model', item => item.name);
                    updateDropdowns.model.value = data.position.model_id || '';

                    updateDropdowns.equipmentGroup.disabled = false;
                    updateDropdowns.model.disabled = false;

                    updateDropdowns.assetNumber.value = data.position.asset_number || '';
                    updateDropdowns.location.value = data.position.location || '';

                    // Site Location is independent; it was already fetched on page load
                })
                .catch(error => console.error('Error fetching equipment groups or models:', error));
        } else {
            console.error('Area ID is missing in position data.');
            alert('Area information is missing in problem details.');
        }

        const updateSection = document.getElementById('pst_updateProblemSection');
        if (updateSection) {
            updateSection.style.display = 'block';
        }
    }

    function editRelatedSolutions(problemId) {
        window.AppState.currentProblemId = problemId;  // Set the current problem ID here
        console.log('Current Problem ID updated to:', window.AppState.currentProblemId);
        fetchData(`${GET_SOLUTIONS_URL}${problemId}`)
            .then(data => {
                if (!data || !Array.isArray(data) || data.length === 0) {
                    console.log(`No related solutions found for problem ID: ${problemId}`);
                }
                populateSolutionsTab(data);
                document.getElementById('solution-tab').click();
            })
            .catch(error => console.error('Error fetching related solutions:', error));
    }


    function populateSolutionsTab(solutions) {
        const solutionsDropdown = document.getElementById('existing_solutions');
        solutionsDropdown.innerHTML = '';

        // Check if solutions is an array, otherwise assign an empty array
        if (!Array.isArray(solutions)) {
            solutions = [];
        }

        // Populate the dropdown with existing solutions
        solutions.forEach(solution => {
            solutionsDropdown.innerHTML += `<option value="${solution.id}">${solution.name} - ${solution.description}</option>`;
        });

        // Optionally, disable the dropdown if there are no solutions
        if (solutions.length === 0) {
            solutionsDropdown.disabled = true;
        } else {
            solutionsDropdown.disabled = false;
        }
    }


    // Utility functions
    function populateDropdown(dropdown, data, placeholder, displayTextFunc) {
        dropdown.innerHTML = `<option value="">${placeholder}</option>`;
        data.forEach(item => {
            const displayText = displayTextFunc(item);
            dropdown.innerHTML += `<option value="${item.id}">${displayText}</option>`;
        });
        dropdown.disabled = false;
    }

    function resetDropdown(dropdown, placeholder) {
        if (dropdown !== dropdowns.siteLocation) { // Do not reset Site Location dropdown
            dropdown.innerHTML = `<option value="">${placeholder}</option>`;
            dropdown.disabled = true;
        }
    }

    function resetAllDropdowns() {
        resetDropdown(dropdowns.equipmentGroup, 'Select Equipment Group');
        resetDropdown(dropdowns.model, 'Select Model');
        resetDropdown(dropdowns.assetNumber, 'Select Asset Number');
        resetDropdown(dropdowns.location, 'Select Location');
        // Site Location dropdown is independent; do not reset here
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
