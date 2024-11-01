// static/js/pst_troubleshooting_solution.js

document.addEventListener('DOMContentLoaded', () => {
    'use strict';

    // Define the backend endpoint URLs
    const GET_SOLUTIONS_URL = '/pst_troubleshooting_solution/get_solutions/';
    const ADD_SOLUTION_URL = '/pst_troubleshooting_solution/add_solution/';
    const REMOVE_SOLUTIONS_URL = '/pst_troubleshooting_solution/remove_solutions/';

    // Variable to store the currently selected problem ID
    let currentProblemId = null;

    // Variable to store solutions to remove (used with confirmation modal)
    let solutionsToRemove = [];

    /**
     * Displays an alert message to the user.
     * @param {string} message - The message to display.
     * @param {string} category - The Bootstrap alert category (e.g., 'success', 'warning', 'danger').
     */
    function showAlert(message, category) {
        const alertContainer = document.getElementById('alertContainer');
        if (alertContainer) {
            const alertDiv = document.createElement('div');
            alertDiv.className = `alert alert-${category} alert-dismissible fade show`;
            alertDiv.role = 'alert';
            alertDiv.innerHTML = `
                ${message}
                <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
            `;
            alertContainer.appendChild(alertDiv);

            // Automatically remove the alert after 5 seconds
            setTimeout(() => {
                const alert = bootstrap.Alert.getInstance(alertDiv);
                if (alert) {
                    alert.close();
                }
            }, 5000);
        } else {
            console.error('Alert container not found.');
        }
    }

    /**
     * Displays a loading indicator to inform the user that an operation is in progress.
     */
    function showLoading() {
        const alertContainer = document.getElementById('alertContainer');
        if (alertContainer) {
            const loadingDiv = document.createElement('div');
            loadingDiv.className = `alert alert-info`;
            loadingDiv.role = 'alert';
            loadingDiv.textContent = 'Loading...';
            alertContainer.appendChild(loadingDiv);
        }
    }

    /**
     * Clears all alerts from the alert container.
     */
    function clearAlerts() {
        const alertContainer = document.getElementById('alertContainer');
        if (alertContainer) {
            alertContainer.innerHTML = '';
        }
    }

    /**
     * Updates the problem name displayed in the Solutions tab.
     * @param {string} problemName - The name of the selected problem.
     */
    function updateProblemName(problemName) {
        const problemNameHeader = document.getElementById('selected-problem-name');
        if (problemNameHeader) {
            problemNameHeader.textContent = `Problem Solutions for: ${problemName}`;
        } else {
            console.error('Problem name header element not found.');
        }
    }

    /**
     * Populates the Solutions dropdown with fetched solutions.
     * @param {Array} solutions - An array of solution objects.
     */
    function populateSolutionsTab(solutions) {
        const solutionsDropdown = document.getElementById('existing_solutions');
        if (!solutionsDropdown) {
            console.error('Solutions dropdown element not found.');
            return;
        }

        solutionsDropdown.innerHTML = ''; // Clear existing options
        console.log("Populating Solutions tab with data.");

        if (Array.isArray(solutions)) {
            solutions.forEach(function(solution) {
                console.log("Adding solution to dropdown:", solution);
                const option = document.createElement('option');
                option.value = solution.id;
                option.textContent = `${solution.name} - ${solution.description || 'No description provided.'}`;
                solutionsDropdown.appendChild(option);
            });

            console.log("Solutions dropdown populated.");
        } else {
            console.error('Invalid solutions data format:', solutions);
            showAlert('Invalid data format received for solutions.', 'danger');
        }
    }

    /**
     * Activates the Solutions tab in the UI.
     */
    function activateSolutionsTab() {
        const solutionTabLink = document.getElementById('solution-tab');
        if (solutionTabLink) {
            const tab = new bootstrap.Tab(solutionTabLink);
            tab.show();
        } else {
            console.error('Solution tab element not found.');
        }
    }

    /**
     * Fetches solutions related to a specific problem from the backend.
     * @param {number} problemId - The ID of the problem.
     */
    function fetchSolutions(problemId) {
        currentProblemId = problemId; // Update the current problem ID
        clearAlerts(); // Clear any existing alerts
        showLoading(); // Show loading indicator

        fetch(`${GET_SOLUTIONS_URL}${encodeURIComponent(problemId)}`)
            .then(response => {
                if (!response.ok) {
                    return response.json().then(errorData => {
                        throw new Error(errorData.error || 'Unknown error occurred.');
                    });
                }
                return response.json();
            })
            .then(response => {
                clearAlerts(); // Remove loading indicator
                console.log("Solutions fetched successfully:", response);
                populateSolutionsTab(response.solutions);
                updateProblemName(response.problem_name);
                activateSolutionsTab(); // Switch to the Solutions tab
            })
            .catch(error => {
                clearAlerts(); // Remove loading indicator
                showAlert('Error fetching solutions: ' + error.message, 'danger');
                console.error("Error fetching solutions:", error);
            });
    }

    /**
     * Adds a new solution to the selected problem by sending a POST request to the backend.
     * @param {number} problemId - The ID of the problem.
     * @param {string} solutionName - The name of the new solution.
     * @param {string} solutionDescription - The description of the new solution.
     */
    function addNewSolution(problemId, solutionName, solutionDescription) {
        fetch(ADD_SOLUTION_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
                // Removed 'X-CSRFToken' since CSRF protection is not implemented
            },
            body: JSON.stringify({
                problem_id: problemId,
                name: solutionName,
                description: solutionDescription
            })
        })
            .then(response => {
                if (!response.ok) {
                    return response.json().then(errorData => {
                        throw new Error(errorData.error || 'Unknown error occurred.');
                    });
                }
                return response.json();
            })
            .then(data => {
                console.log("Solution added successfully:", data);
                showAlert('Solution added successfully.', 'success');
                fetchSolutions(problemId); // Refresh the solutions list
                clearNewSolutionFields(); // Clear input fields
            })
            .catch(error => {
                showAlert('Error adding solution: ' + error.message, 'danger');
                console.error("Error adding solution:", error);
            });
    }

    /**
     * Removes selected solutions from the selected problem by sending a POST request to the backend.
     * @param {number} problemId - The ID of the problem.
     * @param {Array} solutionIds - An array of solution IDs to remove.
     */
    function removeSolutions(problemId, solutionIds) {
        fetch(REMOVE_SOLUTIONS_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
                // Removed 'X-CSRFToken' since CSRF protection is not implemented
            },
            body: JSON.stringify({
                problem_id: problemId,
                solution_ids: solutionIds
            })
        })
            .then(response => {
                if (!response.ok) {
                    return response.json().then(errorData => {
                        throw new Error(errorData.error || 'Unknown error occurred.');
                    });
                }
                return response.json();
            })
            .then(data => {
                console.log("Solutions removed successfully:", data);
                showAlert('Selected solutions removed successfully.', 'success');
                fetchSolutions(problemId); // Refresh the solutions list
            })
            .catch(error => {
                showAlert('Error removing solutions: ' + error.message, 'danger');
                console.error("Error removing solutions:", error);
            });
    }

    /**
     * Clears the input fields for adding a new solution.
     */
    function clearNewSolutionFields() {
        const solutionNameInput = document.getElementById('new_solution_name');
        const solutionDescriptionInput = document.getElementById('new_solution_description');

        if (solutionNameInput) {
            solutionNameInput.value = '';
        }

        if (solutionDescriptionInput) {
            solutionDescriptionInput.value = '';
        }
    }

    /**
     * Initializes event listeners for adding and removing solutions.
     */
    function initializeEventListeners() {
        // Add Solution Button
        const addSolutionBtn = document.getElementById('addSolutionBtn');
        if (addSolutionBtn) {
            addSolutionBtn.addEventListener('click', function () {
                const solutionName = document.getElementById('new_solution_name').value.trim();
                const solutionDescription = document.getElementById('new_solution_description') ?
                    document.getElementById('new_solution_description').value.trim() : '';

                if (solutionName === '') {
                    showAlert('Solution name cannot be empty.', 'warning');
                    return;
                }

                if (!currentProblemId) {
                    showAlert('No problem selected for adding solutions.', 'warning');
                    return;
                }

                addNewSolution(currentProblemId, solutionName, solutionDescription);
            });
        } else {
            console.error('Add Solution button not found.');
        }

        // Remove Solutions Button
        const removeSolutionsBtn = document.getElementById('removeSolutionsBtn');
        if (removeSolutionsBtn) {
            removeSolutionsBtn.addEventListener('click', function () {
                const solutionsDropdown = document.getElementById('existing_solutions');
                const selectedOptions = Array.from(solutionsDropdown.selectedOptions);

                if (selectedOptions.length === 0) {
                    showAlert('Please select at least one solution to remove.', 'warning');
                    return;
                }

                const solutionIds = selectedOptions.map(option => option.value);

                if (!currentProblemId) {
                    showAlert('No problem selected for removing solutions.', 'warning');
                    return;
                }

                // Store solutions to remove for the confirmation modal
                solutionsToRemove = solutionIds;

                // Show the confirmation modal
                const confirmModal = new bootstrap.Modal(document.getElementById('confirmModal'));
                confirmModal.show();
            });
        } else {
            console.error('Remove Solutions button not found.');
        }

        // Confirmation Modal Remove Button
        const confirmRemoveBtn = document.getElementById('confirmRemoveBtn');
        if (confirmRemoveBtn) {
            confirmRemoveBtn.addEventListener('click', function () {
                // Hide the modal
                const confirmModalEl = document.getElementById('confirmModal');
                const confirmModal = bootstrap.Modal.getInstance(confirmModalEl);
                confirmModal.hide();

                // Proceed to remove solutions
                if (solutionsToRemove && solutionsToRemove.length > 0 && currentProblemId) {
                    removeSolutions(currentProblemId, solutionsToRemove);
                }
            });
        } else {
            console.error('Confirm Remove button not found in modal.');
        }
    }

    // Initialize event listeners when the DOM is fully loaded
    initializeEventListeners();
});
