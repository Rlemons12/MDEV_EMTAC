// pst_troubleshooting_solution.js


document.addEventListener('DOMContentLoaded', () => {
    'use strict';

    // Define backend endpoint URLs
    const GET_SOLUTIONS_URL = '/pst_troubleshooting_solution/get_solutions/';
    const ADD_SOLUTION_URL = '/pst_troubleshooting_solution/add_solution/';
    const REMOVE_SOLUTIONS_URL = '/pst_troubleshooting_solution/remove_solutions/';
    const GET_TASKS_URL_BASE = '/pst_troubleshooting_solution/get_tasks/';
    const ADD_TASK_URL = '/pst_troubleshooting_solution/add_task/';

    let currentProblemId = null;
    let currentSolutionId = null;
    let solutionsToRemove = [];

    function showAlert(message, category) {
        const alertContainer = document.getElementById('alertContainer');
        if (alertContainer) {
            const alertDiv = document.createElement('div');
            alertDiv.className = `alert alert-${category} alert-dismissible fade show`;
            alertDiv.innerHTML = `${message}
                <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>`;
            alertContainer.appendChild(alertDiv);
            setTimeout(() => alertDiv.remove(), 5000);
        }
    }

    function fetchSolutions(problemId) {
        currentProblemId = problemId;
        showAlert('Loading solutions...', 'info');

        fetch(`${GET_SOLUTIONS_URL}${encodeURIComponent(problemId)}`)
            .then(response => response.ok ? response.json() : Promise.reject(response.json()))
            .then(response => {
                populateSolutionsDropdown(response.solutions);
                updateProblemName(response.problem_name);
                activateTab('solution-tab');
            })
            .catch(error => showAlert('Error fetching solutions: ' + error.message, 'danger'));
    }

    function populateSolutionsDropdown(solutions) {
        const solutionsDropdown = document.getElementById('existing_solutions');
        solutionsDropdown.innerHTML = solutions.map(solution =>
            `<option value="${solution.id}">${solution.name} - ${solution.description || 'No description provided.'}</option>`
        ).join('');
    }

    function updateProblemName(problemName) {
        const header = document.getElementById('selected-problem-name');
        if (header) header.textContent = `Problem Solutions for: ${problemName}`;
    }

    function activateTab(tabId) {
        const tabLink = document.getElementById(tabId);
        if (tabLink) new bootstrap.Tab(tabLink).show();
    }

    function addNewSolution(problemId, solutionName, solutionDescription) {
        fetch(ADD_SOLUTION_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ problem_id: problemId, name: solutionName, description: solutionDescription })
        })
            .then(response => response.ok ? response.json() : Promise.reject(response.json()))
            .then(() => {
                showAlert('Solution added successfully.', 'success');
                fetchSolutions(problemId);
                clearInputFields('new_solution_name', 'new_solution_description');
            })
            .catch(error => showAlert('Error adding solution: ' + error.message, 'danger'));
    }

    function removeSolutions(problemId, solutionIds) {
        fetch(REMOVE_SOLUTIONS_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ problem_id: problemId, solution_ids: solutionIds })
        })
            .then(response => response.ok ? response.json() : Promise.reject(response.json()))
            .then(() => {
                showAlert('Selected solutions removed successfully.', 'success');
                fetchSolutions(problemId);
            })
            .catch(error => showAlert('Error removing solutions: ' + error.message, 'danger'));
    }

    function fetchTasksForSolution(solutionId) {
    if (!solutionId) {
        console.warn("Invalid solutionId passed to fetchTasksForSolution.");
        return;
    }

    fetch(`${GET_TASKS_URL_BASE}${encodeURIComponent(solutionId)}`)
        .then(response => {
            if (!response.ok) {
                return response.json().then(errorData => {
                    throw new Error(errorData.error || `HTTP error! Status: ${response.status}`);
                });
            }
            return response.json();
        })
        .then(data => {
            if (data && data.tasks) {
                populateTasksDropdown(data.tasks);
                activateTab('task-tab');
            } else {
                showAlert('No tasks found for this solution.', 'info');
                populateTasksDropdown([]); // Clear dropdown if no tasks
            }
        })
        .catch(error => {
            console.error('Error fetching tasks:', error);
            showAlert(`Error fetching tasks: ${error.message || 'Unknown error occurred'}`, 'danger');
            activateTab('task-tab');
        });
}


    // Populate tasks in a dropdown list instead of a list-group
    function populateTasksDropdown(tasks) {
        const tasksDropdown = document.getElementById('existing_tasks');
        tasksDropdown.innerHTML = tasks.map(task =>
            `<option value="${task.id}">${task.name} - ${task.description || 'No description provided.'}</option>`
        ).join('');

        // Attach change event listener to open task details
        tasksDropdown.addEventListener('change', (event) => {
            const selectedTaskId = tasksDropdown.value;
            if (selectedTaskId) openTaskDetails(selectedTaskId);
        });
    }

    function openTaskDetails(taskId) {
        fetch(`/pst_troubleshooting_task/get_task_details/${taskId}`)
            .then(response => response.ok ? response.json() : Promise.reject(response.json()))
            .then(data => {
                if (data && data.task) {
                    populateEditTaskForm(data.task);
                    activateTab('edit-task-tab');
                } else {
                    clearEditTaskForm();
                }
            })
            .catch(error => showAlert('Error loading task details: ' + error.message, 'danger'));
    }

    function populateEditTaskForm(task) {
        document.getElementById('pst_task_edit_task_name').value = task.name || '';
        document.getElementById('pst_task_edit_task_description').value = task.description || '';
        document.getElementById('edit_task_id').value = task.id;
    }

    function clearEditTaskForm() {
        document.getElementById('pst_task_edit_task_name').value = '';
        document.getElementById('pst_task_edit_task_description').value = '';
        document.getElementById('edit_task_id').value = '';
    }

    function clearInputFields(...fieldIds) {
        fieldIds.forEach(id => document.getElementById(id).value = '');
    }

    function initializeEventListeners() {
        document.getElementById('addSolutionBtn')?.addEventListener('click', () => {
            const name = document.getElementById('new_solution_name').value.trim();
            const description = document.getElementById('new_solution_description')?.value.trim();
            if (name && currentProblemId) addNewSolution(currentProblemId, name, description);
            else showAlert('Solution name cannot be empty or no problem selected.', 'warning');
        });

        document.getElementById('removeSolutionsBtn')?.addEventListener('click', () => {
            const selectedOptions = Array.from(document.getElementById('existing_solutions').selectedOptions);
            if (selectedOptions.length && currentProblemId) {
                solutionsToRemove = selectedOptions.map(option => option.value);
                new bootstrap.Modal(document.getElementById('confirmModal')).show();
            } else showAlert('Please select at least one solution to remove.', 'warning');
        });

        document.getElementById('confirmRemoveBtn')?.addEventListener('click', () => {
            new bootstrap.Modal(document.getElementById('confirmModal')).hide();
            if (solutionsToRemove.length && currentProblemId) removeSolutions(currentProblemId, solutionsToRemove);
        });

        document.getElementById('existing_solutions')?.addEventListener('change', (event) => {
            const solutionId = event.target.value;
            if (solutionId) fetchTasksForSolution(solutionId);
        });

        document.getElementById('addTaskBtn')?.addEventListener('click', () => {
            const name = document.getElementById('new_task_name').value.trim();
            const description = document.getElementById('new_task_description').value.trim();
            if (name && currentSolutionId) addNewTask(currentSolutionId, name, description);
            else showAlert('Task name cannot be empty or no solution selected.', 'warning');
        });
    }

    initializeEventListeners();
});
