// troubleshooting_guide_edit.js

// Function to search for problems to edit
function searchForProblem() {
    // Capture the search query and the dropdown values
    var searchQuery = $('#searchProblem').val();
    var areaId = $('#tsg_edit_areaDropdown').val();
    var equipmentGroupId = $('#tsg_edit_equipmentGroupDropdown').val();
    var modelId = $('#tsg_edit_modelDropdown').val();
    var assetNumberId = $('#tsg_edit_assetNumberDropdown').val();
    var locationId = $('#tsg_edit_locationDropdown').val();

    $.ajax({
        url: '/get_troubleshooting_guide_edit_data',  // The backend route
        method: 'GET',
        data: {
            query: searchQuery,               // Problem title or ID
            area: areaId,                     // Area ID from dropdown
            equipment_group: equipmentGroupId, // Equipment Group ID from dropdown
            model: modelId,                   // Model ID from dropdown
            asset_number: assetNumberId,       // Asset Number ID from dropdown
            location: locationId              // Location ID from dropdown
        },
        success: function(data) {
            // Clear previous results
            $('#searchResults').empty();

            // Populate the search results with problems
            data.problems.forEach(function(problem) {
                $('#searchResults').append(
                    '<li onclick="loadProblemSolution(' + problem.id + ')">' + problem.name + '</li>'
                );
            });
        },
        error: function(xhr) {
            alert('Failed to search for problem: ' + xhr.responseText);
        }
    });
}

// Function to load the selected problem into the form
function loadProblemSolution(problemId) {
    $.ajax({
        url: '/get_problem_solution_data/' + problemId,
        method: 'GET',
        success: function(data) {
            // Show the form for editing
            $('#editProblemSolutionForm').show();

            // Populate the form fields with the fetched data
            $('#edit_problem_id').val(data.problem.id);
            $('#edit_problem_name').val(data.problem.name);
            $('#edit_problem_description').val(data.problem.description);
            $('#edit_solution_id').val(data.solution.id);
            $('#edit_solution_description').val(data.solution.description);
            // Populate other fields as necessary
        },
        error: function(xhr) {
            alert('Failed to load problem/solution data: ' + xhr.responseText);
        }
    });
}
