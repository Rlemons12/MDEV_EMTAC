// static/js/pst_troubleshoot_new_entry.js

$(document).ready(function () {
    // Initialize Select2 for Site Location Dropdown
    $('#new_pst_siteLocationDropdown').select2({
        placeholder: 'Select Site Location or type "New..."',
        allowClear: true
    });
    console.log('Select2 initialized for Site Location Dropdown.');

    // Function to toggle visibility of new Site Location fields
    function toggleNewSiteLocationFields(selectedValue) {
        console.log(`Toggling Site Location Fields based on selected value: ${selectedValue}`);
        if (selectedValue === 'new') {
            $('#newSiteLocationFields').removeClass('d-none');
            $('#new_siteLocation_title').prop('required', true);
            $('#new_siteLocation_room_number').prop('required', true);
        } else {
            $('#newSiteLocationFields').addClass('d-none');
            $('#new_siteLocation_title').prop('required', false);
            $('#new_siteLocation_room_number').prop('required', false);
        }
    }

    // Handle Site Location Dropdown Change
    $('#new_pst_siteLocationDropdown').on('change', function () {
        var selectedValue = $(this).val();
        console.log(`Site Location Dropdown changed to: ${selectedValue}`);
        toggleNewSiteLocationFields(selectedValue);
    });

    // Fetch Equipment Groups when Area is selected
    $('#new_pst_areaDropdown').on('change', function () {
        var areaId = $(this).val();
        console.log(`Area Dropdown changed to: ${areaId}`);
        if (areaId) {
            $.ajax({
                type: 'GET',
                url: '/pst_troubleshoot_new_entry/get_equipment_groups',  // Ensure this route exists
                data: { area_id: areaId },
                success: function (response) {
                    console.log('Received Equipment Groups:', response);
                    var equipmentGroupDropdown = $('#new_pst_equipmentGroupDropdown');
                    equipmentGroupDropdown.empty();
                    equipmentGroupDropdown.append('<option value="">Select Equipment Group</option>');
                    $.each(response, function (index, equipmentGroup) {
                        equipmentGroupDropdown.append('<option value="' + equipmentGroup.id + '">' + equipmentGroup.name + '</option>');
                    });
                    equipmentGroupDropdown.append('<option value="new">New Equipment Group...</option>');
                    equipmentGroupDropdown.prop('disabled', false);
                },
                error: function (xhr, status, error) {
                    console.error('Error fetching Equipment Groups:', error);
                    showAlert('Error fetching Equipment Groups.', 'danger');
                }
            });
        } else {
            // Reset Equipment Group Dropdown
            resetDropdown($('#new_pst_equipmentGroupDropdown'), 'Select Equipment Group');
            // Also reset subsequent dropdowns
            resetDropdown($('#new_pst_modelDropdown'), 'Select Model');
            resetDropdown($('#new_pst_siteLocationDropdown'), 'Select Site Location');
        }
    });

    // Fetch Models when Equipment Group is selected
    $('#new_pst_equipmentGroupDropdown').on('change', function () {
        var equipmentGroupId = $(this).val();
        console.log(`Equipment Group Dropdown changed to: ${equipmentGroupId}`);
        if (equipmentGroupId === 'new') {
            // Handle creation of new Equipment Group if needed
            alert('Redirecting to create a new Equipment Group.');
            window.location.href = '/dependencies/add_equipment_group';  // Adjust URL as needed
        } else if (equipmentGroupId) {
            $.ajax({
                type: 'GET',
                url: '/pst_troubleshoot_new_entry/get_models',  // Ensure this route exists
                data: { equipment_group_id: equipmentGroupId },
                success: function (response) {
                    console.log('Received Models:', response);
                    var modelDropdown = $('#new_pst_modelDropdown');
                    modelDropdown.empty();
                    modelDropdown.append('<option value="">Select Model</option>');
                    $.each(response, function (index, model) {
                        modelDropdown.append('<option value="' + model.id + '">' + model.name + '</option>');
                    });
                    modelDropdown.append('<option value="new">New Model...</option>');
                    modelDropdown.prop('disabled', false);
                },
                error: function (xhr, status, error) {
                    console.error('Error fetching Models:', error);
                    showAlert('Error fetching Models.', 'danger');
                }
            });
        }
    });

    // Fetch Asset Numbers and Locations when Model is selected
    $('#new_pst_modelDropdown').on('change', function () {
        var modelId = $(this).val();
        console.log(`Model Dropdown changed to: ${modelId}`);
        if (modelId === 'new') {
            // Handle creation of new Model if needed
            alert('Redirecting to create a new Model.');
            window.location.href = '/dependencies/add_model';  // Adjust URL as needed
        } else if (modelId) {
            // Fetch Asset Numbers
            $.ajax({
                type: 'GET',
                url: '/pst_troubleshoot_new_entry/get_asset_numbers',  // Ensure this route exists
                data: { model_id: modelId },
                success: function (response) {
                    console.log('Received Asset Numbers:', response);
                    $('#new_pst_assetNumberInput').val(''); // Clear existing input
                    // Optionally, implement autocomplete for Asset Numbers
                },
                error: function (xhr, status, error) {
                    console.error('Error fetching Asset Numbers:', error);
                    showAlert('Error fetching Asset Numbers.', 'danger');
                }
            });

            // Fetch Locations
            $.ajax({
                type: 'GET',
                url: '/pst_troubleshoot_new_entry/get_locations',  // Ensure this route exists
                data: { model_id: modelId },
                success: function (response) {
                    console.log('Received Locations:', response);
                    $('#new_pst_locationInput').val(''); // Clear existing input
                    // Optionally, implement autocomplete for Locations
                },
                error: function (xhr, status, error) {
                    console.error('Error fetching Locations:', error);
                    showAlert('Error fetching Locations.', 'danger');
                }
            });
        }
    });

    // Handle Form Submission via AJAX
    $('#newProblemForm').on('submit', function (e) {
        e.preventDefault(); // Prevent default form submission
        console.log('New Problem Form submitted.');

        // Show loading spinner or disable submit button if desired

        var formData = $(this).serialize(); // Serialize form data
        console.log('Form Data:', formData);

        $.ajax({
            type: 'POST',
            url: '/pst_troubleshoot_new_entry/create_problem',  // Ensure this route exists
            data: formData,
            success: function (response) {
                console.log('Create Problem Response:', response);
                if (response.success) {
                    showAlert(response.message, 'success');
                    // Optionally, redirect to the new problem's page after a delay
                    setTimeout(function () {
                        window.location.href = '/pst_troubleshoot_new_entry/' + response.problem_id;  // Adjust URL as needed
                    }, 2000);
                } else {
                    showAlert(response.message, 'warning');
                }
            },
            error: function (xhr, status, error) {
                console.error('Error creating Problem:', error);
                var errorMessage = 'An error occurred while creating the problem.';
                if (xhr.responseJSON && xhr.responseJSON.message) {
                    errorMessage = xhr.responseJSON.message;
                }
                showAlert('Error: ' + errorMessage, 'danger');
            }
        });
    });

    // Function to display Bootstrap alerts
    function showAlert(message, category) {
        var alertHtml = `
            <div class="alert alert-${category} alert-dismissible fade show" role="alert">
                ${message}
                <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
            </div>
        `;
        $('#alertContainer').html(alertHtml);
    }

    // Utility Function to Reset Dropdowns
    function resetDropdown(dropdown, placeholder) {
        console.log(`Resetting dropdown. Placeholder: ${placeholder}`);
        dropdown.empty().append('<option value="">' + placeholder + '</option>');
        dropdown.prop('disabled', true);
    }
});
