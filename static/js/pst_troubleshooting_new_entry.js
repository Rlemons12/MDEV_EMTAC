$(document).ready(function () {
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

    // Utility Function to Reset Fields (excluding Site Location)
    function resetField(field, placeholder) {
        // Exclude Site Location Dropdown from being reset
        if (field.attr('id') === 'new_pst_siteLocationDropdown') {
            return; // Do not reset Site Location Dropdown
        }
        console.log(`Resetting field. Placeholder: ${placeholder}`);
        field.empty().append('<option value="">' + placeholder + '</option>');
        field.prop('disabled', true);
        field.val(null).trigger('change'); // Reset Select2 if applicable
    }

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

    // Fetch Site Locations on page load
    fetchSiteLocations();

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
            // Reset subsequent dropdowns
            resetField($('#new_pst_equipmentGroupDropdown'), 'Select Equipment Group');
            resetField($('#new_pst_modelDropdown'), 'Select Model');
            resetField($('#new_pst_assetNumberDropdown'), 'Select Asset Number');
            resetField($('#new_pst_locationDropdown'), 'Select Location');
            // Site Location is independent, no need to reset it here

            $.ajax({
                type: 'GET',
                url: '/pst_troubleshoot_new_entry/get_equipment_groups',
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
            // Reset Equipment Group Dropdown and subsequent fields
            resetField($('#new_pst_equipmentGroupDropdown'), 'Select Equipment Group');
            resetField($('#new_pst_modelDropdown'), 'Select Model');
            resetField($('#new_pst_assetNumberDropdown'), 'Select Asset Number');
            resetField($('#new_pst_locationDropdown'), 'Select Location');
            // Site Location is independent, no need to reset it here
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
            // Reset subsequent dropdowns
            resetField($('#new_pst_modelDropdown'), 'Select Model');
            resetField($('#new_pst_assetNumberDropdown'), 'Select Asset Number');
            resetField($('#new_pst_locationDropdown'), 'Select Location');
            // Site Location is independent, no need to reset it here

            $.ajax({
                type: 'GET',
                url: '/pst_troubleshoot_new_entry/get_models',
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
        } else {
            // Reset Model Dropdown and subsequent fields
            resetField($('#new_pst_modelDropdown'), 'Select Model');
            resetField($('#new_pst_assetNumberDropdown'), 'Select Asset Number');
            resetField($('#new_pst_locationDropdown'), 'Select Location');
            // Site Location is independent, no need to reset it here
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
            // Reset Asset Number and Location Dropdowns
            resetField($('#new_pst_assetNumberDropdown'), 'Select Asset Number');
            resetField($('#new_pst_locationDropdown'), 'Select Location');
            // Site Location is independent, no need to reset it here

            // Fetch Asset Numbers
            $.ajax({
                type: 'GET',
                url: '/pst_troubleshoot_new_entry/get_asset_numbers',
                data: { model_id: modelId },
                success: function (response) {
                    console.log('Received Asset Numbers:', response);
                    var assetNumberDropdown = $('#new_pst_assetNumberDropdown');
                    assetNumberDropdown.empty();
                    assetNumberDropdown.append('<option value="">Select Asset Number</option>');
                    $.each(response, function (index, assetNumber) {
                        assetNumberDropdown.append('<option value="' + assetNumber.id + '">' + assetNumber.number + '</option>');
                    });
                    assetNumberDropdown.prop('disabled', false);
                    assetNumberDropdown.val(null).trigger('change');
                },
                error: function (xhr, status, error) {
                    console.error('Error fetching Asset Numbers:', error);
                    showAlert('Error fetching Asset Numbers.', 'danger');
                }
            });

            // Fetch Locations
            $.ajax({
                type: 'GET',
                url: '/pst_troubleshoot_new_entry/get_locations',
                data: { model_id: modelId },
                success: function (response) {
                    console.log('Received Locations:', response);
                    var locationDropdown = $('#new_pst_locationDropdown');
                    locationDropdown.empty();
                    locationDropdown.append('<option value="">Select Location</option>');
                    $.each(response, function (index, location) {
                        locationDropdown.append('<option value="' + location.id + '">' + location.name + '</option>');
                    });
                    locationDropdown.prop('disabled', false);
                    locationDropdown.val(null).trigger('change');
                },
                error: function (xhr, status, error) {
                    console.error('Error fetching Locations:', error);
                    showAlert('Error fetching Locations.', 'danger');
                }
            });
        } else {
            // Reset Asset Number and Location Dropdowns
            resetField($('#new_pst_assetNumberDropdown'), 'Select Asset Number');
            resetField($('#new_pst_locationDropdown'), 'Select Location');
            // Site Location is independent, no need to reset it here
        }
    });

    function fetchSiteLocations() {
        console.log('Fetching all Site Locations on page load.');

        $.ajax({
            type: 'GET',
            url: '/pst_troubleshoot_new_entry/get_site_locations',
            success: function (response) {
                console.log('Received Site Locations:', response);
                var siteLocationDropdown = $('#new_pst_siteLocationDropdown');

                siteLocationDropdown.empty().append('<option value="">Select Site Location</option>');

                if (response.length === 0) {
                    siteLocationDropdown.append('<option value="">No Site Locations Available</option>');
                } else {
                    $.each(response, function (index, siteLocation) {
                        var optionText = siteLocation.title + ' - Room ' + siteLocation.room_number;
                        siteLocationDropdown.append('<option value="' + siteLocation.id + '">' + optionText + '</option>');
                    });
                }

                // Append "New Site Location" option
                siteLocationDropdown.append('<option value="new">New Site Location...</option>');
                siteLocationDropdown.prop('disabled', false);  // Enable dropdown if disabled

                // Initialize or refresh Select2
                siteLocationDropdown.select2({
                    placeholder: 'Select Site Location or type "New..."',
                    allowClear: true
                });

                console.log('Select2 initialized for Site Location Dropdown after options loaded.');
            },
            error: function (xhr, status, error) {
                console.error('Error fetching Site Locations:', error);
                showAlert('Error fetching Site Locations.', 'danger');
            }
        });
    }

    // Handle Form Submission via AJAX
$('#newProblemForm').on('submit', function (e) {
    e.preventDefault(); // Prevent default form submission
    console.log('New Problem Form submitted.');

    // Validate required fields before submission
    var problemName = $('#problemName').val(); // Corrected ID
    var problemDescription = $('#problemDescription').val(); // Corrected ID
    var areaId = $('#new_pst_areaDropdown').val();
    var equipmentGroupId = $('#new_pst_equipmentGroupDropdown').val();
    var modelId = $('#new_pst_modelDropdown').val();

    if (!problemName || !problemDescription || !areaId || !equipmentGroupId || !modelId) {
        showAlert('Name, Description, Area, Equipment Group, and Model are required.', 'warning');
        return;
    }

    // Proceed with AJAX submission since all required fields are filled
    var formData = $(this).serialize(); // Serialize form data
    console.log('Form Data:', formData);

    $.ajax({
        type: 'POST',
        url: '/pst_troubleshoot_new_entry/create_problem', // Update this to your actual route if different
        data: formData,
        success: function (response) {
            console.log('Create Problem Response:', response);
            if (response.success) {
                showAlert(response.message, 'success');
                // Optionally, redirect to the new problem's page after a delay
                setTimeout(function () {
                    window.location.href = '/pst_troubleshoot_new_entry/' + response.problem_id;
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


    // Utility Function to Reset Dropdowns (if you prefer to use this)
    function resetDropdown(dropdown, placeholder) {
        console.log(`Resetting dropdown. Placeholder: ${placeholder}`);
        dropdown.empty().append('<option value="">' + placeholder + '</option>');
        dropdown.prop('disabled', true);
        dropdown.val(null).trigger('change'); // Reset Select2
    }
});
