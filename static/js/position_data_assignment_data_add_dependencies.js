$(document).ready(function () {
    // Function to toggle visibility of forms
    function toggleForm(formId) {
        var form = document.getElementById(formId);
        form.style.display = form.style.display === 'none' ? 'block' : 'none';
    }

    // Initially hide "Add Another" buttons
    $('#addAnotherAreaBtn').hide();
    $('#addAnotherEquipmentGroupBtn').hide();
    $('#addAnotherModelBtn').hide();
    $('#addAnotherAssetNumberBtn').hide();
    $('#addAnotherLocationBtn').hide();
    $('#addAnotherSiteLocationBtn').hide();

    // Add event listeners for form toggles
    $('#toggleNewAreaBtn').on('click', function () {
        toggleForm('newAreaForm');
    });

    $('#toggleNewEquipmentGroupBtn').on('click', function () {
        toggleForm('newEquipmentGroupForm');
    });

    $('#toggleNewModelBtn').on('click', function () {
        toggleForm('newModelForm');
    });

    $('#toggleNewAssetNumberBtn').on('click', function () {
        toggleForm('newAssetNumberForm');
    });

    $('#toggleNewLocationBtn').on('click', function () {
        toggleForm('newLocationForm');
    });

    $('#toggleNewSiteLocationBtn').on('click', function () {
        toggleForm('newSiteLocationForm');
    });

    // Fetch Equipment Groups when Area is selected
    $('#new_areaDropdown').on('change', function () {
        var areaId = $(this).val();
        if (areaId === "new") {
            $('#newAreaFields').show();
            $('#addAnotherAreaBtn').show();
        } else if (areaId) {
            $('#new_equipmentGroupDropdown').prop('disabled', false);
            $('#newAreaFields').hide();
            $('#addAnotherAreaBtn').hide();
            $.getJSON('/get_equipment_groups', { area_id: areaId }, function (data) {
                $('#new_equipmentGroupDropdown').empty().append('<option value="">Select Equipment Group</option>');
                $.each(data, function (index, group) {
                    $('#new_equipmentGroupDropdown').append('<option value="' + group.id + '">' + group.name + '</option>');
                });
                $('#new_equipmentGroupDropdown').append('<option value="new">New Equipment Group...</option>');
            }).fail(function () {
                alert('Error fetching equipment groups');
            });
        }
    });

    // Fetch Models when Equipment Group is selected
    $('#new_equipmentGroupDropdown').on('change', function () {
        var equipmentGroupId = $(this).val();
        if (equipmentGroupId === "new") {
            $('#newEquipmentGroupFields').show();
            $('#addAnotherEquipmentGroupBtn').show();
        } else if (equipmentGroupId) {
            $('#new_modelDropdown').prop('disabled', false);
            $('#newEquipmentGroupFields').hide();
            $('#addAnotherEquipmentGroupBtn').hide();
            $.getJSON('/get_models', { equipment_group_id: equipmentGroupId }, function (data) {
                $('#new_modelDropdown').empty().append('<option value="">Select Model</option>');
                $.each(data, function (index, model) {
                    $('#new_modelDropdown').append('<option value="' + model.id + '">' + model.name + '</option>');
                });
                $('#new_modelDropdown').append('<option value="new">New Model...</option>');
            }).fail(function () {
                alert('Error fetching models');
            });
        }
    });

    // Fetch Asset Numbers and Locations when Model is selected
    $('#new_modelDropdown').on('change', function () {
        var modelId = $(this).val();
        if (modelId === "new") {
            $('#newModelFields').show();
            $('#addAnotherModelBtn').show();
        } else if (modelId) {
            $('#new_assetNumberDropdown, #new_locationDropdown').prop('disabled', false);
            $('#newModelFields').hide();
            $('#addAnotherModelBtn').hide();

            // Fetch Asset Numbers
            $.getJSON('/get_asset_numbers', { model_id: modelId }, function (data) {
                $('#new_assetNumberDropdown').empty().append('<option value="">Select Asset Number</option>');
                $.each(data, function (index, asset) {
                    $('#new_assetNumberDropdown').append('<option value="' + asset.id + '">' + asset.number + '</option>');
                });
                $('#new_assetNumberDropdown').append('<option value="new">New Asset Number...</option>');
            }).fail(function () {
                alert('Error fetching asset numbers');
            });

            // Fetch Locations
            $.getJSON('/get_locations', { model_id: modelId }, function (data) {
                $('#new_locationDropdown').empty().append('<option value="">Select Location</option>');
                $.each(data, function (index, location) {
                    $('#new_locationDropdown').append('<option value="' + location.id + '">' + location.name + '</option>');
                });
                $('#new_locationDropdown').append('<option value="new">New Location...</option>');
            }).fail(function () {
                alert('Error fetching locations');
            });
        }
    });

    // Show new fields when "New..." is selected in Asset Number or Location dropdowns
    $('#new_assetNumberDropdown').on('change', function () {
        if ($(this).val() === 'new') {
            $('#newAssetNumberFields').show();
            $('#addAnotherAssetNumberBtn').show();
        } else {
            $('#newAssetNumberFields').hide();
            $('#addAnotherAssetNumberBtn').hide();
        }
    });

    $('#new_locationDropdown').on('change', function () {
        if ($(this).val() === 'new') {
            $('#newLocationFields').show();
            $('#addAnotherLocationBtn').show();
        } else {
            $('#newLocationFields').hide();
            $('#addAnotherLocationBtn').hide();
        }
    });

    // Initialize the Select2 plugin for searchable dropdown
    $('#new_siteLocationDropdown').select2({
        placeholder: 'Search, Select Site Location or type "New..',
        ajax: {
            url: '/search_site_locations',
            dataType: 'json',
            delay: 250,
            data: function (params) {
                return { search: params.term }; // Pass search term to the server
            },
            processResults: function (data) {
                // Include the "New Site Location" option in the search results
                data.unshift({ id: 'new', title: 'New Site Location', room_number: '' });

                return {
                    results: data.map(function (location) {
                        return { id: location.id, text: location.title + (location.room_number ? ' (Room: ' + location.room_number + ')' : '') };
                    })
                };
            },
            cache: true
        },
        minimumInputLength: 1
    });

    // Handle selection of "New Site Location"
    $('#new_siteLocationDropdown').on('change', function () {
        var selectedValue = $(this).val();
        console.log("Selected value:", selectedValue); // Debugging log
        if (selectedValue === 'new') {
            $('#newSiteLocationFields').show(); // Show fields to input new site location
            $('#addAnotherSiteLocationBtn').show(); // Show the button to add another site location
        } else {
            $('#newSiteLocationFields').hide(); // Hide the fields when other options are selected
            $('#addAnotherSiteLocationBtn').hide(); // Hide the button when other options are selected
        }
    });

    // Add new Site Location fields dynamically
    $('#addAnotherSiteLocationBtn').on('click', function () {
        var newSiteLocationHtml = `
            <div class="new-site-location-entry">
                <h4>New Site Location</h4>
                <label for="new_siteLocation_title">New Site Location Title:</label>
                <input type="text" name="new_siteLocation_title[]" required>
                <label for="new_siteLocation_roomNumber">Room Number:</label>
                <input type="text" name="new_siteLocation_room_number[]" required>
                <button type="button" class="remove-entry">Remove</button>
            </div>
        `;
        $('#siteLocationFieldsWrapper').append(newSiteLocationHtml);
    });

    // Remove dynamically added site location fields
    $(document).on('click', '.remove-entry', function () {
        $(this).parent().remove();
    });

    // Function to add new fields dynamically
    function addNewField(containerId, fieldHtml) {
        $(containerId).append(fieldHtml);
    }
});


    $('#addPositionDependenciesForm').on('submit', function (e) {
        e.preventDefault(); // prevent the form from refreshing the page

        var formData = $(this).serialize(); // get all form data

        $.ajax({
            type: 'POST',
            url: $(this).attr('action'), // get the form's action URL
            data: formData,
            success: function (response) {
                // handle success
                alert('Position successfully created with ID: ' + response.position_id);
                // Optionally reload or redirect
            },
            error: function (xhr) {
                // handle error
                alert('Error: ' + xhr.responseText);
            }
        });
    });

