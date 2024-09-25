$(document).ready(function() {
    // Function to reset and disable dropdowns
    function resetDropdowns(selectors) {
        selectors.forEach(function(selector) {
            $(selector).prop('disabled', true).html('<option value="">Select...</option>');
        });
    }

    // Area change event
    $('#pda_areaDropdown').change(function() {
        var areaId = $(this).val();
        if (areaId) {
            $('#pda_equipmentGroupDropdown').prop('disabled', false);
            // Fetch Equipment Groups
            $.ajax({
                url: getEquipmentGroupsUrl, // URL passed from the HTML template
                method: 'GET',
                data: { area_id: areaId },
                success: function(data) {
                    var equipmentGroupDropdown = $('#pda_equipmentGroupDropdown');
                    equipmentGroupDropdown.empty();
                    equipmentGroupDropdown.append('<option value="">Select Equipment Group</option>');
                    $.each(data, function(index, group) {
                        equipmentGroupDropdown.append('<option value="' + group.id + '">' + group.name + '</option>');
                    });
                },
                error: function(xhr, status, error) {
                    console.error("Error fetching equipment groups:", error);
                    alert("An error occurred while fetching equipment groups.");
                }
            });
        } else {
            resetDropdowns(['#pda_equipmentGroupDropdown', '#pda_modelDropdown', '#pda_assetNumberDropdown', '#pda_locationDropdown', '#pda_siteLocation', '#pda_position']);
        }
    });

    // Equipment Group change event
    $('#pda_equipmentGroupDropdown').change(function() {
        var equipmentGroupId = $(this).val();
        if (equipmentGroupId) {
            $('#pda_modelDropdown').prop('disabled', false);
            // Fetch Models
            $.ajax({
                url: getModelsUrl, // URL passed from the HTML template
                method: 'GET',
                data: { equipment_group_id: equipmentGroupId },
                success: function(data) {
                    var modelDropdown = $('#pda_modelDropdown');
                    modelDropdown.empty();
                    modelDropdown.append('<option value="">Select Model</option>');
                    $.each(data, function(index, model) {
                        modelDropdown.append('<option value="' + model.id + '">' + model.name + '</option>');
                    });
                },
                error: function(xhr, status, error) {
                    console.error("Error fetching models:", error);
                    alert("An error occurred while fetching models.");
                }
            });
        } else {
            resetDropdowns(['#pda_modelDropdown', '#pda_assetNumberDropdown', '#pda_locationDropdown', '#pda_siteLocation', '#pda_position']);
        }
    });

    // Model change event
    $('#pda_modelDropdown').change(function() {
        var modelId = $(this).val();
        if (modelId) {
            $('#pda_assetNumberDropdown').prop('disabled', false);
            $('#pda_locationDropdown').prop('disabled', false);
            // Fetch Asset Numbers
            $.ajax({
                url: getAssetNumbersUrl, // URL passed from the HTML template
                method: 'GET',
                data: { model_id: modelId },
                success: function(data) {
                    var assetNumberDropdown = $('#pda_assetNumberDropdown');
                    assetNumberDropdown.empty();
                    assetNumberDropdown.append('<option value="">Select Asset Number</option>');
                    $.each(data, function(index, assetNumber) {
                        assetNumberDropdown.append('<option value="' + assetNumber.id + '">' + assetNumber.number + '</option>');
                    });
                },
                error: function(xhr, status, error) {
                    console.error("Error fetching asset numbers:", error);
                    alert("An error occurred while fetching asset numbers.");
                }
            });
            // Fetch Locations
            $.ajax({
                url: getLocationsUrl, // URL passed from the HTML template
                method: 'GET',
                data: { model_id: modelId },
                success: function(data) {
                    var locationDropdown = $('#pda_locationDropdown');
                    locationDropdown.empty();
                    locationDropdown.append('<option value="">Select Location</option>');
                    $.each(data, function(index, location) {
                        locationDropdown.append('<option value="' + location.id + '">' + location.name + '</option>');
                    });
                },
                error: function(xhr, status, error) {
                    console.error("Error fetching locations:", error);
                    alert("An error occurred while fetching locations.");
                }
            });
        } else {
            resetDropdowns(['#pda_assetNumberDropdown', '#pda_locationDropdown', '#pda_siteLocation', '#pda_position']);
        }
    });

    // Asset Number or Location change event
    $('#pda_assetNumberDropdown, #pda_locationDropdown').change(function() {
        var assetNumberId = $('#pda_assetNumberDropdown').val();
        var locationId = $('#pda_locationDropdown').val();
        var modelId = $('#pda_modelDropdown').val();

        if (assetNumberId && locationId) {
            $('#pda_siteLocation').prop('disabled', false);
            // Fetch Site Locations
            $.ajax({
                url: getSiteLocationsUrl, // URL passed from the HTML template
                method: 'GET',
                data: {
                    model_id: modelId,
                    asset_number_id: assetNumberId,
                    location_id: locationId
                },
                success: function(data) {
                    var siteLocationDropdown = $('#pda_siteLocation');
                    siteLocationDropdown.empty();
                    siteLocationDropdown.append('<option value="">Select Site Location</option>');
                    $.each(data, function(index, siteLocation) {
                        siteLocationDropdown.append('<option value="' + siteLocation.id + '">' + siteLocation.title + ' - ' + siteLocation.room_number + '</option>');
                    });
                },
                error: function(xhr, status, error) {
                    console.error("Error fetching site locations:", error);
                    alert("An error occurred while fetching site locations.");
                }
            });
        } else {
            resetDropdowns(['#pda_siteLocation', '#pda_position']);
        }
    });

    // Site Location change event
    $('#pda_siteLocation').change(function() {
        var siteLocationId = $(this).val();
        var assetNumberId = $('#pda_assetNumberDropdown').val();
        var locationId = $('#pda_locationDropdown').val();

        if (siteLocationId) {
            $('#pda_position').prop('disabled', false);
            // Fetch Positions
            $.ajax({
                url: getPositionsUrl, // URL passed from the HTML template
                method: 'GET',
                data: {
                    site_location_id: siteLocationId,
                    asset_number_id: assetNumberId,
                    location_id: locationId
                },
                success: function(data) {
                    var positionDropdown = $('#pda_position');
                    positionDropdown.empty();
                    positionDropdown.append('<option value="">Select Position</option>');
                    $.each(data, function(index, position) {
                        positionDropdown.append('<option value="' + position.id + '">' + position.name + '</option>');
                    });
                },
                error: function(xhr, status, error) {
                    console.error("Error fetching positions:", error);
                    alert("An error occurred while fetching positions.");
                }
            });
        } else {
            resetDropdowns(['#pda_position']);
        }
    });

    // Allow manual input for Asset Number and Location
    $('#pda_assetNumberInput').on('input', function() {
        var assetNumber = $(this).val();
        if (assetNumber.length > 1) {
            $('#pda_assetNumberDropdown').prop('disabled', true); // Disable dropdown when typing manually
        } else {
            $('#pda_assetNumberDropdown').prop('disabled', false); // Enable dropdown if input is cleared
        }
    });

    $('#pda_locationInput').on('input', function() {
        var location = $(this).val();
        if (location.length > 1) {
            $('#pda_locationDropdown').prop('disabled', true); // Disable dropdown when typing manually
        } else {
            $('#pda_locationDropdown').prop('disabled', false); // Enable dropdown if input is cleared
        }
    });

    // Functions to manage parts entries
    function addPartEntry() {
        var container = document.getElementById('parts-container');
        var newEntry = document.createElement('div');
        newEntry.className = 'part-entry';
        newEntry.innerHTML = `
            <label>Part Number:</label>
            <input type="text" name="part_numbers[]" required>
            <button type="button" onclick="removePartEntry(this)">Remove</button>
        `;
        container.appendChild(newEntry);
    }

    function removePartEntry(button) {
        button.parentElement.remove();
    }

    // Optional: Initialize Select2 for enhanced dropdowns
    $('#pda_areaDropdown, #pda_equipmentGroupDropdown, #pda_model
