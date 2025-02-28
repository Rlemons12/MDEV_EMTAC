$(document).ready(function () {
    //------------------------------------------------
    // 1) Hide "Add Another" Buttons Initially
    //------------------------------------------------
    $('#addAnotherAreaBtn').hide();
    $('#addAnotherEquipmentGroupBtn').hide();
    $('#addAnotherModelBtn').hide();
    $('#addAnotherAssetNumberBtn').hide();
    $('#addAnotherLocationBtn').hide();
    $('#addAnotherSiteLocationBtn').hide();
    // NEW: for Subassembly, Subassembly, Subassembly View
    $('#addAnotherAssemblyBtn').hide();
    $('#addAnotherSubassemblyBtn').hide();
    $('#addAnotherAssemblyViewBtn').hide();

    //------------------------------------------------
    // 2) Toggle/Hide "New..." Sections with Buttons
    //------------------------------------------------
    // (Only relevant if you have hidden sub-forms toggled by some button,
    //  but your snippet shows minimal usage, so we keep them.)
    function toggleForm(formId) {
        var form = document.getElementById(formId);
        form.style.display = (form.style.display === 'none') ? 'block' : 'none';
    }

    // Example toggles — if you have these buttons in your HTML
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

    //------------------------------------------------
    // 3) AREA → EQUIPMENT GROUP → MODEL → ASSET # → LOCATION
    //------------------------------------------------
    // (Already in your snippet. We just keep it.)
    // A) Area change => fetch Equipment Groups
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
                $('#new_equipmentGroupDropdown')
                    .empty()
                    .append('<option value="">Select Equipment Group</option>');
                $.each(data, function (index, group) {
                    $('#new_equipmentGroupDropdown')
                        .append('<option value="' + group.id + '">' + group.name + '</option>');
                });
                $('#new_equipmentGroupDropdown').append('<option value="new">New Equipment Group...</option>');
            }).fail(function () {
                alert('Error fetching equipment groups');
            });
        }
    });

    // B) Equipment Group => fetch Models
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
                $('#new_modelDropdown')
                    .empty()
                    .append('<option value="">Select Model</option>');
                $.each(data, function (index, model) {
                    $('#new_modelDropdown')
                        .append('<option value="' + model.id + '">' + model.name + '</option>');
                });
                $('#new_modelDropdown').append('<option value="new">New Model...</option>');
            }).fail(function () {
                alert('Error fetching models');
            });
        }
    });

    // C) Model => fetch Asset Numbers & Locations
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
                $('#new_assetNumberDropdown')
                    .empty()
                    .append('<option value="">Select Asset Number</option>');
                $.each(data, function (index, asset) {
                    $('#new_assetNumberDropdown')
                        .append('<option value="' + asset.id + '">' + asset.number + '</option>');
                });
                $('#new_assetNumberDropdown').append('<option value="new">New Asset Number...</option>');
            }).fail(function () {
                alert('Error fetching asset numbers');
            });

            // Fetch Locations
            $.getJSON('/get_locations', { model_id: modelId }, function (data) {
                $('#new_locationDropdown')
                    .empty()
                    .append('<option value="">Select Location</option>');
                $.each(data, function (index, location) {
                    $('#new_locationDropdown')
                        .append('<option value="' + location.id + '">' + location.name + '</option>');
                });
                $('#new_locationDropdown').append('<option value="new">New Location...</option>');
            }).fail(function () {
                alert('Error fetching locations');
            });
        }
    });

    // D) Show new fields for "New Asset Number" or "New Location"
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

    //------------------------------------------------
    // 4) SITE LOCATION (Select2)
    //------------------------------------------------
    $('#new_siteLocationDropdown').select2({
        placeholder: 'Search, Select Site Location or type "New.."',
        ajax: {
            url: '/search_site_locations',
            dataType: 'json',
            delay: 250,
            data: function (params) {
                return { search: params.term };
            },
            processResults: function (data) {
                // Insert a "new" option at the top
                data.unshift({ id: 'new', title: 'New Site Location', room_number: '' });
                return {
                    results: data.map(function (location) {
                        var label = location.title;
                        if (location.room_number) {
                            label += ' (Room: ' + location.room_number + ')';
                        }
                        return { id: location.id, text: label };
                    })
                };
            },
            cache: true
        },
        minimumInputLength: 1
    });

    // If "New Site Location" is chosen
    $('#new_siteLocationDropdown').on('change', function () {
        var selectedValue = $(this).val();
        if (selectedValue === 'new') {
            $('#newSiteLocationFields').show();
            $('#addAnotherSiteLocationBtn').show();
        } else {
            $('#newSiteLocationFields').hide();
            $('#addAnotherSiteLocationBtn').hide();
        }
    });

    // Add new site location fields dynamically
    $('#addAnotherSiteLocationBtn').on('click', function () {
        var newSiteLocationHtml = `
            <div class="new-site-location-entry">
                <h4>New Site Location</h4>
                <label>New Site Location Title:</label>
                <input type="text" name="new_siteLocation_title[]" required>
                <label>Room Number:</label>
                <input type="text" name="new_siteLocation_room_number[]" required>
                <button type="button" class="remove-entry">Remove</button>
            </div>
        `;
        $('#siteLocationFieldsWrapper').append(newSiteLocationHtml);
    });

    // Remove extra site location fields
    $(document).on('click', '.remove-entry', function () {
        $(this).parent().remove();
    });

    //------------------------------------------------
    // 5) ASSEMBLY → SUBASSEMBLY → ASSEMBLY VIEW
    //------------------------------------------------
    // A) Subassembly change => fetch Subassemblies
    $('#new_assemblyDropdown').on('change', function () {
        var assemblyId = $(this).val();

        if (assemblyId === 'new') {
            // Show "New Subassembly" fields
            $('#newAssemblyFields').show();
            $('#addAnotherAssemblyBtn').show();

            // Disable Subassembly & AssemblyView
            $('#new_subassemblyDropdown').prop('disabled', true).val('');
            $('#new_assemblyViewDropdown').prop('disabled', true).val('');
        }
        else if (assemblyId) {
            $('#newAssemblyFields').hide();
            $('#addAnotherAssemblyBtn').hide();

            // Enable subassembly dropdown
            $('#new_subassemblyDropdown').prop('disabled', false);
            $('#newSubassemblyFields').hide();
            $('#addAnotherSubassemblyBtn').hide();

            // Clear & fetch subassemblies
            $.getJSON('/component_assemblies', { assembly_id: assemblyId }, function (data) {
                $('#new_subassemblyDropdown')
                    .empty()
                    .append('<option value="">Select Subassembly</option>');

                $.each(data, function (index, subasm) {
                    $('#new_subassemblyDropdown')
                        .append('<option value="' + subasm.id + '">' + subasm.name + '</option>');
                });
                // Add "New Subassembly..."
                $('#new_subassemblyDropdown').append('<option value="new">New Subassembly...</option>');

                // Also disable assembly view until subassembly chosen
                $('#new_assemblyViewDropdown').prop('disabled', true).val('');
            }).fail(function () {
                alert('Error fetching subassemblies');
            });
        }
        else {
            // If user selected nothing or cleared it
            $('#newAssemblyFields').hide();
            $('#addAnotherAssemblyBtn').hide();
            $('#new_subassemblyDropdown').prop('disabled', true).val('');
            $('#newSubassemblyFields').hide();
            $('#addAnotherSubassemblyBtn').hide();
            $('#new_assemblyViewDropdown').prop('disabled', true).val('');
            $('#newAssemblyViewFields').hide();
            $('#addAnotherAssemblyViewBtn').hide();
        }
    });

    // B) Subassembly change => fetch Subassembly Views
    $('#new_subassemblyDropdown').on('change', function () {
        var subassemblyId = $(this).val();

        if (subassemblyId === 'new') {
            $('#newSubassemblyFields').show();
            $('#addAnotherSubassemblyBtn').show();

            // Disable assembly view
            $('#new_assemblyViewDropdown').prop('disabled', true).val('');
            $('#newAssemblyViewFields').hide();
            $('#addAnotherAssemblyViewBtn').hide();
        }
        else if (subassemblyId) {
            $('#newSubassemblyFields').hide();
            $('#addAnotherSubassemblyBtn').hide();

            // Enable assembly view dropdown
            $('#new_assemblyViewDropdown').prop('disabled', false);
            $('#newAssemblyViewFields').hide();
            $('#addAnotherAssemblyViewBtn').hide();

            // Clear & fetch assembly views
            $.getJSON('/get_assembly_views', { subassembly_id: subassemblyId }, function (data) {
                $('#new_assemblyViewDropdown')
                    .empty()
                    .append('<option value="">Select Subassembly View</option>');

                $.each(data, function (index, av) {
                    $('#new_assemblyViewDropdown')
                        .append('<option value="' + av.id + '">' + av.name + '</option>');
                });
                // "New Subassembly View"
                $('#new_assemblyViewDropdown').append('<option value="new">New Subassembly View...</option>');
            }).fail(function () {
                alert('Error fetching assembly views');
            });
        }
        else {
            // If user cleared
            $('#newSubassemblyFields').hide();
            $('#addAnotherSubassemblyBtn').hide();
            $('#new_assemblyViewDropdown').prop('disabled', true).val('');
            $('#newAssemblyViewFields').hide();
            $('#addAnotherAssemblyViewBtn').hide();
        }
    });

    // C) Subassembly View change => show "New..." fields
    $('#new_assemblyViewDropdown').on('change', function () {
        var avId = $(this).val();
        if (avId === 'new') {
            $('#newAssemblyViewFields').show();
            $('#addAnotherAssemblyViewBtn').show();
        } else {
            $('#newAssemblyViewFields').hide();
            $('#addAnotherAssemblyViewBtn').hide();
        }
    });

    //------------------------------------------------
    // 6) FORM SUBMIT => POST via AJAX
    //------------------------------------------------
    $('#addPositionDependenciesForm').on('submit', function (e) {
        e.preventDefault(); // Prevent page refresh

        var formData = $(this).serialize(); // gather all form data

        $.ajax({
            type: 'POST',
            url: $(this).attr('action'), // your /add_position endpoint
            data: formData,
            success: function (response) {
                // handle success
                alert('Position successfully created with ID: ' + response.position_id);
                // optionally reload or redirect, e.g.:
                // window.location.reload();
            },
            error: function (xhr) {
                // handle error
                alert('Error: ' + xhr.responseText);
            }
        });
    });
});
