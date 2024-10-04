$(document).ready(function() {
    // Function to reset and disable dropdowns
    function resetDropdowns(selectors) {
        selectors.forEach(function(selector) {
            $(selector).prop('disabled', true).html('<option value="">Select...</option>');
        });
    }

    // Preload form for updates if position_id is provided in URL
    function preloadForm(positionId) {
        if (positionId) {
            $.ajax({
                url: getPositionsUrl, // Fetch the position data
                method: 'GET',
                data: { position_id: positionId },
                success: function(data) {
                    if (data.position) {
                        $('#pda_areaDropdown').val(data.position.area_id).trigger('change');
                        $('#pda_equipmentGroupDropdown').val(data.position.equipment_group_id).prop('disabled', false);
                        $('#pda_modelDropdown').val(data.position.model_id).prop('disabled', false);
                        $('#pda_assetNumberDropdown').val(data.position.asset_number_id).prop('disabled', false);
                        $('#pda_locationDropdown').val(data.position.location_id).prop('disabled', false);
                        $('#pda_siteLocation').val(data.position.site_location_id).prop('disabled', false);
                    }
                },
                error: function(xhr, status, error) {
                    console.error("Error fetching position data:", error);
                    alert("An error occurred while fetching the position.");
                }
            });
        }
    }

    // Check if position_id is available in the URL to preload form
    var positionId = new URLSearchParams(window.location.search).get('position_id');
    if (positionId) {
        preloadForm(positionId);
    }

    // Area change event
    $('#pda_areaDropdown').change(function() {
        var areaId = $(this).val();
        if (areaId) {
            $('#pda_equipmentGroupDropdown').prop('disabled', false);
            // Fetch Equipment Groups
            $.ajax({
                url: getEquipmentGroupsUrl,
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
                url: getModelsUrl,
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
                url: getAssetNumbersUrl,
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
                url: getLocationsUrl,
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
                url: getSiteLocationsUrl,
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

        // Fetch Positions without requiring all fields
        $.ajax({
            url: getPositionsUrl,
            method: 'GET',
            data: {
                site_location_id: siteLocationId || '',
                asset_number_id: assetNumberId || '',
                location_id: locationId || ''
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
    });

    // Allow manual input for Asset Number and Location
    $('#pda_assetNumberInput').on('input', function() {
        var assetNumber = $(this).val();
        if (assetNumber.length > 1) {
            $('#pda_assetNumberDropdown').prop('disabled', true);
        } else {
            $('#pda_assetNumberDropdown').prop('disabled', false);
        }
    });

    $('#pda_locationInput').on('input', function() {
        var location = $(this).val();
        if (location.length > 1) {
            $('#pda_locationDropdown').prop('disabled', true);
        } else {
            $('#pda_locationDropdown').prop('disabled', false);
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
// Function to fetch and display parts, images, drawings, and other position details
$('#searchPositionBtn').click(function() {
    $.ajax({
        url: getPositionsUrl,
        method: 'GET',
        data: $('#searchPositionForm').serialize(),
        success: function(data) {
            // Clear existing data in sections
            $('#existing-parts-list').empty();
            $('#existing-images-list').empty();
            $('#existing-drawings-list').empty();
            $('#position_display').val('');  // Clear position display
            $('#edit_areaDropdown').val('');
            $('#edit_equipmentGroupDropdown').val('');
            $('#edit_modelDropdown').val('');
            $('#edit_assetNumberDropdown').val('');
            $('#edit_locationDropdown').val('');
            $('#edit_area_name').val('');
            $('#edit_area_description').val('');
            $('#edit_model_name').val('');
            $('#edit_model_description').val('');
            $('#edit_assetNumber').val('');
            $('#edit_assetNumber_description').val('');
            $('#edit_siteLocationDropdown').val('');  // Clear site location dropdown
            $('#edit_siteLocation_title').val('');  // Clear site location title
            $('#edit_siteLocation_roomNumber').val('');  // Clear site location room number

            // Assuming 'data' is an array of positions
            data.forEach(function(position) {
                // Log the position data for debugging
                console.log('Position Data:', position);

                // Set position ID in the position display (visible field)
                if (position.position_id) {
                    $('#position_display').val(position.position_id); // Display the position ID
                    console.log('Position ID set to display:', position.position_id);
                }

                // Set position ID in the hidden field as well (for form submission)
                if (position.position_id) {
                    $('#position_id').val(position.position_id); // Hidden field for form submission
                    console.log('Position ID set to hidden field:', position.position_id);
                }

                // Set area
                if (position.area && position.area.id) {
                    $('#edit_areaDropdown').val(position.area.id);
                    $('#edit_area_name').val(position.area.name); // Set area name
                    $('#edit_area_description').val(position.area.description); // Set area description
                    console.log('Area set to:', position.area.id);
                }

                // Set equipment group
                if (position.equipment_group && position.equipment_group.id) {
                    $('#edit_equipmentGroupDropdown').val(position.equipment_group.id);
                    console.log('Equipment Group set to:', position.equipment_group.id);
                }

                // Set model
                if (position.model && position.model.id) {
                    $('#edit_modelDropdown').val(position.model.id);
                    $('#edit_model_name').val(position.model.name); // Set model name
                    $('#edit_model_description').val(position.model.description); // Set model description
                    console.log('Model set to:', position.model.id);
                }

                // Set asset number
                if (position.asset_number && position.asset_number.id) {
                    $('#edit_assetNumberDropdown').val(position.asset_number.id);
                    $('#edit_assetNumber').val(position.asset_number.number); // Set asset number
                    $('#edit_assetNumber_description').val(position.asset_number.description); // Set asset number description
                    console.log('Asset Number set to:', position.asset_number.id);
                }

                // Set location
                if (position.location && position.location.id) {
                    // Set the location ID in the dropdown
                    $('#edit_locationDropdown').val(position.location.id);
                    console.log('Location set to:', position.location.id);

                    // Set the location name
                    if (position.location.name) {
                        $('#edit_location_name').val(position.location.name);
                        console.log('Location name set to:', position.location.name);
                    }

                    // Set the location description
                    if (position.location.description) {
                        $('#edit_location_description').val(position.location.description);
                        console.log('Location description set to:', position.location.description);
                    } else {
                        $('#edit_location_description').val('');  // Clear the description if not available
                        console.log('Location description cleared');
                    }
                }

                // Set site location
                if (position.site_location && position.site_location.id) {
                    $('#edit_siteLocationDropdown').val(position.site_location.id);  // Set site location in dropdown
                    $('#edit_siteLocation_title').val(position.site_location.title);  // Set site location title
                    $('#edit_siteLocation_roomNumber').val(position.site_location.room_number);  // Set site location room number
                    console.log('Site Location set to:', position.site_location.id);
                    console.log('Site Location title set to:', position.site_location.title);
                    console.log('Site Location room number set to:', position.site_location.room_number);
                } else {
                    // Clear fields if site location is not available
                    $('#edit_siteLocationDropdown').val('');
                    $('#edit_siteLocation_title').val('');
                    $('#edit_siteLocation_roomNumber').val('');
                    console.log('Site Location cleared');
                }

                // Render parts
                if (position.parts.length > 0) {
                    position.parts.forEach(function(part) {
                        $('#existing-parts-list').append(`<p>Part Number: ${part.part_number}, Name: ${part.name}</p>`);
                    });
                    console.log(`Rendered ${position.parts.length} parts`);
                } else {
                    $('#existing-parts-list').append('<p>No parts available.</p>');
                    console.log('No parts available for this position');
                }

                // Render images
                if (position.images.length > 0) {
                    position.images.forEach(function(image) {
                        $('#existing-images-list').append(`<p>Image Title: ${image.image_title}, Description: ${image.description || 'No description available'}</p>`);
                    });
                    console.log(`Rendered ${position.images.length} images`);
                } else {
                    $('#existing-images-list').append('<p>No images available.</p>');
                    console.log('No images available for this position');
                }

                // Render drawings
                if (position.drawings.length > 0) {
                    position.drawings.forEach(function(drawing) {
                        $('#existing-drawings-list').append(`<p>Drawing Name: ${drawing.drawing_name}, Revision: ${drawing.drawing_revision}</p>`);
                    });
                    console.log(`Rendered ${position.drawings.length} drawings`);
                } else {
                    $('#existing-drawings-list').append('<p>No drawings available.</p>');
                    console.log('No drawings available for this position');
                }
            });
        },
        error: function(xhr, status, error) {
            console.error("Error fetching positions:", error);
            alert("An error occurred while fetching positions.");
        }
    });
});





    // Functions to paginate and filter parts, images, and drawings
    let allParts = [], allImages = [], allDrawings = [];
    let currentPartsPage = 1, currentImagesPage = 1, currentDrawingsPage = 1;
    const partsPerPage = 10, imagesPerPage = 10, drawingsPerPage = 10;

    // Function to render parts for the current page
    function renderPartsPage(page = 1) {
        const startIndex = (page - 1) * partsPerPage;
        const endIndex = startIndex + partsPerPage;
        const currentParts = allParts.slice(startIndex, endIndex);

        const partsList = document.getElementById('existing-parts-list');
        partsList.innerHTML = '';  // Clear the list

        currentParts.forEach(part => {
            const partItem = document.createElement('p');
            partItem.textContent = `Part Number: ${part.part_number}, Name: ${part.name}`;
            partsList.appendChild(partItem);
        });

        renderPartsPagination(page);
    }

    // Function to render pagination controls
    function renderPartsPagination(page) {
        const totalPages = Math.ceil(allParts.length / partsPerPage);
        const paginationContainer = document.getElementById('parts-pagination');
        paginationContainer.innerHTML = '';

        if (totalPages > 1) {
            for (let i = 1; i <= totalPages; i++) {
                const pageButton = document.createElement('button');
                pageButton.textContent = i;
                pageButton.disabled = i === page;
                pageButton.onclick = () => renderPartsPage(i);
                paginationContainer.appendChild(pageButton);
            }
        }
    }

    // Filter parts based on search input
    function filterParts() {
        const searchTerm = document.getElementById('search-parts').value.toLowerCase();
        allParts = allParts.filter(part =>
            part.part_number.toLowerCase().includes(searchTerm) ||
            part.name.toLowerCase().includes(searchTerm)
        );
        renderPartsPage(1);
    }

    // Similar functions for Images and Drawings...

    // Simulate loading of parts, images, and drawings (replace this with actual AJAX request)
    document.addEventListener('DOMContentLoaded', function() {
        const samplePartsData = [ /* Your parts data */ ];
        const sampleImagesData = [ /* Your images data */ ];
        const sampleDrawingsData = [ /* Your drawings data */ ];

        allParts = samplePartsData;
        allImages = sampleImagesData;
        allDrawings = sampleDrawingsData;

        renderPartsPage(1);
        renderImagesPage(1);
        renderDrawingsPage(1);
    });
});
