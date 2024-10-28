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
$('#searchPositionBtn').click(function () {
    $.ajax({
        url: getPositionsUrl,
        method: 'GET',
        data: $('#searchPositionForm').serialize(),
        success: function (data) {
            clearAllSections();

            data.forEach(function (position) {
                console.log('Position Data:', position);

                setPositionDetails(position);
                renderParts(position.parts);
                renderImages(position.images);
                renderDocuments(position.documents);
                renderDrawings(position.drawings);
            });
        },
        error: function (xhr, status, error) {
            console.error("Error fetching positions:", error);
            alert("An error occurred while fetching positions.");
        }
    });
});

// Function to clear all sections before rendering
function clearAllSections() {
    $('#existing-parts-list, #existing-images-list, #existing-documents-list, #existing-drawings-list').empty();
    $('#position_display, #edit_areaDropdown, #edit_equipmentGroupDropdown, #edit_modelDropdown, #edit_assetNumberDropdown, #edit_locationDropdown, #edit_area_name, #edit_area_description, #edit_model_name, #edit_model_description, #edit_assetNumber, #edit_assetNumber_description, #edit_siteLocationDropdown, #edit_siteLocation_title, #edit_siteLocation_roomNumber').val('');
}

// Function to set position details
function setPositionDetails(position) {
    const positionId = position.position_id || position.id;
    if (positionId) {
        $('#position_display, #position_id').val(positionId);
        console.log('Position ID set:', positionId);
    }

    if (position.area) {
        $('#edit_areaDropdown').val(position.area.id);
        $('#edit_area_name').val(position.area.name);
        $('#edit_area_description').val(position.area.description);
    }

    if (position.equipment_group) {
        $('#edit_equipmentGroupDropdown').val(position.equipment_group.id);
    }

    if (position.model) {
        $('#edit_modelDropdown').val(position.model.id);
        $('#edit_model_name').val(position.model.name);
        $('#edit_model_description').val(position.model.description);
    }

    if (position.asset_number) {
        $('#edit_assetNumberDropdown').val(position.asset_number.id);
        $('#edit_assetNumber').val(position.asset_number.number);
        $('#edit_assetNumber_description').val(position.asset_number.description);
    }

    if (position.location) {
        $('#edit_locationDropdown').val(position.location.id);
        $('#edit_location_name').val(position.location.name);
        $('#edit_location_description').val(position.location.description || '');
    }

    if (position.site_location) {
        $('#edit_siteLocationDropdown').val(position.site_location.id);
        $('#edit_siteLocation_title').val(position.site_location.title);
        $('#edit_siteLocation_roomNumber').val(position.site_location.room_number);
    }
}

function renderParts(parts) {
    const partsList = $('#existing-parts-list');
    partsList.empty();

    if (!parts || parts.length === 0) {
        partsList.append('<p>No parts available.</p>');
        return;
    }

    parts.forEach(function (part) {
        console.log('Rendering part:', part); // Debugging log
        const partEntry = $(`
            <div class="existing-part" id="part-${part.part_id}">
                <span>Part Number: ${escapeHtml(part.part_number)}, Name: ${escapeHtml(part.name)}</span>
                <button type="button" class="remove-existing-part-button" data-part-id="${part.part_id}">Remove</button>
            </div>
        `);
        partsList.append(partEntry);
    });
}



// Function to render images
function renderImages(images) {
    const imagesList = $('#existing-images-list');
    imagesList.empty();

    if (!images || images.length === 0) {
        imagesList.append('<p>No images available.</p>');
        return;
    }

    images.forEach(function (image) {
        const safeTitle = escapeHtml(image.title || 'N/A');
        const safeDescription = escapeHtml(image.description || 'No description available');

        imagesList.append(`
            <div class="existing-image" id="image-${image.image_id}">
                <span>Title: ${safeTitle}, Description: ${safeDescription}</span>
                <button type="button" class="remove-existing-image-button" data-image-id="${image.image_id}">Remove</button>
            </div>
        `);
    });
}

// Function to render documents
function renderDocuments(documents) {
    const documentsList = $('#existing-documents-list');
    documentsList.empty();

    if (!documents || documents.length === 0) {
        documentsList.append('<p>No documents available.</p>');
        return;
    }

    documents.forEach(function (doc) {
        const safeTitle = escapeHtml(doc.title || 'N/A');
        const safeRev = escapeHtml(doc.rev || 'N/A');

        documentsList.append(`
            <div class="existing-document" id="document-${doc.document_id}">
                <span>Title: ${safeTitle}, Revision: ${safeRev}</span>
                <button type="button" class="remove-existing-document-button" data-document-id="${doc.document_id}">Remove</button>
            </div>
        `);
    });
}

// Function to render drawings
function renderDrawings(drawings) {
    const drawingsList = $('#existing-drawings-list');
    drawingsList.empty();

    if (!drawings || drawings.length === 0) {
        drawingsList.append('<p>No drawings available.</p>');
        return;
    }

    drawings.forEach(function (drawing) {
        drawingsList.append(`
            <div class="existing-drawing">
                <span>Drawing Name: ${drawing.drw_name}, Number: ${drawing.drw_number}</span>
                <button type="button" class="remove-existing-drawing-button" data-drawing-id="${drawing.drawing_id}">Remove</button>
            </div>
        `);
    });
}

// Utility function to escape HTML to prevent XSS
function escapeHtml(text) {
    return $('<div>').text(text).html();
}


    // Functions to paginate and filter parts, images, and drawings
    let allParts = [], allImages = [], allDrawings = [], allDocuments = [];
    let currentPartsPage = 1, currentImagesPage = 1, currentDrawingsPage = 1, currentDocumentsPage = 1;
    const partsPerPage = 10, imagesPerPage = 10, drawingsPerPage = 10, documentsPerPage = 10;


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

    // Function to render documents for the current page
function renderDocumentsPage(page = 1) {
    const startIndex = (page - 1) * documentsPerPage;
    const endIndex = startIndex + documentsPerPage;
    const currentDocuments = allDocuments.slice(startIndex, endIndex);

    const documentsList = document.getElementById('existing-documents-list');
    documentsList.innerHTML = '';  // Clear the list

    currentDocuments.forEach(doc => {
        const safeTitle = doc.title ? escapeHtml(doc.title) : 'N/A';
        const safeRev = doc.rev ? escapeHtml(doc.rev) : 'N/A';

        const docEntry = document.createElement('div');
        docEntry.className = 'existing-document';
        docEntry.id = `document-${doc.document_id}`;
        docEntry.innerHTML = `
            <span>Title: ${safeTitle}, Revision: ${safeRev}</span>
            <button type="button" class="remove-existing-document-button" data-document-id="${doc.document_id}">Remove</button>
        `;
        documentsList.appendChild(docEntry);
    });

    renderDocumentsPagination(page);
}

// Function to render pagination controls for documents
function renderDocumentsPagination(page) {
    const totalPages = Math.ceil(allDocuments.length / documentsPerPage);
    const paginationContainer = document.getElementById('documents-pagination');
    paginationContainer.innerHTML = '';

    if (totalPages > 1) {
        for (let i = 1; i <= totalPages; i++) {
            const pageButton = document.createElement('button');
            pageButton.textContent = i;
            pageButton.disabled = i === page;
            pageButton.onclick = () => renderDocumentsPage(i);
            paginationContainer.appendChild(pageButton);
        }
    }
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

function escapeHtml(text) {
    return $('<div>').text(text).html();
}

// Event delegation for removing existing images
$('#existing-images-list').on('click', '.remove-existing-image-button', function() {
    const imageId = $(this).data('image-id');
    const positionId = $('#position_id').val();

    $.ajax({
        url: removeImageFromPositionUrl, // Replace with your actual URL
        method: 'POST',
        contentType: 'application/json',
        data: JSON.stringify({ image_id: imageId, position_id: positionId }),
        success: function(response) {
            // Remove the image element from the DOM
            $(`#image-${imageId}`).remove();
            console.log(`Removed image ID ${imageId} from position ID ${positionId}`);
        },
        error: function(xhr, status, error) {
            console.error(`Error removing image ID ${imageId}:`, error);
            alert('An error occurred while removing the image.');
        }
    });
});


