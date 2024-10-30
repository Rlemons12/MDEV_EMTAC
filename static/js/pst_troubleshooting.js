// static/js/pst_troubleshooting.js

document.addEventListener('DOMContentLoaded', () => {
    const BASE_URL = '/pst_troubleshooting';  // Matches the blueprint's url_prefix
    const GET_EQUIPMENT_GROUPS_URL = `${BASE_URL}/get_equipment_groups`;
    const GET_MODELS_URL = `${BASE_URL}/get_models`;
    const GET_ASSET_NUMBERS_URL = `${BASE_URL}/get_asset_numbers`;
    const GET_LOCATIONS_URL = `${BASE_URL}/get_locations`;
    const GET_SITE_LOCATIONS_URL = `${BASE_URL}/get_site_locations`;
    const GET_POSITIONS_URL = `${BASE_URL}/get_positions`;

    // Dropdown elements
    const dropdowns = {
        area: document.getElementById('pst_areaDropdown'),
        equipmentGroup: document.getElementById('pst_equipmentGroupDropdown'),
        model: document.getElementById('pst_modelDropdown'),
        assetNumber: document.getElementById('pst_assetNumberDropdown'),
        assetNumberInput: document.getElementById('pst_assetNumberInput'),
        location: document.getElementById('pst_locationDropdown'),
        locationInput: document.getElementById('pst_locationInput'),
        siteLocation: document.getElementById('pst_siteLocationDropdown')
    };

    // Event listener for Area change
    dropdowns.area.addEventListener('change', function() {
        const areaId = this.value;
        if (areaId) {
            fetch(`${GET_EQUIPMENT_GROUPS_URL}?area_id=${areaId}`)
                .then(response => response.json())
                .then(data => {
                    populateDropdown(dropdowns.equipmentGroup, data, 'Select Equipment Group');
                })
                .catch(error => console.error('Error fetching equipment groups:', error));
        } else {
            resetDropdown(dropdowns.equipmentGroup, 'Select Equipment Group');
            resetDropdown(dropdowns.model, 'Select Model');
            resetDropdown(dropdowns.assetNumber, 'Select Asset Number');
            resetDropdown(dropdowns.location, 'Select Location');
            resetDropdown(dropdowns.siteLocation, 'Select Site Location');
        }
    });

    // Event listener for Equipment Group change
    dropdowns.equipmentGroup.addEventListener('change', function() {
        const equipmentGroupId = this.value;
        if (equipmentGroupId) {
            fetch(`${GET_MODELS_URL}?equipment_group_id=${equipmentGroupId}`)
                .then(response => response.json())
                .then(data => {
                    populateDropdown(dropdowns.model, data, 'Select Model');
                })
                .catch(error => console.error('Error fetching models:', error));
        } else {
            resetDropdown(dropdowns.model, 'Select Model');
            resetDropdown(dropdowns.assetNumber, 'Select Asset Number');
            resetDropdown(dropdowns.location, 'Select Location');
            resetDropdown(dropdowns.siteLocation, 'Select Site Location');
        }
    });

    // Event listener for Model change
    dropdowns.model.addEventListener('change', function() {
        const modelId = this.value;
        if (modelId) {
            fetch(`${GET_ASSET_NUMBERS_URL}?model_id=${modelId}`)
                .then(response => response.json())
                .then(data => {
                    populateDropdown(dropdowns.assetNumber, data, 'Select Asset Number');
                })
                .catch(error => console.error('Error fetching asset numbers:', error));

            fetch(`${GET_LOCATIONS_URL}?model_id=${modelId}`)
                .then(response => response.json())
                .then(data => {
                    populateDropdown(dropdowns.location, data, 'Select Location');
                })
                .catch(error => console.error('Error fetching locations:', error));
        } else {
            resetDropdown(dropdowns.assetNumber, 'Select Asset Number');
            resetDropdown(dropdowns.location, 'Select Location');
            resetDropdown(dropdowns.siteLocation, 'Select Site Location');
        }
    });

    // Event listener for Asset Number and Location change to fetch Site Locations
    dropdowns.assetNumber.addEventListener('change', fetchSiteLocations);
    dropdowns.location.addEventListener('change', fetchSiteLocations);

    function fetchSiteLocations() {
        const modelId = dropdowns.model.value;
        const assetNumberId = dropdowns.assetNumber.value;
        const locationId = dropdowns.location.value;

        if (modelId && assetNumberId && locationId) {
            fetch(`${GET_SITE_LOCATIONS_URL}?model_id=${modelId}&asset_number_id=${assetNumberId}&location_id=${locationId}`)
                .then(response => response.json())
                .then(data => {
                    populateDropdown(dropdowns.siteLocation, data, 'Select Site Location');
                })
                .catch(error => console.error('Error fetching site locations:', error));
        } else {
            resetDropdown(dropdowns.siteLocation, 'Select Site Location');
        }
    }

    // Utility function to populate dropdown
    function populateDropdown(dropdown, data, placeholder) {
        dropdown.innerHTML = `<option value="">${placeholder}</option>`;
        data.forEach(item => {
            let displayText = item.name || `${item.title} - Room ${item.room_number}`;
            dropdown.innerHTML += `<option value="${item.id}">${displayText}</option>`;
        });
        dropdown.disabled = false;
    }

    // Utility function to reset dropdown
    function resetDropdown(dropdown, placeholder) {
        dropdown.innerHTML = `<option value="">${placeholder}</option>`;
        dropdown.disabled = true;
    }

    // Initialize dropdowns on page load if editing an existing problem
    window.onload = function() {
        // Optionally, populate fields if problem data is present
    };
});
