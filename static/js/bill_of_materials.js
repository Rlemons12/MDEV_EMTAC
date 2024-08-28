// Function to fetch data and populate dropdowns for the BOM form
function populateDropdownsBOM() {
    // Define an array of dropdown elements along with their corresponding data keys
    var dropdowns = [
        { element: $('#bom_areaDropdown'), dataKey: 'areas' },
        { element: $('#bom_equipmentGroupDropdown'), dataKey: 'equipment_groups' },
        { element: $('#bom_modelDropdown'), dataKey: 'models' },
        { element: $('#bom_assetNumberDropdown'), dataKey: 'asset_numbers' },
        { element: $('#bom_locationDropdown'), dataKey: 'locations' }
    ];

    // AJAX request to fetch data
    $.ajax({
        url: '/get_bom_list_data', // URL to fetch data from
        type: 'GET',
        success: function(data) {
            // Populate areas dropdown
            var areaDropdown = $('#bom_areaDropdown');
            areaDropdown.empty(); // Clear existing options
            $.each(data['areas'], function(index, area) {
                areaDropdown.append('<option value="' + area.id + '">' + area.name + '</option>');
            });

            // Event listener for area dropdown change
            areaDropdown.change(function() {
                var selectedAreaId = $(this).val();
                var equipmentGroupDropdown = $('#bom_equipmentGroupDropdown');
                equipmentGroupDropdown.empty(); // Clear existing options
                equipmentGroupDropdown.append('<option value="">Select...</option>');

                // Populate equipment group dropdown with associated groups based on selected area
                $.each(data['equipment_groups'], function(index, group) {
                    if (group.area_id == selectedAreaId) {
                        equipmentGroupDropdown.append('<option value="' + group.id + '">' + group.name + '</option>');
                    }
                });
                equipmentGroupDropdown.change(); // Trigger change event for equipment group dropdown
            });

            // Event listener for equipment group dropdown change
            $('#bom_equipmentGroupDropdown').change(function() {
                var selectedGroupId = $(this).val();
                var modelDropdown = $('#bom_modelDropdown');
                modelDropdown.empty(); // Clear existing options
                modelDropdown.append('<option value="">Select...</option>');

                // Populate model dropdown with associated models based on selected equipment group
                $.each(data['models'], function(index, model) {
                    if (model.equipment_group_id == selectedGroupId) {
                        modelDropdown.append('<option value="' + model.id + '">' + model.name + '</option>');
                    }
                });
                modelDropdown.change(); // Trigger change event for model dropdown
            });

            // Event listener for model dropdown change
            $('#bom_modelDropdown').change(function() {
                var selectedModelId = $(this).val();
                var assetNumberDropdown = $('#bom_assetNumberDropdown');
                assetNumberDropdown.empty(); // Clear existing options
                assetNumberDropdown.append('<option value="">Select...</option>');

                // Populate asset number dropdown with associated asset numbers based on selected model
                $.each(data['asset_numbers'], function(index, assetNumber) {
                    if (assetNumber.model_id == selectedModelId) {
                        assetNumberDropdown.append('<option value="' + assetNumber.id + '">' + assetNumber.number + '</option>');
                    }
                });
                assetNumberDropdown.change(); // Trigger change event for asset number dropdown

                // Populate location dropdown with associated locations based on selected model
                var locationDropdown = $('#bom_locationDropdown');
                locationDropdown.empty(); // Clear existing options
                locationDropdown.append('<option value="">Select...</option>');

                $.each(data['locations'], function(index, location) {
                    if (location.model_id == selectedModelId) {
                        locationDropdown.append('<option value="' + location.id + '">' + location.name + '</option>');
                    }
                });

                // Initialize Select2 or any other necessary actions
                locationDropdown.select2();
                assetNumberDropdown.select2();
                areaDropdown.select2();
            });

            // Call change event to populate equipment group dropdown initially based on the default selected area
            areaDropdown.change();
        },
        error: function(xhr, status, error) {
            console.error('Error fetching data:', error);
        }
    });
}




// Call the function to populate dropdowns when the page loads
$(document).ready(function() {
    populateDropdownsBOM();
});
