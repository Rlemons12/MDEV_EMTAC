$(document).ready(function () {
    var ToolManagement = {
        init: function () {
            console.log('%c[ToolManagement] Initializing module.', 'color: green; font-weight: bold;');
            try {
                this.cacheElements();
                this.bindEvents();
                console.log('%c[ToolManagement] Module initialized successfully.', 'color: green;');
            } catch (error) {
                console.error('%c[ToolManagement] Initialization failed:', 'color: red; font-weight: bold;', error);
            }
        },

        cacheElements: function () {
            console.log('%c[ToolManagement] Caching DOM elements...', 'color: blue;');
            this.$subTabs = $('.sub-tab-item'); // Tab buttons
            this.$subTabContents = $('.sub-tab-content'); // Tab content sections
            this.$searchButton = $('#toolSearchBtn'); // Search button
            this.$searchForm = $('#toolManagementForm'); // Search form
            this.$searchResults = $('#toolSearchResults'); // Search results container

            console.log('%c[ToolManagement] Elements cached:', 'color: blue;', {
                subTabs: this.$subTabs.length,
                subTabContents: this.$subTabContents.length,
                searchButton: this.$searchButton.length,
                searchForm: this.$searchForm.length,
                searchResults: this.$searchResults.length
            });

            // Validate that essential elements are present
            if (this.$subTabs.length === 0) {
                console.warn('%c[ToolManagement] Warning: No sub-tabs found.', 'color: orange;');
            }
            if (this.$searchButton.length === 0) {
                console.warn('%c[ToolManagement] Warning: Search button not found.', 'color: orange;');
            }
            if (this.$searchForm.length === 0) {
                console.warn('%c[ToolManagement] Warning: Search form not found.', 'color: orange;');
            }
            if (this.$searchResults.length === 0) {
                console.warn('%c[ToolManagement] Warning: Search results container not found.', 'color: orange;');
            }
        },

        bindEvents: function () {
            console.log('%c[ToolManagement] Binding event listeners...', 'color: blue;');
            var self = this;

            // Sub-tab click event
            self.$subTabs.on('click', function () {
                var targetTab = $(this).data('subtab'); // Get target tab ID
                console.log('%c[ToolManagement] Sub-tab clicked.', 'color: purple;', {
                    clickedTab: $(this).text(),
                    targetTabID: targetTab
                });

                if (!targetTab) {
                    console.error('%c[ToolManagement] Error: Target tab ID not found.', 'color: red;');
                    return;
                }

                // Remove active class from all tabs and hide all contents
                console.log('%c[ToolManagement] Resetting active classes for all tabs and hiding contents.', 'color: purple;');
                self.$subTabs.removeClass('active');
                self.$subTabContents.removeClass('active').hide();

                // Activate clicked tab and show corresponding content
                console.log('%c[ToolManagement] Activating clicked tab and displaying corresponding content.', 'color: purple;');
                $(this).addClass('active');
                $('#' + targetTab).addClass('active').fadeIn();

                // Log the state after activation
                console.log('%c[ToolManagement] Current active tab and content:', 'color: purple;', {
                    activeTab: $('.sub-tab-item.active').text(),
                    activeContent: $('.sub-tab-content.active').attr('id')
                });
            });

            // Search button click event
            self.$searchButton.on('click', function (e) {
                console.log('%c[ToolManagement] Search button clicked.', 'color: purple;');
                e.preventDefault(); // Prevent default form submission

                self.performSearch();
            });

            // Handle form submission via Enter key
            self.$searchForm.on('submit', function (e) {
                console.log('%c[ToolManagement] Search form submitted (via Enter key).', 'color: purple;');
                e.preventDefault(); // Prevent default form submission

                self.performSearch();
            });
        },

        performSearch: function () {
    var self = this;

    console.log('[ToolManagement] Initiating tool search...');

    // Serialize form data
    var formData = self.$searchForm.serialize();
    console.log('[ToolManagement] Serialized form data:', formData);

    // Extract CSRF token from the form
    var csrfToken = self.$searchForm.find('input[name="csrf_token"]').val();
    if (csrfToken) {
        formData += '&csrf_token=' + encodeURIComponent(csrfToken);
        console.log('[ToolManagement] CSRF token included in form data.');
    } else {
        console.warn('[ToolManagement] Warning: CSRF token not found in form.');
    }

    // Validate that form data is not empty
    if (!formData || formData.trim() === '') {
        console.warn('[ToolManagement] Warning: No form data to submit.');
        self.$searchResults.html('<p>Please enter search criteria.</p>');
        return;
    }

    // Disable the search button to prevent multiple clicks
    self.$searchButton.prop('disabled', true);
    console.log('[ToolManagement] Search button disabled to prevent duplicate searches.');

    // Show a loading indicator
    self.$searchResults.html('<p>Loading...</p>');
    console.log('[ToolManagement] Displayed loading indicator.');

    // Perform AJAX request
    $.ajax({
        url: self.$searchForm.attr('action') || window.location.href, // Fallback to current URL
        method: 'POST', // Explicitly set to POST
        data: formData,
        dataType: 'json', // Expect JSON response
        beforeSend: function () {
            console.log('[ToolManagement] AJAX request is being sent.');
        },
        success: function (response) {
            console.log('[ToolManagement] AJAX request succeeded. Response:', response);

            // Validate response structure
            if (response && Array.isArray(response.tools)) {
                if (response.tools.length > 0) {
                    self.renderSearchResults(response.tools);
                    console.log('[ToolManagement] Rendered search results.');
                } else {
                    self.$searchResults.html('<p>No tools found matching the criteria.</p>');
                    console.log('[ToolManagement] No tools found for the given criteria.');
                }
            } else {
                console.warn('[ToolManagement] Unexpected response format:', response);
                self.$searchResults.html('<p>Unexpected response from the server.</p>');
            }
        },
        error: function (xhr, status, error) {
            console.error('[ToolManagement] AJAX request failed:', {
                status: status,
                error: error,
                response: xhr.responseText
            });
            self.$searchResults.html('<p>An error occurred while searching for tools.</p>');
        },
        complete: function () {
            // Re-enable the search button after the request completes
            self.$searchButton.prop('disabled', false);
            console.log('[ToolManagement] Search button re-enabled after AJAX request.');
        }
    });
},


        renderSearchResults: function (tools) {
            var self = this;
            console.log('%c[ToolManagement] Rendering search results.', 'color: green;');

            if (!tools || tools.length === 0) {
                console.warn('%c[ToolManagement] No tools to render.', 'color: orange;');
                self.$searchResults.html('<p>No tools found matching the criteria.</p>');
                return;
            }

            // Build HTML table for tools
            var html = '<table class="table table-bordered">';
            html += '<thead><tr>';
            html += '<th>Tool ID</th>';
            html += '<th>Name</th>';
            html += '<th>Size</th>';
            html += '<th>Type</th>';
            html += '<th>Material</th>';
            html += '<th>Category</th>';
            html += '<th>Manufacturer</th>';
            html += '<th>Actions</th>';
            html += '</tr></thead><tbody>';

            tools.forEach(function (tool) {
                html += '<tr>';
                html += '<td>' + tool.id + '</td>';
                html += '<td>' + tool.name + '</td>';
                html += '<td>' + (tool.size || 'N/A') + '</td>';
                html += '<td>' + (tool.type || 'N/A') + '</td>';
                html += '<td>' + (tool.material || 'N/A') + '</td>';
                html += '<td>' + (tool.tool_category ? tool.tool_category.name : 'N/A') + '</td>';
                html += '<td>' + (tool.tool_manufacturer ? tool.tool_manufacturer.name : 'N/A') + '</td>';
                html += '<td>';
                html += '<a href="' + tool.edit_url + '" class="btn btn-sm btn-primary">Edit</a> ';
                html += '<a href="' + tool.delete_url + '" class="btn btn-sm btn-danger" onclick="return confirm(\'Are you sure you want to delete this tool?\')">Delete</a>';
                html += '</td>';
                html += '</tr>';
            });

            html += '</tbody></table>';

            // Update the search results container
            self.$searchResults.html(html);
            console.log('%c[ToolManagement] Search results rendered successfully.', 'color: green;');
        }
    };

    ToolManagement.init();
});
