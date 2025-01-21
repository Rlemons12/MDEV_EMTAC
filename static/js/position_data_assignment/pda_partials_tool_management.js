$(document).ready(function () {
    // ToolManagement Module
    window.ToolManagement = {
        init: function () {
            console.log('%c[ToolManagement] Initializing module.', 'color: green; font-weight: bold;');
            try {
                this.cacheElements();
                this.bindEvents();

                // 1) Read position ID from hidden input #position_id
                // this.positionId = $('#position_id').val();
                console.log('[ToolManagement] positionId (from hidden input):', this.positionId);

                // 2) If valid, fetch associated tools immediately
                if (this.positionId) {
                    this.fetchAssociatedTools(this.positionId);
                } else {
                    console.warn('[ToolManagement] No position_id found in hidden input.');
                }

                console.log('%c[ToolManagement] Module initialized successfully.', 'color: green;');
            } catch (error) {
                console.error('%c[ToolManagement] Initialization failed:', 'color: red; font-weight: bold;', error);
            }
        },

        cacheElements: function () {
            console.log('%c[ToolManagement] Caching DOM elements...', 'color: blue;');

            // Sub-tabs
            this.$subTabs = $('.sub-tab-item');
            this.$subTabContents = $('.sub-tab-content');

            // "Associated Tools" sub-tab
            this.$associatedToolsTableBody = $('#associatedToolsTableBody');
            this.$goToAddToolsBtn = $('#goToAddToolsBtn');

            // Buttons/Forms in the Add/Search Tools sub-tab
            this.$searchButton = $('#toolSearchBtn');
            this.$searchForm = $('#toolManagementForm');
            this.$searchResults = $('#toolSearchResults');

            console.log('%c[ToolManagement] Elements cached:', 'color: blue;', {
                subTabs: this.$subTabs.length,
                subTabContents: this.$subTabContents.length,
                searchButton: this.$searchButton.length,
                searchForm: this.$searchForm.length,
                searchResults: this.$searchResults.length
            });
        },

        bindEvents: function () {
            var self = this;
            console.log('%c[ToolManagement] Binding event listeners...', 'color: blue;');

            // Sub-tab click
            self.$subTabs.on('click', function () {
                var targetTab = $(this).data('subtab');
                console.log('%c[ToolManagement] Sub-tab clicked.', 'color: purple;', {
                    clickedTab: $(this).text(),
                    targetTabID: targetTab
                });

                self.$subTabs.removeClass('active');
                self.$subTabContents.removeClass('active').hide();

                // Activate clicked tab
                $(this).addClass('active');
                $('#' + targetTab).addClass('active').fadeIn();
            });

            // "Go to Add Tools" button
            self.$goToAddToolsBtn.on('click', function () {
                console.log('%c[ToolManagement] Going to Add/Search Tools sub-tab.', 'color: purple;');
                self.$subTabs.removeClass('active');
                self.$subTabContents.removeClass('active').hide();

                self.$subTabs.filter('[data-subtab="add-search-tools"]').addClass('active');
                $('#add-search-tools').addClass('active').fadeIn();
            });

            // Search button
            self.$searchButton.on('click', function (e) {
                console.log('%c[ToolManagement] Search button clicked.', 'color: purple;');
                e.preventDefault();
                self.performSearch();
            });

            // Search form "Enter" submission
            self.$searchForm.on('submit', function (e) {
                console.log('%c[ToolManagement] Search form submitted (Enter key).', 'color: purple;');
                e.preventDefault();
                self.performSearch();
            });

            // Listen for clicks on "Add to Position" buttons in the search results
            // We use event delegation because the table is generated dynamically.
            $(document).on('click', '.btn-add-tool', function () {
                var toolId = $(this).data('tool-id');
                console.log('[ToolManagement] "Add to Position" clicked for toolId:', toolId);
                self.addToolToPosition(toolId);
            });
        },

        // ------------------------------------
        // FETCH ASSOCIATED TOOLS
        // ------------------------------------
        fetchAssociatedTools: function (positionId) {
            var self = this;
            console.log('[ToolManagement] Fetching associated tools for position:', positionId);

            $.ajax({
                url: '/pda_get_tools_by_position',
                method: 'GET',
                data: { position_id: positionId },
                dataType: 'json',
                success: function (response) {
                    console.log('[ToolManagement] Associated tools response:', response);
                    if (response && response.tools) {
                        self.renderAssociatedTools(response.tools);
                    } else {
                        console.warn('[ToolManagement] No "tools" key in response');
                        self.$associatedToolsTableBody.html('<tr><td colspan="6">No associated tools found.</td></tr>');
                    }
                },
                error: function (xhr, status, error) {
                    console.error('[ToolManagement] Error fetching associated tools:', error);
                }
            });
        },

        renderAssociatedTools: function (tools) {
            console.log('[ToolManagement] Rendering associated tools:', tools);
            var $tbody = this.$associatedToolsTableBody;

            // Clear existing rows
            $tbody.empty();

            if (!tools || tools.length === 0) {
                $tbody.append('<tr><td colspan="6">No associated tools found.</td></tr>');
                return;
            }

            // Build table rows
            tools.forEach(function (tool) {
                var rowHtml = '<tr>' +
                    '<td>' + tool.id + '</td>' +
                    '<td>' + (tool.name || '') + '</td>' +
                    '<td>' + (tool.manufacturer || '') + '</td>' +
                    '<td>' + (tool.type || '') + '</td>' +
                    '<td>' + (tool.material || '') + '</td>' +
                    '<td>' + (tool.category || '') + '</td>' +
                '</tr>';
                $tbody.append(rowHtml);
            });
        },

        // ------------------------------------
        // SEARCH TOOLS
        // ------------------------------------
        performSearch: function () {
            var self = this;
            console.log('[ToolManagement] Initiating tool search...');
            var formData = self.$searchForm.serialize();
            console.log('[ToolManagement] Serialized form data:', formData);

            // Basic check
            if (!formData || formData.trim() === '') {
                console.warn('[ToolManagement] Warning: No form data to submit.');
                self.$searchResults.html('<p>Please enter search criteria.</p>');
                return;
            }

            // Show loading
            self.$searchButton.prop('disabled', true);
            self.$searchResults.html('<p>Loading...</p>');

            // Send form data to /pda_search_tools
            $.ajax({
                url: self.$searchForm.attr('action') || '/pda_search_tools',
                method: 'POST',
                data: formData,
                dataType: 'json',
                success: function (response) {
                    console.log('[ToolManagement] Search response:', response);
                    if (response && Array.isArray(response.tools)) {
                        if (response.tools.length > 0) {
                            self.renderSearchResults(response.tools);
                        } else {
                            self.$searchResults.html('<p>No tools found matching the criteria.</p>');
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
                    self.$searchButton.prop('disabled', false);
                }
            });
        },

        renderSearchResults: function (tools) {
            console.log('[ToolManagement] Rendering search results:', tools);

            if (!tools || tools.length === 0) {
                this.$searchResults.html('<p>No tools found matching the criteria.</p>');
                return;
            }

            var html = '<table class="table table-bordered">';
            html += '<thead><tr>' +
                    '<th>ID</th><th>Name</th><th>Size</th><th>Type</th><th>Material</th><th>Category</th><th>Manufacturer</th><th>Actions</th>' +
                    '</tr></thead><tbody>';

            tools.forEach(function (tool) {
                html += '<tr>';
                html += '<td>' + tool.id + '</td>';
                html += '<td>' + (tool.name || '') + '</td>';
                html += '<td>' + (tool.size || 'N/A') + '</td>';
                html += '<td>' + (tool.type || 'N/A') + '</td>';
                html += '<td>' + (tool.material || 'N/A') + '</td>';
                html += '<td>' + (tool.tool_category || 'N/A') + '</td>';
                html += '<td>' + (tool.tool_manufacturer || 'N/A') + '</td>';
                html += '<td><button class="btn-add-tool" data-tool-id="' + tool.id + '">Add to Position</button></td>';
                html += '</tr>';
            });

            html += '</tbody></table>';
            this.$searchResults.html(html);
        },

        // ------------------------------------
        // ADD TOOL TO POSITION
        // ------------------------------------
        addToolToPosition: function (toolId) {
    var self = this;
    var positionId = $('#position_id').val(); // Dynamically read positionId
    console.log('[ToolManagement] addToolToPosition called. toolId:', toolId, 'positionId:', positionId);

    if (!toolId || !positionId) {
        alert('Missing tool ID or position ID.');
        return;
    }

    // Example POST to /pda_add_tool_to_position
    $.ajax({
        url: '/pda_add_tool_to_position',
        method: 'POST',
        contentType: 'application/json',
        data: JSON.stringify({
            tool_id: toolId,
            position_id: positionId
        }),
        success: function (resp) {
            console.log('[ToolManagement] addToolToPosition success:', resp);

            // Show a success message
            alert(resp.message || 'Tool added successfully!');

            // Re-fetch associated tools so we see the newly added one
            self.fetchAssociatedTools(positionId);

            // Optionally switch sub-tabs to show the user the updated list
            self.$subTabs.removeClass('active');
            self.$subTabContents.removeClass('active').hide();
            self.$subTabs.filter('[data-subtab="associated-tools"]').addClass('active');
            $('#associated-tools').addClass('active').fadeIn();
        },
        error: function (xhr, status, error) {
            console.error('[ToolManagement] addToolToPosition error:', error);
            alert('Failed to add tool. See console for more info.');
        }
    });
}
    };

    // Initialize the ToolManagement module
    ToolManagement.init();
});
