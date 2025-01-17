document.addEventListener('DOMContentLoaded', function () {
    // Utility function to escape HTML to prevent XSS
    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    // Function to add a new part entry for input
    function addPartEntry() {
        const newPartsContainer = document.getElementById('new-parts-container');
        const partEntry = document.createElement('div');
        partEntry.className = 'part-entry';
        partEntry.innerHTML = `
            <label>Part Number:</label>
            <input type="text" name="part_numbers[]" class="new-part-input" required>
            <button type="button" class="remove-part-button">Remove</button>
        `;
        newPartsContainer.appendChild(partEntry);
    }

    // Function to remove a new part entry (from the UI)
    function removePartEntry(button) {
        const partEntry = button.parentElement;
        partEntry.remove();
    }

    // Function to add a new part to the database and associate it with a position
    function submitNewPart(partNumber, positionId, callback) {
        if (!partNumber || !positionId) {
            callback('Part number and Position ID are required.');
            return;
        }

        fetch('/create_and_add_part', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ part_number: partNumber, position_id: positionId })
        })
        .then(response => response.json().then(data => ({ data, status: response.status })))
        .then(({ data, status }) => {
            console.log('Submit New Part Response:', data); // Debugging log
            if (status === 200 && data.success) {
                callback(null, data);
            } else {
                callback(data.message || 'Failed to add part.');
            }
        })
        .catch(error => {
            console.error('Error adding part:', error);
            callback('Failed to add part.');
        });
    }

    // Function to submit an existing part to be associated with a position
    function submitExistingPart(partId, positionId, callback) {
    if (!partId || !positionId) {
        callback('Part ID and Position ID are required.');
        return;
    }

    fetch('/add_part_to_position', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ part_id: partId, position_id: positionId })
    })
    .then(response => response.json().then(data => ({ data, status: response.status })))
    .then(({ data, status }) => {
        console.log('Submit Existing Part Response:', data); // Debugging log
        if (status === 200) {
            callback(null, data);
        } else {
            callback(data.message || 'Failed to add part.');
        }
    })
    .catch(error => {
        console.error('Error adding existing part:', error);
        callback('Failed to add part.');
    });
}


    // Function to handle part form submission for new parts
    function handleAddParts() {
        const newPartInputs = document.querySelectorAll('.new-part-input');
        const positionIdElement = document.getElementById('position_id');
        if (!positionIdElement) {
            alert('Position ID element not found.');
            return;
        }
        const positionId = positionIdElement.value;

        newPartInputs.forEach(input => {
            const partNumber = input.value.trim();
            if (partNumber) {
                submitNewPart(partNumber, positionId, (err, data) => {
                    if (err) {
                        alert(`Error: ${err}`);
                    } else {
                        alert(`Success: ${data.message}`);
                        const existingPartsList = document.getElementById('existing-parts-list');
                        const newPartEntry = document.createElement('div');
                        newPartEntry.className = 'existing-part';
                        newPartEntry.id = `part-${data.id}`; // Changed from data.part_id to data.id
                        newPartEntry.innerHTML = `
                            <span>Part Number: ${escapeHtml(data.part_number)}</span>
                            <button type="button" class="remove-existing-part-button"
                                    data-part-id="${data.id}" data-position-id="${positionId}">Remove</button>
                        `;
                        existingPartsList.appendChild(newPartEntry);
                        input.parentElement.remove();
                    }
                });
            }
        });
    }

    // Function to search for parts and display them in a suggestion box
    function searchParts() {
        const searchInputElement = document.getElementById('parts-search');
        const searchInput = searchInputElement.value.trim().toLowerCase();
        const suggestionBox = document.getElementById('parts-suggestion-box');

        console.log('Search input:', searchInput); // Debugging statement

        if (!searchInput) {
            suggestionBox.innerHTML = '';  // Clear suggestion box
            suggestionBox.style.display = 'none';  // Hide the suggestion box
            return;
        }

        // Add a timestamp to prevent caching
        const timestamp = new Date().getTime();

        fetch(`/pda_search_parts?query=${encodeURIComponent(searchInput)}&t=${timestamp}`, {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json'
            }
        })
        .then(response => response.json())
        .then(data => {
            console.log('Search Parts Response:', data); // Debugging log
            suggestionBox.innerHTML = '';  // Clear previous results

            if (data.length > 0) {
                data.forEach(part => {
                    const partEntry = document.createElement('div');
                    partEntry.className = 'suggestion-item';
                    partEntry.innerHTML = `
                        <span>Part Number: ${escapeHtml(part.part_number)}</span>
                    `;
                    partEntry.addEventListener('click', function () {
                        addPartToPosition(part.id, part.part_number); // Changed from part.part_id to part.id
                        suggestionBox.style.display = 'none'; // Hide suggestion box after selection
                        searchInputElement.value = ''; // Clear the search input
                    });
                    suggestionBox.appendChild(partEntry);
                });
                suggestionBox.style.display = 'block'; // Show the suggestion box when results are available
            } else {
                suggestionBox.innerHTML = '<p>No parts found.</p>';
                suggestionBox.style.display = 'block';
            }
        })
        .catch(error => {
            alert('Error searching parts: ' + error.message);
            console.error('Error details:', error);
        });
    }

    // Function to add part to position from search result
    function addPartToPosition(partId, partNumber) {
        const positionIdElement = document.getElementById('position_id');
        if (!positionIdElement) {
            alert('Position ID element not found.');
            return;
        }
        const positionId = positionIdElement.value;

        submitExistingPart(partId, positionId, (err, data) => {
            if (err) {
                alert(`Error: ${err}`);
            } else {
                alert(`Success: ${data.message}`);
                const existingPartsList = document.getElementById('existing-parts-list');
                const newPartEntry = document.createElement('div');
                newPartEntry.className = 'existing-part';
                newPartEntry.id = `part-${partId}`;
                newPartEntry.innerHTML = `
                    <span>Part Number: ${escapeHtml(partNumber)}</span>
                    <button type="button" class="remove-existing-part-button"
                            data-part-id="${partId}" data-position-id="${positionId}">Remove</button>
                `;
                existingPartsList.appendChild(newPartEntry);
            }
        });
    }

    // Event listener for the "Add Another Part" button
    document.getElementById('add-part-button').addEventListener('click', addPartEntry);

    // Event listener for the "Submit New Parts" button
    document.getElementById('submit-parts-button').addEventListener('click', handleAddParts);

    // Event listener for the "Search Parts" input field with debounce
    let debounceTimer;
    document.getElementById('parts-search').addEventListener('keyup', function() {
        clearTimeout(debounceTimer);
        const query = this.value.trim();
        debounceTimer = setTimeout(function() {
            if (query.length >= 2) { // Start searching after 2 characters
                searchParts();
            }
        }, 300); // 300 milliseconds delay
    });

    // Event delegation for removing existing parts
    document.getElementById('existing-parts-list').addEventListener('click', function(event) {
    if (event.target && event.target.classList.contains('remove-existing-part-button')) {
        console.log('Event listener attached for remove button');
        const button = event.target;
        const partId = button.getAttribute('data-part-id');
        const positionIdElement = document.getElementById('position_id');
        console.log('positionIdElement:', positionIdElement); // Debug log

        const positionId = positionIdElement ? positionIdElement.value : null;
        console.log('Remove button clicked:', { partId, positionId }); // Debug log

        if (!partId || !positionId) {
            alert('Invalid part or position ID.');
            return;
        }

        if (confirm('Are you sure you want to remove this part?')) {
            fetch('/remove_part_from_position', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                    // Include CSRF token if necessary
                },
                body: JSON.stringify({ part_id: partId, position_id: positionId })
            })
            .then(response => response.json())
            .then(data => {
                console.log('Remove Part Response:', data); // Debugging log
                if (data.success) {
                    // Remove the part from the DOM
                    button.parentElement.remove();
                    alert('Part removed successfully.');
                } else {
                    alert('Error removing part: ' + data.message);
                }
            })
            .catch(error => {
                alert('Failed to remove part.');
                console.error('Error:', error);
            });
        }
    }
});


    // Event delegation for removing dynamically added new parts (optional)
    document.getElementById('new-parts-container').addEventListener('click', function(event) {
        if (event.target && event.target.classList.contains('remove-part-button')) {
            removePartEntry(event.target);
        }
    });
});
