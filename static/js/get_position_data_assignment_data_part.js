document.addEventListener('DOMContentLoaded', function () {
    // Function to add a new part entry for input
    function addPartEntry() {
        const newPartsContainer = document.getElementById('new-parts-container');
        const partEntry = document.createElement('div');
        partEntry.className = 'part-entry';
        partEntry.innerHTML = `
            <label>Part Number:</label>
            <input type="text" name="part_numbers[]" class="new-part-input">
            <button type="button" onclick="removePartEntry(this)">Remove</button>
        `;
        newPartsContainer.appendChild(partEntry);
    }

    // Function to remove a new part entry (from the UI)
    window.removePartEntry = function (button) {
        button.parentNode.remove();
    };

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
        .then(response => {
            return response.json().then(data => ({ data, status: response.status }));
        })
        .then(({ data, status }) => {
            if (status === 200) {
                callback(null, data);
            } else {
                callback(data.message);
            }
        })
        .catch(error => {
            callback('Failed to add part');
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
        .then(response => {
            return response.json().then(data => ({ data, status: response.status }));
        })
        .then(({ data, status }) => {
            if (status === 200) {
                callback(null, data);
            } else {
                callback(data.message);
            }
        })
        .catch(error => {
            callback('Failed to add part');
        });
    }

    // Function to handle part form submission for new parts
    function handleAddParts() {
        const newPartInputs = document.querySelectorAll('.new-part-input');
        const positionId = document.getElementById('position_id').value;

        newPartInputs.forEach(input => {
            const partNumber = input.value;
            if (partNumber) {
                submitNewPart(partNumber, positionId, (err, data) => {
                    if (err) {
                        alert(`Error: ${err}`);
                    } else {
                        alert(`Success: ${data.message}`);
                        const existingPartsList = document.getElementById('existing-parts-list');
                        const newPartEntry = document.createElement('div');
                        newPartEntry.className = 'existing-part';
                        newPartEntry.innerHTML = `
                            <span>Part Number: ${data.part_number}</span>
                            <button type="button" onclick="removeExistingPart(this, '${data.part_id}', '${positionId}')">Remove</button>
                        `;
                        existingPartsList.appendChild(newPartEntry);
                        input.parentNode.remove();
                    }
                });
            }
        });
    }

    // Function to remove an existing part from the UI and the database
    window.removeExistingPart = function (button, partId, positionId) {
        if (confirm('Are you sure you want to remove this part?')) {
            fetch('/remove_part_from_position', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ part_id: partId, position_id: positionId })
            })
            .then(response => {
                return response.json().then(data => ({ data, status: response.status }));
            })
            .then(({ data, status }) => {
                if (status === 200) {
                    button.parentNode.remove();
                } else {
                    alert('Error removing part: ' + data.message);
                }
            })
            .catch(error => {
                alert('Failed to remove part.');
            });
        }
    };

    // Function to search for parts and display them in a suggestion box
    function searchParts() {
        const searchInputElement = document.getElementById('parts-search');
        const searchInput = searchInputElement.value.trim().toLowerCase();
        const suggestionBox = document.getElementById('suggestion-box'); // The box to display search results

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
            suggestionBox.innerHTML = '';  // Clear previous results

            if (data.length > 0) {
                data.forEach(part => {
                    const partEntry = document.createElement('div');
                    partEntry.className = 'suggestion-item';
                    partEntry.innerHTML = `
                        <span>Part Number: ${part.part_number}</span>
                    `;
                    partEntry.addEventListener('click', function () {
                        addPartToPosition(part.id, part.part_number);
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
        const positionId = document.getElementById('position_id').value;
        submitExistingPart(partId, positionId, (err, data) => {
            if (err) {
                alert(`Error: ${err}`);
            } else {
                alert(`Success: ${data.message}`);
                const existingPartsList = document.getElementById('existing-parts-list');
                const newPartEntry = document.createElement('div');
                newPartEntry.className = 'existing-part';
                newPartEntry.innerHTML = `
                    <span>Part Number: ${partNumber}</span>
                    <button type="button" onclick="removeExistingPart(this, '${partId}', '${positionId}')">Remove</button>
                `;
                existingPartsList.appendChild(newPartEntry);
            }
        });
    }

    // Add event listener to the "Add Another Part" button
    document.getElementById('add-part-button').addEventListener('click', addPartEntry);

    // Add event listener to the "Submit New Parts" button
    document.getElementById('submit-parts-button').addEventListener('click', handleAddParts);

    // Add event listener to the "Search Parts" input field
    document.getElementById('parts-search').addEventListener('keyup', searchParts);
});
