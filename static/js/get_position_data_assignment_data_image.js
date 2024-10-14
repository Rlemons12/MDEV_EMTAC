document.addEventListener('DOMContentLoaded', function () {
    // Function to add a new image entry for input
    function addImageEntry() {
        const newImagesContainer = document.getElementById('new-images-container');
        const imageEntry = document.createElement('div');
        imageEntry.className = 'image-entry';

        // Create label
        const label = document.createElement('label');
        label.textContent = 'Image Name:';

        // Create input
        const input = document.createElement('input');
        input.type = 'text';
        input.name = 'image_names[]';
        input.className = 'new-image-input';

        // Create remove button
        const removeButton = document.createElement('button');
        removeButton.type = 'button';
        removeButton.textContent = 'Remove';
        removeButton.addEventListener('click', function () {
            imageEntry.remove();
        });

        // Append elements to imageEntry
        imageEntry.appendChild(label);
        imageEntry.appendChild(input);
        imageEntry.appendChild(removeButton);

        // Append imageEntry to newImagesContainer
        newImagesContainer.appendChild(imageEntry);
    }

    // Function to add a new image to the database and associate it with a position
    function submitNewImage(imageName, positionId, callback) {
        if (!imageName || !positionId) {
            console.error('Image name and Position ID are required.');
            callback('Image name and Position ID are required.');
            return;
        }

        console.log(`Submitting new image: ${imageName}, positionId: ${positionId}`);

        fetch('/create_and_add_image', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ image_name: imageName, position_id: positionId })
        })
        .then(response => {
            console.log('Received response:', response);
            return response.json().then(data => ({ data, status: response.status }));
        })
        .then(({ data, status }) => {
            console.log('Received data:', data, 'Status:', status);
            if (status === 200) {
                callback(null, data);
            } else {
                console.error('Error:', data.message);
                callback(data.message);
            }
        })
        .catch(error => {
            console.error('Failed to add image:', error);
            callback('Failed to add image');
        });
    }

    // Function to submit an existing image to be associated with a position
    function submitExistingImage(imageId, positionId, callback) {
        if (!imageId || !positionId) {
            console.error('Image ID and Position ID are required.');
            callback('Image ID and Position ID are required.');
            return;
        }

        console.log(`Submitting existing image: ${imageId}, positionId: ${positionId}`);

        fetch('/add_image_to_position', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ image_id: imageId, position_id: positionId })
        })
        .then(response => {
            console.log('Received response:', response);
            return response.json().then(data => ({ data, status: response.status }));
        })
        .then(({ data, status }) => {
            console.log('Received data:', data, 'Status:', status);
            if (status === 200) {
                callback(null, data);
            } else {
                console.error('Error:', data.message);
                callback(data.message);
            }
        })
        .catch(error => {
            console.error('Failed to add image:', error);
            callback('Failed to add image');
        });
    }

    // Function to handle image form submission for new images
    function handleAddImages() {
        const newImageInputs = document.querySelectorAll('.new-image-input');
        const positionId = document.getElementById('position_id').value;

        newImageInputs.forEach(input => {
            const imageName = input.value.trim();
            if (imageName) {
                console.log(`Processing new image: ${imageName}`);
                submitNewImage(imageName, positionId, (err, data) => {
                    if (err) {
                        alert(`Error: ${err}`);
                        console.error(`Error adding image '${imageName}':`, err);
                    } else {
                        alert(`Success: ${data.message}`);
                        console.log(`Image '${imageName}' added successfully.`);
                        const existingImagesList = document.getElementById('existing-images-list');
                        const newImageEntry = document.createElement('div');
                        newImageEntry.className = 'existing-image';

                        // Create span for image name
                        const imageNameSpan = document.createElement('span');
                        imageNameSpan.textContent = `Image Name: ${data.image_name}`;

                        // Create remove button
                        const removeButton = document.createElement('button');
                        removeButton.type = 'button';
                        removeButton.textContent = 'Remove';
                        removeButton.addEventListener('click', function () {
                            removeExistingImage(this, data.image_id, positionId);
                        });

                        // Append to newImageEntry
                        newImageEntry.appendChild(imageNameSpan);
                        newImageEntry.appendChild(removeButton);

                        existingImagesList.appendChild(newImageEntry);
                        input.parentNode.remove();
                    }
                });
            } else {
                console.warn('Empty image name input skipped.');
            }
        });
    }

    // Function to remove an existing image from the UI and the database
    function removeExistingImage(button, imageId, positionId) {
        if (confirm('Are you sure you want to remove this image?')) {
            console.log(`Removing image: ${imageId} from position: ${positionId}`);
            fetch('/remove_image_from_position', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ image_id: imageId, position_id: positionId })
            })
            .then(response => {
                console.log('Received response:', response);
                return response.json().then(data => ({ data, status: response.status }));
            })
            .then(({ data, status }) => {
                console.log('Received data:', data, 'Status:', status);
                if (status === 200) {
                    button.parentNode.remove();
                    console.log(`Image ${imageId} removed successfully.`);
                } else {
                    alert('Error removing image: ' + data.message);
                    console.error('Error:', data.message);
                }
            })
            .catch(error => {
                alert('Failed to remove image.');
                console.error('Failed to remove image:', error);
            });
        }
    }

   // Function to search for images and display them in a suggestion box
function searchImages() {
    const searchInputElement = document.getElementById('images-search');
    const searchInput = searchInputElement.value.trim();
    const suggestionBox = document.getElementById('image-suggestion-box'); // The box to display search results

    console.log('Search input:', searchInput); // Debugging statement

    if (!searchInput) {
        suggestionBox.innerHTML = '';  // Clear suggestion box
        suggestionBox.style.display = 'none';  // Hide the suggestion box
        return;
    }

    // Add a timestamp to prevent caching
    const timestamp = new Date().getTime();

    const fetchUrl = `/pda_search_images?query=${encodeURIComponent(searchInput)}&t=${timestamp}`;
    console.log('Fetching URL:', fetchUrl);

    fetch(fetchUrl, {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => {
        console.log('Received response:', response);
        return response.json();
    })
    .then(data => {
        console.log('Received data:', data);
        suggestionBox.innerHTML = '';  // Clear previous results

        if (Array.isArray(data) && data.length > 0) {
            data.forEach((image, index) => {
                console.log(`Processing image ${index}:`, image);
                const imageEntry = document.createElement('div');
                imageEntry.className = 'suggestion-item';

                // Display both title and description
                imageEntry.innerHTML = `
                    <div>
                        <strong>Title:</strong> ${image.title}<br>
                        <strong>Description:</strong> ${image.description}
                    </div>
                `;

                imageEntry.addEventListener('click', function () {
                    addImageToPosition(image.id, image.title);
                    suggestionBox.style.display = 'none'; // Hide suggestion box after selection
                    searchInputElement.value = ''; // Clear the search input
                });
                suggestionBox.appendChild(imageEntry);
            });
            suggestionBox.style.display = 'block'; // Show the suggestion box when results are available
        } else {
            console.log('No images found for search input:', searchInput);
            suggestionBox.innerHTML = '<p>No images found.</p>';
            suggestionBox.style.display = 'block';
        }
    })
    .catch(error => {
        alert('Error searching images: ' + error.message);
        console.error('Error searching images:', error);
    });
}


    // Function to add image to position from search result
    function addImageToPosition(imageId, imageName) {
        const positionId = document.getElementById('position_id').value;
        console.log(`Adding image ${imageId} to position ${positionId}`);
        submitExistingImage(imageId, positionId, (err, data) => {
            if (err) {
                alert(`Error: ${err}`);
                console.error(`Error adding image '${imageName}' to position:`, err);
            } else {
                alert(`Success: ${data.message}`);
                console.log(`Image '${imageName}' added to position successfully.`);
                const existingImagesList = document.getElementById('existing-images-list');
                const newImageEntry = document.createElement('div');
                newImageEntry.className = 'existing-image';

                // Create span for image name
                const imageNameSpan = document.createElement('span');
                imageNameSpan.textContent = `Image Name: ${imageName}`;

                // Create remove button
                const removeButton = document.createElement('button');
                removeButton.type = 'button';
                removeButton.textContent = 'Remove';
                removeButton.addEventListener('click', function () {
                    removeExistingImage(this, imageId, positionId);
                });

                // Append to newImageEntry
                newImageEntry.appendChild(imageNameSpan);
                newImageEntry.appendChild(removeButton);

                existingImagesList.appendChild(newImageEntry);
            }
        });
    }

    // Function to attach event listeners to remove buttons for existing images
    function attachRemoveImageListeners() {
        const removeButtons = document.querySelectorAll('.remove-existing-image-button');
        removeButtons.forEach(button => {
            button.addEventListener('click', function () {
                const imageId = this.getAttribute('data-image-id');
                const positionId = document.getElementById('position_id').value;
                removeExistingImage(this, imageId, positionId);
            });
        });
    }

    // Call the function to attach event listeners to existing remove buttons
    attachRemoveImageListeners();

    // Add event listener to the "Add Another Image" button
    const addImageButton = document.getElementById('add-image-button');
    if (addImageButton) {
        addImageButton.addEventListener('click', addImageEntry);
    } else {
        console.error('Add Image Button not found.');
    }

    // Add event listener to the "Submit New Images" button
    const submitImagesButton = document.getElementById('submit-images-button');
    if (submitImagesButton) {
        submitImagesButton.addEventListener('click', handleAddImages);
    } else {
        console.error('Submit Images Button not found.');
    }

    // Add event listener to the "Search Images" input field
    const imagesSearchInput = document.getElementById('images-search');
    if (imagesSearchInput) {
        imagesSearchInput.addEventListener('keyup', searchImages);
    } else {
        console.error('Images Search Input not found.');
    }
});
