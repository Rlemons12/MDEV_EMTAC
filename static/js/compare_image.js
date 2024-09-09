document.addEventListener('DOMContentLoaded', function () {
    const uploadForm = document.getElementById('upload-compare-form');
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const resultsContainer = document.getElementById('results');

    // Prevent default drag behaviors
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    // Highlight drop zone when item is dragged over it
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.add('dragover'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.remove('dragover'), false);
    });

    // Handle drop event
    dropZone.addEventListener('drop', handleDrop, false);

    // Handle file input change
    fileInput.addEventListener('change', handleFiles, false);

    // Handle form submission
    uploadForm.addEventListener('submit', function (event) {
        event.preventDefault();
        handleUpload();
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;

        handleFiles(files);
    }

    function handleFiles(files) {
        fileInput.files = files;
    }

    function clearPreviousResults() {
        document.getElementById('image-results').innerHTML = ''; // Clear image search results
        document.getElementById('image-compare-results').innerHTML = ''; // Clear previous comparison results
    }

    function handleUpload() {
        const formData = new FormData(uploadForm);
        fetch('/upload_and_compare', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            clearPreviousResults(); // Clear previous results before showing new ones
            if (data.error) {
                showResults(`<p>${data.error}</p>`);
            } else {
                renderComparisonResults(data.image_similarity_search);
            }
        })
        .catch(error => {
            console.error('Error:', error);
            showResults(`<p>Error occurred: ${error}</p>`);
        });
    }

    function showResults(content) {
        console.log('Showing results.');
        document.getElementById('results').style.display = 'block';
        document.getElementById('documents-list').innerHTML = content;
        clearPreviousResults();
    }

    function renderComparisonResults(results) {
        clearPreviousResults(); // Ensure previous results are cleared before rendering new ones
        let imagesList = '';
        results.forEach(image => {
            console.log('Rendering result for image:', image);
            imagesList += `<div class="image-details">
                            <a href="/uploads/${image.file_path}" target="_blank">
                                <img class="thumbnail" src="/uploads/${image.file_path}" alt="${image.title}">
                            </a>
                            <div class="description">
                                <h2>${image.title}</h2>
                                <p>${image.description}</p>
                                <p>Similarity: ${image.similarity.toFixed(2)}</p>
                            </div>
                            <div style="clear: both;"></div>
                          </div>`;
        });
        console.log('Comparison results prepared.');
        document.getElementById('results').style.display = 'block';
        document.getElementById('image-compare-results').innerHTML = imagesList;
    }
});
