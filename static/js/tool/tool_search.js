// tool_search.js
document.addEventListener('DOMContentLoaded', function () {
    const searchForm = document.getElementById('tool_search_form');
    const resultsContainer = document.getElementById('search_results');

    searchForm.addEventListener('submit', function (event) {
        event.preventDefault(); // Prevent form submission

        const formData = new FormData(searchForm);
        const queryParams = new URLSearchParams(formData).toString();

        fetch(`/tool_search?${queryParams}`)
            .then(response => response.json())
            .then(tools => {
                resultsContainer.innerHTML = '';
                if (tools.length === 0) {
                    resultsContainer.textContent = 'No tools found.';
                    return;
                }
                tools.forEach(tool => {
                    const toolDiv = document.createElement('div');
                    toolDiv.className = 'tool-item';
                    toolDiv.innerHTML = `
                        <h3>${tool.name}</h3>
                        <p>Size: ${tool.size || 'N/A'}</p>
                        <p>Type: ${tool.type || 'N/A'}</p>
                        <p>Material: ${tool.material || 'N/A'}</p>
                        <p>Description: ${tool.description || 'N/A'}</p>
                        <h4>Images:</h4>
                        <div class="tool-images">
                            ${tool.images.map(image => `
                                <img src="${image.file_path}" alt="${image.description}" title="${image.title}" />
                            `).join('')}
                        </div>
                    `;
                    resultsContainer.appendChild(toolDiv);
                });
            })
            .catch(error => {
                console.error('Error fetching tools:', error);
                resultsContainer.textContent = 'An error occurred while searching for tools.';
            });
    });
});
