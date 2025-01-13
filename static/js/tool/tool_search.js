// static/js/tool/tool_search.js

document.addEventListener('DOMContentLoaded', () => {
    initializeToolSearch();
});

/**
 * Initializes the tool search functionality by setting up event listeners.
 */
function initializeToolSearch() {
    const searchForm = document.getElementById('tool_search_form');
    const resultsContainer = document.getElementById('search-results-container'); // Updated ID

    if (!searchForm) {
        console.warn('Search form not found.');
        return;
    }

    if (!resultsContainer) {
        console.warn('Search results container not found.');
        return;
    }

    // Optional: Add a loading spinner element
    const loadingSpinner = createLoadingSpinner();
    resultsContainer.appendChild(loadingSpinner);

    searchForm.addEventListener('submit', async function (event) {
        event.preventDefault(); // Prevent default form submission

        // Show the loading spinner
        loadingSpinner.style.display = 'block';
        resultsContainer.innerHTML = ''; // Clear existing results

        const formData = new FormData(searchForm);
        const queryParams = new URLSearchParams(formData).toString();

        // Default to first page if not specified
        const page = 1;
        const perPage = 10;

        try {
            const response = await fetch(`/tool/submit_tool_data?${queryParams}&page=${page}&per_page=${perPage}`, { // Updated endpoint
                method: 'GET',
                headers: {
                    'Accept': 'application/json',
                    'X-Requested-With': 'XMLHttpRequest' // Inform the server it's an AJAX request
                }
            });

            if (!response.ok) {
                throw new Error(`Server responded with status ${response.status}`);
            }

            const data = await response.json();

            // Hide the loading spinner
            loadingSpinner.style.display = 'none';

            if (data.success) {
                displaySearchResults(data.tools, resultsContainer);
                displayPaginationControls(data.total, data.page, data.per_page, formData);
            } else {
                displayErrorMessage(resultsContainer, data.message || 'An unexpected error occurred.');
            }
        } catch (error) {
            console.error('Error fetching tools:', error);
            loadingSpinner.style.display = 'none';
            displayErrorMessage(resultsContainer, 'An error occurred while searching for tools. Please try again later.');
        }
    });
}

/**
 * Creates a loading spinner element.
 * @returns {HTMLElement} - The loading spinner element.
 */
function createLoadingSpinner() {
    const spinnerDiv = document.createElement('div');
    spinnerDiv.className = 'd-flex justify-content-center my-3';
    spinnerDiv.innerHTML = `
        <div class="spinner-border text-primary" role="status">
            <span class="visually-hidden">Loading...</span>
        </div>
    `;
    spinnerDiv.style.display = 'none'; // Hidden by default
    return spinnerDiv;
}

/**
 * Displays the search results in the designated container.
 * @param {Array} tools - Array of tool objects returned from the server.
 * @param {HTMLElement} container - The DOM element to display the results.
 */
function displaySearchResults(tools, container) {
    if (!Array.isArray(tools) || tools.length === 0) {
        container.innerHTML = '<p>No tools found matching your criteria.</p>';
        return;
    }

    // Create a table to display tools
    const table = document.createElement('table');
    table.className = 'table table-bordered table-striped';

    // Create table header
    const thead = document.createElement('thead');
    thead.innerHTML = `
        <tr>
            <th>Tool Name</th>
            <th>Size</th>
            <th>Type</th>
            <th>Material</th>
            <th>Category</th>
            <th>Manufacturer</th>
            <th>Description</th>
            <th>Image</th>
        </tr>
    `;
    table.appendChild(thead);

    // Create table body
    const tbody = document.createElement('tbody');

    tools.forEach(tool => {
        const tr = document.createElement('tr');

        tr.innerHTML = `
            <td>${escapeHTML(tool.name)}</td>
            <td>${escapeHTML(tool.size) || 'N/A'}</td>
            <td>${escapeHTML(tool.type) || 'N/A'}</td>
            <td>${escapeHTML(tool.material) || 'N/A'}</td>
            <td>${escapeHTML(tool.category) || 'N/A'}</td>
            <td>${escapeHTML(tool.manufacturer) || 'N/A'}</td>
            <td>${escapeHTML(tool.description) || 'N/A'}</td>
            <td>
                ${tool.image ? `<img src="/static/${escapeAttribute(tool.image)}" alt="${escapeAttribute(tool.name)}" width="100">` : 'No Image'}
            </td>
        `;

        tbody.appendChild(tr);
    });

    table.appendChild(tbody);
    container.appendChild(table);
}

/**
 * Displays pagination controls based on the total number of tools.
 * @param {number} total - Total number of matching tools.
 * @param {number} currentPage - Current page number.
 * @param {number} perPage - Number of tools per page.
 * @param {FormData} formData - The search form data.
 */
function displayPaginationControls(total, currentPage, perPage, formData) {
    const container = document.getElementById('search-results-container');
    const totalPages = Math.ceil(total / perPage);

    if (totalPages <= 1) return; // No need for pagination

    const paginationNav = document.createElement('nav');
    paginationNav.setAttribute('aria-label', 'Search results pages');

    const ul = document.createElement('ul');
    ul.className = 'pagination justify-content-center';

    // Previous button
    const prevLi = document.createElement('li');
    prevLi.className = `page-item ${currentPage === 1 ? 'disabled' : ''}`;
    prevLi.innerHTML = `
        <a class="page-link" href="#" aria-label="Previous" data-page="${currentPage - 1}">
            <span aria-hidden="true">&laquo; Previous</span>
        </a>
    `;
    ul.appendChild(prevLi);

    // Current page indicator
    const currentLi = document.createElement('li');
    currentLi.className = 'page-item disabled';
    currentLi.innerHTML = `
        <span class="page-link">
            Page ${currentPage} of ${totalPages}
        </span>
    `;
    ul.appendChild(currentLi);

    // Next button
    const nextLi = document.createElement('li');
    nextLi.className = `page-item ${currentPage === totalPages ? 'disabled' : ''}`;
    nextLi.innerHTML = `
        <a class="page-link" href="#" aria-label="Next" data-page="${currentPage + 1}">
            <span aria-hidden="true">Next &raquo;</span>
        </a>
    `;
    ul.appendChild(nextLi);

    paginationNav.appendChild(ul);
    container.appendChild(paginationNav);

    // Add event listeners to pagination links
    paginationNav.addEventListener('click', async function(event) {
        event.preventDefault();
        const target = event.target.closest('a');
        if (!target) return;

        const selectedPage = parseInt(target.getAttribute('data-page'));
        if (isNaN(selectedPage) || selectedPage < 1 || selectedPage > totalPages) return;

        // Optional: Show loading spinner or disable pagination
        const loadingSpinner = createLoadingSpinner();
        container.appendChild(loadingSpinner);
        loadingSpinner.style.display = 'block';

        try {
            // Reconstruct query parameters with the new page number
            const searchParams = new URLSearchParams(formData);
            searchParams.set('page', selectedPage);
            searchParams.set('per_page', perPage);

            const response = await fetch(`/tool/submit_tool_data?${searchParams.toString()}`, {
                method: 'GET',
                headers: {
                    'Accept': 'application/json',
                    'X-Requested-With': 'XMLHttpRequest'
                }
            });

            if (!response.ok) {
                throw new Error(`Server responded with status ${response.status}`);
            }

            const data = await response.json();

            // Clear existing results and pagination
            container.innerHTML = '';

            if (data.success) {
                displaySearchResults(data.tools, container);
                displayPaginationControls(data.total, data.page, data.per_page, formData);
            } else {
                displayErrorMessage(container, data.message || 'An unexpected error occurred.');
            }
        } catch (error) {
            console.error('Error fetching tools:', error);
            displayErrorMessage(container, 'An error occurred while fetching the next page. Please try again later.');
        } finally {
            // Hide the loading spinner
            loadingSpinner.style.display = 'none';
            loadingSpinner.remove();
        }
    });
}

/**
 * Displays an error message within the results container.
 * @param {HTMLElement} container - The DOM element to display the error message.
 * @param {string} message - The error message to display.
 */
function displayErrorMessage(container, message) {
    container.innerHTML = `
        <div class="alert alert-danger alert-dismissible fade show" role="alert">
            ${escapeHTML(message)}
            <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
        </div>
    `;
}

/**
 * Escapes HTML to prevent XSS attacks.
 * @param {string} unsafe - The string to escape.
 * @returns {string} - The escaped string.
 */
function escapeHTML(unsafe) {
    if (typeof unsafe !== 'string') return '';
    return unsafe
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

/**
 * Escapes attribute values to prevent XSS attacks.
 * @param {string} unsafe - The attribute value to escape.
 * @returns {string} - The escaped attribute value.
 */
function escapeAttribute(unsafe) {
    if (typeof unsafe !== 'string') return '';
    return unsafe
        .replace(/&/g, "&amp;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");
}
