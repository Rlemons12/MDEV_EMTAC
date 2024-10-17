$(document).ready(function () {
    // Declare the debounce timer at the top to avoid ReferenceError
    let searchDebounceTimer;

    // Initialize Toastr
    toastr.options = {
        closeButton: true,
        progressBar: true,
        positionClass: "toast-top-right",
        timeOut: "3000"
    };

    // Show and hide loading indicator functions
    function showLoading() {
        $('#loading-indicator').show();
    }

    function hideLoading() {
        $('#loading-indicator').hide();
    }

    // Get CSRF token from meta tag if applicable
    function getCsrfToken() {
        return $('meta[name="csrf-token"]').attr('content');
    }

    // Escape HTML to prevent XSS attacks
    function escapeHtml(text) {
        return $('<div>').text(text).html();
    }

    // Search documents function with debounce
    function searchDocuments() {
        const searchInput = $('#search-documents').val().trim();
        const suggestionBox = $('#document-suggestion-box');

        if (!searchInput) {
            suggestionBox.empty().hide();
            return;
        }

        const fetchUrl = `/pda_search_documents?query=${encodeURIComponent(searchInput)}&t=${Date.now()}`;

        showLoading();

        $.ajax({
            url: fetchUrl,
            method: 'GET',
            success: function (data) {
                suggestionBox.empty();
                if (Array.isArray(data) && data.length > 0) {
                    data.forEach(doc => {
                        const safeTitle = escapeHtml(doc.title || 'N/A');
                        const safeRev = escapeHtml(doc.rev || 'N/A');

                        const docEntry = $(`
                            <div class="suggestion-item" data-document-id="${doc.id}">
                                <strong>Title:</strong> ${safeTitle}<br>
                                <strong>Revision:</strong> ${safeRev}
                            </div>
                        `);

                        docEntry.click(function () {
                            addDocumentToPosition(doc.id, safeTitle, safeRev);
                            suggestionBox.hide();
                            $('#search-documents').val('');
                        });

                        suggestionBox.append(docEntry);
                    });
                    suggestionBox.show();
                } else {
                    suggestionBox.html('<p>No documents found.</p>').show();
                }
            },
            error: function (xhr, status, error) {
                toastr.error(`Error searching documents: ${error}`);
                console.error('Error searching documents:', error);
            },
            complete: hideLoading
        });
    }

    // Add document to position function
    function addDocumentToPosition(documentId, title, rev) {
        const positionId = $('#position_id').val().trim();

        if (!documentId || !positionId) {
            toastr.error('Document ID and Position ID are required.');
            console.error('Missing documentId or positionId:', { documentId, positionId });
            return;
        }

        showLoading();

        $.ajax({
            url: '/pda_add_document_to_position',
            method: 'POST',
            headers: { 'X-CSRFToken': getCsrfToken() },
            contentType: 'application/json',
            data: JSON.stringify({ document_id: documentId, position_id: positionId }),
            success: function (data) {
                if (data.document_id && !$(`#document-${data.document_id}`).length) {
                    const safeTitle = escapeHtml(data.title || 'N/A');
                    const safeRev = escapeHtml(data.rev || 'N/A');

                    const docEntry = $(`
                        <div class="existing-document" id="document-${data.document_id}">
                            <span>Title: ${safeTitle}, Revision: ${safeRev}</span>
                            <button type="button" class="remove-existing-document-button" data-document-id="${data.document_id}">Remove</button>
                        </div>
                    `);
                    $('#existing-documents-list').append(docEntry);
                }
            },
            error: function (xhr) {
                const errorMessage = xhr.responseJSON?.message || 'Error adding document.';
                toastr.error(errorMessage);
                console.error('Error adding document:', errorMessage);
            },
            complete: hideLoading
        });
    }

    // Event handler for removing a document
    $('#existing-documents-list').on('click', '.remove-existing-document-button', function () {
        const documentId = $(this).data('document-id');
        const positionId = $('#position_id').val().trim();
        const documentEntry = $(this).closest('.existing-document');

        if (confirm('Are you sure you want to remove this document?')) {
            showLoading();

            $.ajax({
                url: '/pda_remove_document_from_position',
                method: 'POST',
                headers: { 'X-CSRFToken': getCsrfToken() },
                contentType: 'application/json',
                data: JSON.stringify({ document_id: documentId, position_id: positionId }),
                success: function (data) {
                    toastr.success(data.message);
                    documentEntry.remove();
                },
                error: function (xhr) {
                    const errorMessage = xhr.responseJSON?.message || 'Error removing document.';
                    toastr.error(errorMessage);
                    console.error('Error removing document:', errorMessage);
                },
                complete: hideLoading
            });
        }
    });

    // Form submission for uploading a document
    $('#document-upload-form').submit(function (event) {
        event.preventDefault();

        const title = $('#doc_title').val().trim();
        const description = $('#doc_description').val().trim();
        const positionId = $('#position_id').val().trim();
        const file = $('#documents-upload')[0].files[0];

        if (!title || !file || !positionId) {
            toastr.error('Please provide a document title, select a file, and ensure Position ID is set.');
            return;
        }

        const formData = new FormData();
        formData.append('title', title);
        formData.append('description', description);
        formData.append('position_id', positionId);
        formData.append('file', file);

        showLoading();

        $.ajax({
            url: '/pda_create_and_add_document',
            method: 'POST',
            headers: { 'X-CSRFToken': getCsrfToken() },
            data: formData,
            processData: false,
            contentType: false,
            success: function (data) {
                const safeTitle = escapeHtml(data.title || 'N/A');
                const safeRev = escapeHtml(data.rev || 'N/A');

                const docEntry = $(`
                    <div class="existing-document" id="document-${data.document_id}">
                        <span>Title: ${safeTitle}, Revision: ${safeRev}</span>
                        <button type="button" class="remove-existing-document-button" data-document-id="${data.document_id}">Remove</button>
                    </div>
                `);
                $('#existing-documents-list').append(docEntry);

                $('#doc_title').val('');
                $('#doc_description').val('');
                $('#documents-upload').val('');
            },
            error: function (xhr) {
                const errorMessage = xhr.responseJSON?.message || 'Error uploading document.';
                toastr.error(errorMessage);
                console.error('Error uploading document:', errorMessage);
            },
            complete: hideLoading
        });
    });

    // Attach search event with debounce
    $('#search-documents').on('input', function () {
        clearTimeout(searchDebounceTimer);
        searchDebounceTimer = setTimeout(searchDocuments, 300);
    });

    // Pagination logic
    let allDocuments = [];
    const documentsPerPage = 10;

    function renderDocumentsPage(page = 1) {
        const startIndex = (page - 1) * documentsPerPage;
        const currentDocuments = allDocuments.slice(startIndex, startIndex + documentsPerPage);
        const documentsList = $('#existing-documents-list').empty();

        currentDocuments.forEach(doc => {
            const safeTitle = escapeHtml(doc.title || 'N/A');
            const safeRev = escapeHtml(doc.rev || 'N/A');
            documentsList.append(`
                <div class="existing-document" id="document-${doc.id}">
                    <span>Title: ${safeTitle}, Revision: ${safeRev}</span>
                    <button type="button" class="remove-existing-document-button" data-document-id="${doc.id}">Remove</button>
                </div>
            `);
        });

        renderDocumentsPagination(page);
    }

    function renderDocumentsPagination(page) {
        const totalPages = Math.ceil(allDocuments.length / documentsPerPage);
        const paginationContainer = $('#documents-pagination').empty();

        for (let i = 1; i <= totalPages; i++) {
            $('<button>')
                .text(i)
                .addClass('pagination-button')
                .prop('disabled', i === page)
                .click(() => renderDocumentsPage(i))
                .appendTo(paginationContainer);
        }
    }
});
