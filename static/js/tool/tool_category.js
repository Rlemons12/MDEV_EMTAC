// static/js/tool/tool_category.js

document.addEventListener('DOMContentLoaded', () => {
    initializeCategoryManagement();
});

/**
 * Initializes category management functionality by setting up event listeners.
 */
function initializeCategoryManagement() {
    // Handle Edit Category Button Clicks
    document.querySelectorAll('.edit-category').forEach(button => {
        button.addEventListener('click', function() {
            const categoryId = this.getAttribute('data-id');
            const row = this.closest('tr');
            const categoryName = row.children[0].textContent.trim();
            const categoryDescription = row.children[1].textContent.trim();
            const parentCategoryText = row.children[2].textContent.trim();

            // Populate the Edit Category form fields
            document.getElementById('edit-category-id').value = categoryId;
            document.getElementById('edit-category-name').value = categoryName;
            document.getElementById('edit-category-description').value = categoryDescription;

            // Set the Parent Category select value
            // Look for an option whose text matches the parent's text
            const parentSelect = document.getElementById('edit-category-parent');
            let optionFound = false;
            for (let i = 0; i < parentSelect.options.length; i++) {
                if (parentSelect.options[i].text.trim() === parentCategoryText) {
                    parentSelect.value = parentSelect.options[i].value;
                    optionFound = true;
                    break;
                }
            }
            // If no matching option, default to "None" (value 0)
            if (!optionFound) {
                parentSelect.value = 0;
            }

            // Expand the Edit accordion item
            const collapseEdit = document.getElementById('collapseEditCategory');
            new bootstrap.Collapse(collapseEdit, { toggle: true });
        });
    });

    // Handle Delete Category Button Clicks
    document.querySelectorAll('.delete-category').forEach(button => {
        button.addEventListener('click', function() {
            const categoryId = this.getAttribute('data-id');
            const row = this.closest('tr');
            const categoryName = row.children[0].textContent.trim();

            // Populate the Delete Category form fields
            document.getElementById('delete-category-id').value = categoryId;
            document.getElementById('delete-category-name').value = categoryName;

            // Expand the Delete accordion item
            const collapseDelete = document.getElementById('collapseDeleteCategory');
            new bootstrap.Collapse(collapseDelete, { toggle: true });
        });
    });
}
