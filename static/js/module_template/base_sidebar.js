document.addEventListener("DOMContentLoaded", function() {
    const sidebarToggleBtn = document.getElementById("sidebarCollapse");
    const closeSidebarBtn = document.getElementById("closeSidebar");
    const mainContainer = document.querySelector(".main-container");
    const toggleVoiceBtn = document.getElementById("toggle-voice");
    const toggleTextToSpeechBtn = document.getElementById("toggle-text-to-speech");

    // Toggle the sidebar collapse state
    const toggleSidebar = () => {
        if (mainContainer) {
            mainContainer.classList.toggle("collapsed");
            // Update aria-expanded attribute for accessibility
            if (sidebarToggleBtn) {
                const isCollapsed = mainContainer.classList.contains("collapsed");
                sidebarToggleBtn.setAttribute("aria-expanded", !isCollapsed);
            }
        }
    };

    // Sidebar collapse button
    if (sidebarToggleBtn) {
        sidebarToggleBtn.onclick = toggleSidebar;
    }

    // Close sidebar button (if one exists in your sidebar template)
    if (closeSidebarBtn) {
        closeSidebarBtn.onclick = toggleSidebar;
    }

    // Toggle Voice button
    if (toggleVoiceBtn) {
        toggleVoiceBtn.onclick = function() {
            this.classList.toggle("active");
        };
    }

    // Toggle Text-to-Speech button
    if (toggleTextToSpeechBtn) {
        toggleTextToSpeechBtn.onclick = function() {
            this.classList.toggle("active");
        };
    }
});
