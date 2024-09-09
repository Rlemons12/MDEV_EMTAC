document.getElementById("sidebarCollapse").onclick = function() {
    document.querySelector(".main-container").classList.toggle("collapsed");
};

document.getElementById("closeSidebar").onclick = function() {
    document.querySelector(".main-container").classList.toggle("collapsed");
};

// Add functionality to toggle buttons to highlight when active
document.getElementById("toggle-voice").onclick = function() {
    this.classList.toggle("active");
};

document.getElementById("toggle-text-to-speech").onclick = function() {
    this.classList.toggle("active");
};
