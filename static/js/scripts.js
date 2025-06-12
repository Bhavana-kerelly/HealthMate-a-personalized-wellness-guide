document.addEventListener("DOMContentLoaded", function () {
    const buttons = document.querySelectorAll(".btn");
    buttons.forEach(button => {
        button.addEventListener("mouseover", function () {
            this.style.transform = "scale(1.1)";
            this.style.transition = "0.3s ease-in-out";
        });
        button.addEventListener("mouseout", function () {
            this.style.transform = "scale(1)";
        });
    });

    // Fade-in animation for the container
    const container = document.querySelector(".container");
    container.style.opacity = "0";
    setTimeout(() => {
        container.style.opacity = "1";
        container.style.transition = "opacity 1s ease-in-out";
    }, 500);

    // Submit button animation
    const form = document.querySelector("form");
    if (form) {
        form.addEventListener("submit", function () {
            const submitBtn = document.querySelector("button[type='submit']");
            submitBtn.innerHTML = "Processing...";
            submitBtn.style.background = "#ffc107";
        });
    }
});
