document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("upload-form");
    const resultSection = document.getElementById("result-section");

    form.addEventListener("submit", (event) => {
        event.preventDefault(); // Prevent form submission

        // Simulate prediction result for demonstration purposes
        const prediction = "COVID-19 Positive"; // Replace with your dynamic prediction result
        const imageUrl = "{{ url_for('static', filename=image) }}"; // Replace with your image URL

        // Animate closing of the previous result (if any)
        if (resultSection.classList.contains("visible")) {
            resultSection.classList.add("closing");
            setTimeout(() => {
                resultSection.innerHTML = ""; // Clear previous content
                resultSection.classList.remove("visible", "closing");
                showNewResult(prediction, imageUrl); // Show the new result
            }, 500); // Match this with the CSS animation duration
        } else {
            showNewResult(prediction, imageUrl); // Show the new result
        }
    });

    function showNewResult(prediction, imageUrl) {
        // Create new prediction result
        const resultContent = `
            <h2>Prediction Result</h2>
            <p>The image is classified as: <strong>${prediction}</strong></p>
            <img src="${imageUrl}" alt="Uploaded Image">
        `;
        resultSection.innerHTML = resultContent;
        resultSection.classList.add("visible");
    }
});
