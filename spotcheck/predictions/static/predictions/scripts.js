document.getElementById('imageUpload').addEventListener('change', function() {
    const uploadTextElement = document.getElementById('uploadText');
    const checkSpotButton = document.getElementById('checkSpotButton');
    const file = this.files[0];

    if (file) {
        uploadTextElement.textContent = file.name; // Show the selected file name
        checkSpotButton.disabled = false; // Enable the button
        checkSpotButton.classList.remove('disabled-button'); // Remove the disabled class
        checkSpotButton.classList.add('enabled-button'); // Add the enabled class
    } else {
        uploadTextElement.textContent = 'Click to Upload an Image'; // Revert text if no file is selected
        checkSpotButton.disabled = true; // Disable the button
        checkSpotButton.classList.add('disabled-button'); // Apply the disabled class
        checkSpotButton.classList.remove('enabled-button'); // Remove the enabled class
    }
});

function submitForm() {
    const formData = new FormData(document.getElementById('uploadForm'));
    const spinner = document.getElementById('loadingSpinner');
    const resultContainer = document.getElementById('result');

    // Show spinner
    spinner.style.display = 'block';

    // Simulate a delay to represent processing time (1.5 seconds)
    setTimeout(() => {
        // Hide spinner
        spinner.style.display = 'none';

        // Proceed with the fetch request
        fetch('/predict/', {
            method: 'POST',
            body: formData,
        })
        .then(response => response.json())
        .then(data => {
            const confidencePct = (data.confidence * 100).toFixed(1);
            resultContainer.innerHTML = `
                <p>Prediction: ${data.prediction} (${confidencePct}% confidence)</p>
                <p class="explain-caption">The heatmap shows the regions the model focused on. Red = high influence on the prediction.</p>
                <div class="image-pair">
                    <figure>
                        <img class="result-image" src="data:image/jpeg;base64,${data.image}" alt="Uploaded Image">
                        <figcaption>Input</figcaption>
                    </figure>
                    <figure>
                        <img class="result-image" src="data:image/jpeg;base64,${data.heatmap}" alt="Grad-CAM Heatmap">
                        <figcaption>Grad-CAM</figcaption>
                    </figure>
                </div>
                <button onclick="reloadPage()">Try Another Photo</button>
            `;
            document.getElementById('uploadForm').style.display = 'none';
            resultContainer.style.display = 'block';
        })
        .catch(error => {
            console.error('Error:', error);
        });
    }, 1500); // Delay of 1.5 seconds
}

function reloadPage() {
    location.reload();
}