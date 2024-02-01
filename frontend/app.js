
const uploadButton = document.getElementById('up-btn');
const imageInput = document.getElementById('image-input');
const outputSection = document.getElementById('output-section');

uploadButton.addEventListener('onclick', function () {
    console.log('inja0')
    imageInput.click();
});

imageInput.addEventListener('change', function () {
    console.log('inja1')
    const file = imageInput.files[0];
    const formData = new FormData();
    formData.append('image', file);

    fetch('http://127.0.0.1:8000/api/core/process_image/', {
        method: 'POST',
        body: formData
    })
        .then(response => response.json())
        .then(data => {
            if (data.class)
                showOutput(data.class);
        })
        .catch(error => {
            showOutput('Error: ' + error.message);
        });
});

function showOutput(imageClass) {
    console.log('inja2')
    outputSection.textContent = 'Class: ' + imageClass;
}
