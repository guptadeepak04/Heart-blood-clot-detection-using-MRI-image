const inpFile = document.getElementById("file");
const previewContainer = document.getElementById("image-preview");
const previewImage = previewContainer.querySelector(".image-preview__img");
const previewText = previewContainer.querySelector(".image-preview__default-text");
const predictBtn = document.getElementById("predict-btn");
const resultBox = document.getElementById("result");
const loadingText = document.getElementById("loading");
const segMaskImage = document.getElementById("seg-mask");
const segOverlayImage = document.getElementById("seg-overlay");
const segmentationPanel = document.querySelector(".segmentation-panel");

inpFile.addEventListener("change", function () {
    const file = this.files[0];
    if (file) {
        const reader = new FileReader();
        previewText.style.display = "none";
        previewImage.style.display = "block";
        reader.addEventListener("load", function () {
            previewImage.setAttribute("src", this.result);
        });
        reader.readAsDataURL(file);
    }
});

document.getElementById("upload-form").addEventListener("submit", async function (e) {
    e.preventDefault();

    if (!inpFile.files[0]) {
        alert("Please select an image first!");
        return;
    }

    predictBtn.disabled = true;
    predictBtn.innerText = "Analyzing...";
    loadingText.classList.remove("hidden");
    resultBox.classList.add("hidden");

    const formData = new FormData();
    formData.append("file", inpFile.files[0]);

    try {
        const response = await fetch("/predict", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        if (data.error) {
            throw new Error(data.error);
        }

        loadingText.classList.add("hidden");
        resultBox.classList.remove("hidden");

        const labelElem = document.getElementById("res-label");
        labelElem.innerText = data.label;
        document.getElementById("res-confidence").innerText = data.confidence;

        if (data.segmentation_enabled && data.segmentation_mask && data.segmentation_overlay) {
            segmentationPanel.classList.remove("hidden");
            segMaskImage.src = data.segmentation_mask;
            segOverlayImage.src = data.segmentation_overlay;
        } else {
            segmentationPanel.classList.add("hidden");
            segMaskImage.removeAttribute("src");
            segOverlayImage.removeAttribute("src");
        }

        resultBox.className = "result-box " + data.status;
    } catch (error) {
        console.error(error);
        alert(error.message || "Something went wrong!");
        loadingText.classList.add("hidden");
    } finally {
        predictBtn.disabled = false;
        predictBtn.innerText = "Analyze MRI";
    }
});
