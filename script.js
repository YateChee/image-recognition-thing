const video = document.getElementById("video");
const startButton = document.getElementById("startButton");
const status = document.getElementById("status");
const URL = "./model/";

let model;

async function startWebcam() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        status.textContent = "This browser does not support webcam access.";
        return;
    }

    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: {
                facingMode: "user",
                width: { ideal: 640 },
                height: { ideal: 480 }
            },
            audio: false
        });

        video.srcObject = stream;
        await video.play();
        status.textContent = "Webcam is live.";
        startButton.disabled = true;
    } catch (error) {
        console.error(error);
        status.textContent = "Camera permission was denied or no camera was found.";
    }
}

async function loadModel() {
    model = await tmImage.load(
        URL + "model.json",
        URL + "metadata.json"
    );

    console.log("Model loaded!");
}

startButton.addEventListener("click", startWebcam);

loadModel();