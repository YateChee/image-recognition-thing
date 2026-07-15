// const URL = "./model/";

// let model;

// async function loadModel() {
//     model = await tmImage.load(
//         URL + "model.json",
//         URL + "metadata.json"
//     );

//     console.log("Model loaded!");
// }

const URL = "./";

let classifierModel;
let metadataObj;
let labels = [];
let handDetector;
let modelLoaded = false;
let normalizationPreset = 'palm_scale';
let capturedExamples = []; // Store feature vectors for debugging
let lastFeatureVector = null; // Track current feature vector for display
let lastLandmarks = null; // store last raw landmarks (for future repro / preset sweep)
const video = document.getElementById("video");
const startButton = document.getElementById("startButton");
const status = document.getElementById("status");
const labelContainer = document.getElementById("labelContainer");
const protocolWarning = document.getElementById("protocolWarning");
const errorBox = document.getElementById("errorBox");
const localLoader = document.getElementById("localLoader");
const modelJsonInput = document.getElementById("modelJson");
const metadataInput = document.getElementById("metadataJson");
const weightsBinInput = document.getElementById("weightsBin");
const loadFilesButton = document.getElementById("loadFilesButton");
const featureVectorDisplay = document.getElementById("featureVectorDisplay");
const captureExampleButton = document.getElementById("captureExampleButton");
const downloadFeaturesButton = document.getElementById("downloadFeaturesButton");
const captureStatusBox = document.getElementById("captureStatusBox");
const analyzeExamplesButton = document.getElementById("analyzeExamplesButton");

startButton.disabled = true;
loadFilesButton.disabled = true;
status.textContent = "Loading model...";
labelContainer.textContent = "Waiting for model load...";

if (window.location.protocol === "file:") {
    protocolWarning.textContent = "You are opening this page via file://. Model files often fail to load in this mode. Run serve.ps1 and open http://localhost:8000 instead.";
    protocolWarning.style.display = "block";
} else {
    protocolWarning.style.display = "none";
}

window.addEventListener("error", (event) => {
    showError(`Error: ${event.message}`);
});
window.addEventListener("unhandledrejection", (event) => {
    showError(`Unhandled promise rejection: ${event.reason}`);
});

function showError(message) {
    errorBox.style.display = "block";
    errorBox.innerHTML += `${message}<br>`;
}

function updateFileButtonState() {
    loadFilesButton.disabled = !(modelJsonInput.files.length && metadataInput.files.length && weightsBinInput.files.length);
}

modelJsonInput.addEventListener("change", updateFileButtonState);
metadataInput.addEventListener("change", updateFileButtonState);
weightsBinInput.addEventListener("change", updateFileButtonState);
loadFilesButton.addEventListener("click", loadModelFromFiles);

async function loadModel() {
    try {
        status.textContent = "Loading classifier model...";
        // load the exported classifier (expects feature vector length 87)
        classifierModel = await tf.loadLayersModel(URL + "model.json");

        // load metadata to get labels if present
        try {
            const metaResp = await fetch(URL + "metadata.json");
            if (metaResp.ok) {
                metadataObj = await metaResp.json();
                labels = metadataObj.labels || metadataObj.classes || [];
            }
        } catch (merr) {
            console.warn("Failed to load metadata.json", merr);
        }

        modelLoaded = true;
        status.textContent = "Model loaded. Click Start webcam.";
        labelContainer.textContent = "Model ready. Click Start webcam.";
        startButton.disabled = false;
        console.log("Classifier loaded", classifierModel);
    } catch (error) {
        console.error("Model load failed", error);
        showError(`Model load failed: ${error.message || error}`);
        status.textContent = "Failed to load the model from the local path. Use the file picker below if you are running from file:// or open via localhost.";
        labelContainer.textContent = "Model failed to load. Load files manually below or run serve.ps1.";
        localLoader.style.display = "block";
        startButton.disabled = true;
    }
}

async function loadModelFromFiles() {
    const modelFile = modelJsonInput.files[0];
    const metadataFile = metadataInput.files[0];
    const weightsFile = weightsBinInput.files[0];

    if (!modelFile || !metadataFile || !weightsFile) {
        return;
    }

    try {
        status.textContent = "Loading model from local files...";
        labelContainer.textContent = "Loading model from local files...";

        // load classifier from provided JSON + weights files
        classifierModel = await tf.loadLayersModel(tf.io.browserFiles([modelFile, weightsFile]));

        // read metadata JSON
        const metadataText = await metadataFile.text();
        try {
            metadataObj = JSON.parse(metadataText);
            labels = metadataObj.labels || metadataObj.classes || [];
        } catch (jerr) {
            console.warn("Failed to parse metadata JSON", jerr);
        }

        modelLoaded = true;
        status.textContent = "Local classifier loaded. Click Start webcam.";
        labelContainer.textContent = "Model ready. Click Start webcam.";
        startButton.disabled = false;
        localLoader.style.display = "none";
    } catch (error) {
        console.error("Local model file load failed", error);
        showError(`Local model file load failed: ${error.message || error}`);
        status.textContent = "Local model file load failed. Check your selected files.";
        labelContainer.textContent = "Local model failed to load.";
    }
}

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

        // ensure hand detector is loaded
        if (!handDetector) {
            try {
                status.textContent = "Loading hand detector...";
                handDetector = await handpose.load();
                console.log("handpose detector loaded");
            } catch (derr) {
                console.error("Hand detector failed to load", derr);
                showError(`Hand detector failed to load: ${derr.message || derr}`);
                labelContainer.textContent = "Hand detector failed to load.";
                return;
            }
        }

        if (modelLoaded && handDetector) {
            predictLoop();
        } else if (!modelLoaded) {
            status.textContent = "Webcam is live, but the classifier is not loaded.";
            labelContainer.textContent = "Load the classifier model first.";
        }
    } catch (error) {
        console.error(error);
        status.textContent = "Camera permission was denied or no camera was found.";
    }
}

async function predictLoop() {
    if (video.paused || video.ended) return;

    try {
        // estimate hands using handpose
        const hands = await handDetector.estimateHands(video, false);

        if (!hands || hands.length === 0) {
            labelContainer.textContent = "No hand detected.";
            requestAnimationFrame(predictLoop);
            return;
        }

        // use first detected hand
        const hand = hands[0];
        const landmarks = hand.landmarks; // array of 21 [x,y,z]

        // Compute hand bounding box to isolate hand from background (like TM does)
        let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
        for (const p of landmarks) {
            minX = Math.min(minX, p[0]);
            maxX = Math.max(maxX, p[0]);
            minY = Math.min(minY, p[1]);
            maxY = Math.max(maxY, p[1]);
        }
        const handBbox = { minX, maxX, minY, maxY, centerX: (minX + maxX) / 2, centerY: (minY + maxY) / 2, width: maxX - minX, height: maxY - minY };

        const features = extract87FeaturesFromLandmarks(landmarks, handBbox);
        lastFeatureVector = features; // Store for capture button
        lastLandmarks = landmarks.map(p => [p[0], p[1], p[2] != null ? p[2] : 0]); // copy raw landmarks

        // Display feature vector in inspector (show first 10 values truncated)
        const displayStr = `[${features.slice(0, 10).map(v => v.toFixed(3)).join(", ")}${features.length > 10 ? ", ..." : ""}] (len: ${features.length})`;
        featureVectorDisplay.textContent = displayStr;
        
        // Log debug info to console to verify hand isolation
        console.log({
            videoResolution: [video.videoWidth, video.videoHeight],
            rawLandmarksample: landmarks.slice(0, 2),
            handBbox,
            featureVector_first10: [...features].slice(0, 10),
            normalizationPreset,
            topPrediction: null
        });
        const out = await tf.tidy(() => {
            const input = tf.tensor([features], [1, features.length]);
            let logits = classifierModel.predict(input);
            if (Array.isArray(logits)) logits = logits[0];
            // convert to probabilities
            const probs = tf.softmax(logits);
            return probs.arraySync()[0];
        });

        if (out && out.length) {
            // map to labels if available
            const mapped = out.map((p, i) => ({ index: i, prob: p, label: labels[i] || `Label ${i}` }));
            mapped.sort((a, b) => b.prob - a.prob);
            const top = mapped.slice(0, 3);
            console.log('Top prediction:', top[0], 'feature vector length:', features.length, 'preset:', normalizationPreset, 'landmarks sample:', landmarks.slice(0, 3));
            labelContainer.innerHTML = top.map(t => `${t.label}: ${(t.prob * 100).toFixed(1)}%`).join("<br>");
            if (top[0] && top[0].label) {
                speakWord(top[0].label);
            }
        } else {
            labelContainer.textContent = "No prediction result.";
        }
    } catch (error) {
        console.error("Prediction failed", error);
        showError(`Prediction failed: ${error.message || error}`);
        labelContainer.textContent = "Prediction failed. See console or error box.";
    }

    requestAnimationFrame(predictLoop);
}

function extract87FeaturesFromLandmarks(landmarks, handBbox) {
    // Normalize landmarks relative to isolated hand bounding box (like TM does)
    // This centers the hand and scales it to be view-invariant
    
    // Create working bbox
    let bbox = { ...handBbox };
    
    // For square normalization, use max(width, height) for both dimensions
    if (normalizationPreset === 'isolated_hand_2d_square') {
        const size = Math.max(bbox.width, bbox.height);
        const dx = (size - bbox.width) / 2;
        const dy = (size - bbox.height) / 2;
        bbox.minX -= dx;
        bbox.maxX += dx;
        bbox.minY -= dy;
        bbox.maxY += dy;
        bbox.width = size;
        bbox.height = size;
        bbox.centerX = (bbox.minX + bbox.maxX) / 2;
        bbox.centerY = (bbox.minY + bbox.maxY) / 2;
    }
    
    // Optional padding around hand bbox (TM might use this)
    const padding_factor = normalizationPreset === 'isolated_hand_2d_padded' ? 0.2 : 0;
    if (padding_factor > 0) {
        const padX = bbox.width * padding_factor;
        const padY = bbox.height * padding_factor;
        bbox.minX -= padX;
        bbox.maxX += padX;
        bbox.minY -= padY;
        bbox.maxY += padY;
        bbox.width = bbox.maxX - bbox.minX;
        bbox.height = bbox.maxY - bbox.minY;
        bbox.centerX = (bbox.minX + bbox.maxX) / 2;
        bbox.centerY = (bbox.minY + bbox.maxY) / 2;
    }
    
    const normalized = landmarks.map(p => {
        const nx = (p[0] - bbox.centerX) / (bbox.width || 1);
        let ny = (p[1] - bbox.centerY) / (bbox.height || 1);
        
        const nz = (p[2] != null ? p[2] : 0);
        return [nx, ny, nz];
    });

    // If using isolated_hand_2d variants, just use x,y (no z) for truly isolated representation
    if (normalizationPreset.startsWith('isolated_hand_2d')) {
        const out = new Float32Array(87);
        let idx = 0;
        for (let i = 0; i < 21; i++) {
            out[idx++] = normalized[i][0];
            out[idx++] = normalized[i][1];
            // Skip z - TM trains on 2D crop images
        }
        // Fill rest with zeros
        while (idx < 87) out[idx++] = 0;
        return out;
    }

    // If using raw_coords_only, just use the 63 normalized landmarks (with z)
    if (normalizationPreset === 'raw_coords_only') {
        const out = new Float32Array(87);
        let idx = 0;
        for (let i = 0; i < 21; i++) {
            out[idx++] = normalized[i][0];
            out[idx++] = normalized[i][1];
            out[idx++] = normalized[i][2];
        }
        // Fill rest with zeros
        while (idx < 87) out[idx++] = 0;
        return out;
    }

    // For other presets: use full feature extraction with transforms
    const transformed = normalized.map(p => [...p]);

    function dist(a, b) {
        const dx = a[0] - b[0];
        const dy = a[1] - b[1];
        const dz = (a[2] || 0) - (b[2] || 0);
        return Math.sqrt(dx * dx + dy * dy + dz * dz);
    }

    // If using palm_scale, center at wrist and rotate so wrist->middle MCP aligns with +X axis
    if (normalizationPreset === 'palm_scale') {
        const wrist = normalized[0];
        const mid = normalized[9];
        const dx = mid[0] - wrist[0];
        const dy = mid[1] - wrist[1];
        const angle = Math.atan2(dy, dx);
        const cosA = Math.cos(angle);
        const sinA = Math.sin(angle);

        for (let i = 0; i < transformed.length; i++) {
            const tx = transformed[i][0] - wrist[0];
            const ty = transformed[i][1] - wrist[1];
            const rx = cosA * tx + sinA * ty;
            const ry = -sinA * tx + cosA * ty;
            transformed[i][0] = rx;
            transformed[i][1] = ry;
        }
    }

    // Compute the flattened features from transformed landmarks
    const out = new Float32Array(87);
    let idx = 0;
    for (let i = 0; i < 21; ++i) {
        const p = transformed[i];
        out[idx++] = p[0];
        out[idx++] = p[1];
        out[idx++] = p[2];
    }

    function vec(a, b) { return [b[0] - a[0], b[1] - a[1], (b[2] || 0) - (a[2] || 0)]; }
    function dot(u, v) { return u[0]*v[0]+u[1]*v[1]+u[2]*v[2]; }
    function norm(u) { return Math.sqrt(dot(u,u)) || 1e-6; }

    const fingertips = [4,8,12,16,20];
    for (let i=0;i<fingertips.length;i++){
        for (let j=i+1;j<fingertips.length;j++){
            out[idx++] = dist(transformed[fingertips[i]], transformed[fingertips[j]]);
        }
    }

    const wristT = transformed[0];
    for (let i=0;i<fingertips.length;i++){
        out[idx++] = dist(transformed[fingertips[i]], wristT);
    }

    const mcpPairs = [[2,4],[5,8],[9,12],[13,16],[17,20]];
    for (let i=0;i<mcpPairs.length;i++){
        const [a,b] = mcpPairs[i];
        out[idx++] = dist(transformed[a], transformed[b]);
    }

    const dirPairs = [[4,2],[8,5],[12,9],[16,13],[20,17]];
    const vectors = dirPairs.map(([tip,mcp]) => vec(transformed[mcp], transformed[tip]));
    const adj = [[0,1],[1,2],[2,3],[3,4]];
    for (let k=0;k<adj.length;k++){
        const [a,b]=adj[k];
        const u = vectors[a];
        const v = vectors[b];
        const cosine = dot(u,v)/(norm(u)*norm(v));
        out[idx++] = cosine;
    }

    while (idx < 87) out[idx++] = 0;

    // Apply normalization scale (after transform)
    if (normalizationPreset === 'palm_scale') {
        const scale = dist(transformed[0], transformed[9]) || 1;
        for (let i=0;i<63;i+=3){ out[i] /= scale; out[i+1] /= scale; out[i+2] /= scale; }
        for (let i=63;i<87;i++){ if (i < 87-4) out[i] /= scale; }
    }

    return out;
}

// Feature capture helpers for debugging
function captureExample() {
    if (!lastFeatureVector) {
        captureStatusBox.textContent = "❌ No feature vector available. Detect a hand first.";
        captureStatusBox.style.color = "#dc2626";
        return;
    }
    
    // Prompt user for label
    const label = prompt("Enter label for this example (e.g., 'thumbs_up_1'):");
    if (!label) return;
    
    const example = {
        label: label,
        preset: normalizationPreset,
        features: lastFeatureVector,
        landmarks: lastLandmarks,
        timestamp: new Date().toISOString()
    };
    
    capturedExamples.push(example);
    captureStatusBox.textContent = `✓ Captured example: ${label} (total: ${capturedExamples.length})`;
    captureStatusBox.style.color = "#059669";
}

function downloadFeatures() {
    if (capturedExamples.length === 0) {
        alert("No examples captured yet.");
        return;
    }
    
    const json = JSON.stringify(capturedExamples, null, 2);
    const blob = new Blob([json], { type: "application/json" });
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `feature_examples_${new Date().toISOString().slice(0, 10)}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    window.URL.revokeObjectURL(url);
    
    captureStatusBox.textContent = `✓ Downloaded ${capturedExamples.length} examples`;
    captureStatusBox.style.color = "#059669";
}

startButton.addEventListener("click", startWebcam);
loadModel();

// wire normalization selector when DOM ready
const normalizationSelect = document.getElementById('normalizationSelect');
if (normalizationSelect) {
    normalizationSelect.value = normalizationPreset;
    normalizationSelect.addEventListener('change', () => {
        normalizationPreset = normalizationSelect.value;
        console.log('Normalization preset set to', normalizationPreset);
    });
}

// wire feature capture buttons
if (captureExampleButton) {
    captureExampleButton.addEventListener('click', captureExample);
}
if (downloadFeaturesButton) {
    downloadFeaturesButton.addEventListener('click', downloadFeatures);
}

async function analyzeExamples() {
    if (!modelLoaded) {
        alert('Model not loaded yet. Load model first.');
        return;
    }

    // Try known filenames
    const candidates = [
        'feature_examples_2026-07-15.json',
        'feature_examples_2026-07-15 (1).json'
    ];

    let examples = null;
    for (const fn of candidates) {
        try {
            const resp = await fetch(fn);
            if (resp.ok) {
                examples = await resp.json();
                console.log('Loaded examples from', fn);
                break;
            }
        } catch (e) {
            // ignore
        }
    }

    // If fetch failed (often happens when opening file://), prompt user to pick the JSON file
    if (!examples) {
        console.warn('Could not fetch examples; prompting for local file selection.');
        const fileInput = document.getElementById('examplesFileInput');
        if (!fileInput) {
            alert('No examples file found and no file picker available. Serve the folder over HTTP or use the file picker.');
            return;
        }

        // trigger file selection and wait for user pick
        const pickPromise = new Promise((resolve) => {
            fileInput.onchange = async (ev) => {
                const f = fileInput.files && fileInput.files[0];
                if (!f) return resolve(null);
                try {
                    const text = await f.text();
                    const parsed = JSON.parse(text);
                    resolve(parsed);
                } catch (err) {
                    console.error('Failed to read selected file', err);
                    resolve(null);
                }
            };
            fileInput.click();
        });

        examples = await pickPromise;
        if (!examples) {
            alert('No examples loaded.');
            return;
        }
    }

    const results = [];
    const presetsToTest = ['isolated_hand_2d','isolated_hand_2d_square','isolated_hand_2d_padded','isolated_hand_2d_yflip','palm_scale','raw_coords_only'];
    for (const ex of examples) {
        // features may be object with numeric string keys
        let feat = ex.features;
        let arr;
        if (Array.isArray(feat)) arr = feat;
        else {
            arr = new Array(87).fill(0);
            for (const k of Object.keys(feat)) {
                const idx = Number(k);
                if (!Number.isNaN(idx) && idx >= 0 && idx < 87) arr[idx] = feat[k];
            }
        }

        try {
            const probs = await tf.tidy(() => {
                const input = tf.tensor([arr], [1, arr.length]);
                let logits = classifierModel.predict(input);
                if (Array.isArray(logits)) logits = logits[0];
                return tf.softmax(logits).arraySync()[0];
            });

            const mapped = probs.map((p, i) => ({ i, p, label: labels[i] || `Label ${i}` }));
            mapped.sort((a, b) => b.p - a.p);
            const top = mapped[0];
            const correct = (ex.label && (ex.label.toLowerCase() === (top.label || '').toLowerCase()));

            const record = { exampleLabel: ex.label, predicted: top.label, prob: top.p, correct };

            // If raw landmarks available, run preset sweep to see which preset helps this example
            if (ex.landmarks && Array.isArray(ex.landmarks) && ex.landmarks.length >= 21) {
                const trueIndex = labels.findIndex(l => l && ex.label && l.toLowerCase() === ex.label.toLowerCase());
                const sweep = [];
                const prevPreset = normalizationPreset;
                for (const pset of presetsToTest) {
                    normalizationPreset = pset;
                    const bbox = computeHandBboxFromLandmarks(ex.landmarks);
                    const feat = extract87FeaturesFromLandmarks(ex.landmarks, bbox);
                    const probs2 = await tf.tidy(() => {
                        const input = tf.tensor([feat], [1, feat.length]);
                        let logits = classifierModel.predict(input);
                        if (Array.isArray(logits)) logits = logits[0];
                        return tf.softmax(logits).arraySync()[0];
                    });
                    const trueProb = (trueIndex >= 0 && trueIndex < probs2.length) ? probs2[trueIndex] : null;
                    sweep.push({ preset: pset, topPred: labels[probs2.indexOf(Math.max(...probs2))] || `Label ${probs2.indexOf(Math.max(...probs2))}`, topProb: Math.max(...probs2), trueProb });
                }
                normalizationPreset = prevPreset;
                // pick best preset by trueProb (if available)
                sweep.sort((a,b) => (b.trueProb || 0) - (a.trueProb || 0));
                record.sweep = sweep;
                record.bestPresetByTrueProb = sweep[0] || null;
            }

            results.push(record);
            console.log('Example', ex.label, '=>', top.label, (top.p*100).toFixed(1)+'%', 'correct?', correct, record.bestPresetByTrueProb ? `bestPreset:${record.bestPresetByTrueProb.preset}` : '');
        } catch (err) {
            console.error('Prediction error for example', ex.label, err);
            results.push({ exampleLabel: ex.label, error: err.message || String(err) });
        }
    }

    // Summary
    const correctCount = results.filter(r => r.correct).length;
    const total = results.length;
    captureStatusBox.textContent = `Analyzed ${total} examples — ${correctCount} correct`;
    captureStatusBox.style.color = correctCount === total ? '#059669' : '#b45309';
    console.table(results);
}
if (analyzeExamplesButton) {
    analyzeExamplesButton.addEventListener('click', analyzeExamples);
}

function computeHandBboxFromLandmarks(landmarks) {
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (const p of landmarks) {
        minX = Math.min(minX, p[0]);
        maxX = Math.max(maxX, p[0]);
        minY = Math.min(minY, p[1]);
        maxY = Math.max(maxY, p[1]);
    }
    return { minX, maxX, minY, maxY, centerX: (minX + maxX) / 2, centerY: (minY + maxY) / 2, width: (maxX - minX), height: (maxY - minY) };
}

let lastSpokenWord = ""; 
let isSpeaking = false;
const voiceSelect = document.getElementById("voice-select");

function getSelectedVoice() {
    if (!voiceSelect) return null;
    const selected = voiceSelect.value;
    const voices = window.speechSynthesis.getVoices();
    return voices.find(v => v.name === selected) || voices.find(v => v.default) || null;
}

function populateVoiceList() {
    if (!voiceSelect || !window.speechSynthesis) return;
    const voices = window.speechSynthesis.getVoices();
    if (!voices || voices.length === 0) return;
    voiceSelect.innerHTML = "";
    voices.forEach(v => {
        const option = document.createElement("option");
        option.value = v.name;
        option.textContent = `${v.name} (${v.lang})${v.default ? " [default]" : ""}`;
        voiceSelect.appendChild(option);
    });
    if (!voiceSelect.value && voices.length) {
        voiceSelect.value = voices[0].name;
    }
}

if (window.speechSynthesis) {
    populateVoiceList();
    window.speechSynthesis.onvoiceschanged = populateVoiceList;
}

/**
 * Speaks the recognized word out loud with volume control and a natural sentence flow pause.
 * @param {string} word - The word recognized by the Teachable Machine model.
 */
function speakWord(word) {
    // 1. Ignore background noise or empty frames
    if (!word || word.toLowerCase() === "background" || word.toLowerCase() === "nothing") {
        return;
    }

    // 2. Prevent spamming the same word repeatedly while speaking
    if (word === lastSpokenWord || isSpeaking) {
        return; 
    }

    // 3. Get the current volume from the HTML slider
    const volumeSlider = document.getElementById("volume-control");
    const currentVolume = volumeSlider ? parseFloat(volumeSlider.value) : 1.0;

    // 4. Set up the speech engine
    const utterance = new SpeechSynthesisUtterance(word);
    utterance.volume = currentVolume; // Set volume (0 to 1)
    utterance.rate = 1.0;             // Speed (0.1 to 10)
    utterance.pitch = 1.0;            // Pitch (0 to 2)

    
    //test
    const selectedVoice = getSelectedVoice();
    if (selectedVoice) {
        utterance.voice = selectedVoice;
    }
//test

    // Set locks so it doesn't overlap
    isSpeaking = true;
    lastSpokenWord = word;

    // Speak!
    window.speechSynthesis.speak(utterance);

    // 5. This is the sentence flow magic!
    // Once the browser finishes speaking, wait a tiny "breathing gap" (250ms)
    // before unlocking, allowing the next word to flow naturally.
    utterance.onend = () => {
        setTimeout(() => {
            isSpeaking = false;
        }, 250); // Increase this number for a longer pause, decrease it for a shorter pause
    };
}


async function predict() {
    const prediction = await model.predict(webcam.canvas);
    
    // ... [Your existing code that finds the bestMatchLabel goes here] ...

    // ADD THIS EXACT LINE to trigger the speaking logic:
    speakWord(bestMatchLabel);      
}


// let lastSpokenWord = ""; 
// let isSpeaking = false;

/**
 * Speaks the recognized word out loud with volume control and a natural sentence flow pause.
 * @param {string} word - The word recognized by the Teachable Machine model.
 */
function speakWord(word) {
    // 1. Ignore background noise or empty frames
    if (!word || word.toLowerCase() === "background" || word.toLowerCase() === "nothing") {
        return;
    }

    // 2. Prevent spamming the same word repeatedly while speaking
    if (word === lastSpokenWord || isSpeaking) {
        return; 
    }

    // 3. Get the current volume from the HTML slider
    const volumeSlider = document.getElementById("volume-control");
    const currentVolume = volumeSlider ? parseFloat(volumeSlider.value) : 1.0;

    // 4. Set up the speech engine
    const utterance = new SpeechSynthesisUtterance(word);
    utterance.volume = currentVolume; // Set volume (0 to 1)
    utterance.rate = 1.0;             // Speed (0.1 to 10)
    utterance.pitch = 1.0;            // Pitch (0 to 2)

    // Set locks so it doesn't overlap
    isSpeaking = true;
    lastSpokenWord = word;

    // Speak!
    window.speechSynthesis.speak(utterance);

    // 5. This is the sentence flow magic!
    // Once the browser finishes speaking, wait a tiny "breathing gap" (250ms)
    // before unlocking, allowing the next word to flow naturally.
    utterance.onend = () => {
        setTimeout(() => {
            isSpeaking = false;
        }, 400); // Increase this number for a longer pause, decrease it for a shorter pause
    };
}







// === SIDEBAR STATE & SPEECH RECOGNITION SETUP ===
let sidebarOpen = false;
let isListening = false;
let recognition;

// Initialize speech recognition if supported by the browser
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
if (SpeechRecognition) {
    recognition = new SpeechRecognition();
    recognition.continuous = false;     // Stops listening after you finish speaking a sentence
    recognition.interimResults = false;  // Only returns final results

    // Handle results
    recognition.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        document.getElementById("speech-transcript").innerText = transcript;
        processSpokenText(transcript);
    };

    recognition.onend = () => {
        isListening = false;
        document.getElementById("listen-btn").innerText = "🎙️ Start Listening";
        document.getElementById("listen-btn").style.backgroundColor = "#007bff";
    };

    recognition.onerror = (event) => {
        console.error("Speech recognition error", event.error);
        stopListening();
    };
} else {
    console.warn("Speech recognition is not supported in this browser.");
}

// === FIX 1: Defined toggleSidebar() ===
// This handles sliding the panel in and out from the right side.
function toggleSidebar() {
    const sidebar = document.getElementById("sidebar");
    const toggleBtn = document.getElementById("sidebar-toggle");
    
    if (sidebarOpen) {
        sidebar.style.right = "-350px";
        toggleBtn.innerText = "🗣️ Speak to Sign";
        stopListening();
    } else {
        sidebar.style.right = "0px";
        toggleBtn.innerText = "❌ Close Sidebar";
    }
    sidebarOpen = !sidebarOpen;
}

// === FIX 2: Defined toggleListening() and helper controls ===
// This activates the user's microphone.
function toggleListening() {
    if (!recognition) {
        alert("Speech recognition is not supported in this browser. Try Google Chrome!");
        return;
    }
    if (isListening) {
        stopListening();
    } else {
        startListening();
    }
}

function startListening() {
    isListening = true;
    const selectedLang = document.getElementById("speech-lang").value;
    recognition.lang = selectedLang; // Set language dynamically from UI

    document.getElementById("listen-btn").innerText = "🛑 Stop Listening";
    document.getElementById("listen-btn").style.backgroundColor = "#dc3545";
    document.getElementById("speech-transcript").innerText = "Listening...";
    recognition.start();
}

function stopListening() {
    if (isListening && recognition) {
        recognition.stop();
    }
}

// === YOUR DICTIONARY & MATCHING LOGIC ===
/**
 * Processes spoken sentences, checking for full phrases first, then individual words.
 * @param {string} sentence - The text spoken by the user.
 */
function processSpokenText(sentence) {
    // 1. Clean up punctuation and convert to lowercase
    const cleanSentence = sentence.replace(/[.,\/#!$%\^&\*;:{}=\-_`~()]/g, "").toLowerCase().trim();
    
    // Your dictionary of gifs
    const signDictionary = {
        // --- Full Phrases ---
        "thank you": "https://i.pinimg.com/originals/46/62/8a/46628a276183dcdc4173bf909e9e2f57.gif", 
        "i love you": "https://i.giphy.com/jsCGdKbS9suooCEFSi.webp", // Lowercased "i love you" to match cleanSentence

        // --- Single Words / Letters ---
        "hello": "https://media3.giphy.com/media/v1.Y2lkPTc5MGI3NjExaTlkanVuZTI5b3Rtc2k4ejl5Mm9zOTYwYTQzY25yN3l2aXNwMnV0NCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/3o7TKNKOfKlIhbD3gY/giphy.gif", 
        "goodbye": "https://media.tenor.com/LG8VHUujRWsAAAAM/bsl-goodbye-bsl.gif", // Lowercased "goodbye"
        "yes": "https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExaDk5cWkzcTgyOGhyNmVqZWV0cGlpbmVydWNmc3kzZHNxd2dhamhndyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/l4Jz0THKhQLo61NBK/giphy.gif",
        "no": "https://media4.giphy.com/media/v1.Y2lkPTc5MGI3NjExeWw2cnU0djh3eDAwb2M0MWY4NDQzbWJ6Mjg0NzN3ZmN4dm1qbWo1cCZlcD12MV9naWZzX3NlYXJjaCZjdD1n/l4Jz4faxuS1FiSEV2/200.webp",
        "sorry": "https://media2.giphy.com/media/v1.Y2lkPTc5MGI3NjExYXhmMDNhemNsNmYxejI1cGx2aWYyYnd3djVnanVrZ2c1Y2owdTlmZCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/3o7TKq0oNLk8ljH7vG/giphy.gif",
        "hungry": "https://media4.giphy.com/media/v1.Y2lkPTc5MGI3NjExMW5iY2I1cnI0ZWZ1aHI1NWcwYTd2MXExNzVzZXY3N2FnZGQxbG03dCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/l3vR0xkdFEz4tnfTq/giphy.gif",
    };

    let foundWordOrPhrase = "";
    let foundUrl = "";

    // 2. Check for FULL PHRASE matches first!
    for (let key in signDictionary) {
        if (key.includes(" ") && cleanSentence.includes(key)) {
            foundWordOrPhrase = key;
            foundUrl = signDictionary[key];
            break; 
        }
    }

    // 3. Fallback: If no phrase matched, split into individual words and search
    if (!foundUrl) {
        const words = cleanSentence.split(" ");
        for (let word of words) {
            if (signDictionary[word]) {
                foundWordOrPhrase = word;
                foundUrl = signDictionary[word];
                break; 
            }
        }
    }

    const placeholder = document.getElementById("sign-placeholder");
    const signImg = document.getElementById("sign-image");
    const label = document.getElementById("sign-word-label");

    // 4. Update the UI
    if (foundUrl) {
        placeholder.style.display = "none";
        signImg.src = foundUrl;
        signImg.style.display = "block";
        label.innerText = foundWordOrPhrase.toUpperCase();
    } else {
        placeholder.style.display = "block";
        placeholder.innerText = `No sign found for "${cleanSentence}". Try saying 'thank you', 'i love you', or 'goodbye'!`;
        signImg.style.display = "none";
        label.innerText = "";
    }
}







// === SIDEBAR STATE & SPEECH RECOGNITION SETUP ===
// let sidebarOpen = false;
// let isListening = false;
// let recognition;
let userManuallyStopped = false; // Tracks if the user clicked "Stop"

// Initialize speech recognition if supported by the browser

if (SpeechRecognition) {
    recognition = new SpeechRecognition();
    recognition.continuous = false;     
    recognition.interimResults = false;  

    // Handle results
    recognition.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        document.getElementById("speech-transcript").innerText = transcript;
        processSpokenText(transcript);
    };

    // Keep listening loop active!
    recognition.onend = () => {
        // If the user didn't manually press "Stop", auto-restart the microphone
        if (isListening && !userManuallyStopped) {
            try {
                recognition.start();
            } catch (e) {
                console.log("Recognition auto-restart sweep:", e);
            }
        } else {
            isListening = false;
            document.getElementById("listen-btn").innerText = "🎙️ Start Listening";
            document.getElementById("listen-btn").style.backgroundColor = "#007bff";
        }
    };

    recognition.onerror = (event) => {
        console.error("Speech recognition error", event.error);
        // Don't kill the session on quiet timeouts
        if (event.error !== 'no-speech') {
            stopListening();
        }
    };
} else {
    console.warn("Speech recognition is not supported in this browser.");
}

// Opens/Closes the Sidebar
function toggleSidebar() {
    const sidebar = document.getElementById("sidebar");
    const toggleBtn = document.getElementById("sidebar-toggle");
    
    if (sidebarOpen) {
        sidebar.style.right = "-350px";
        toggleBtn.innerText = "🗣️ Speak to Sign";
        stopListening();
    } else {
        sidebar.style.right = "0px";
        toggleBtn.innerText = "❌ Close Sidebar";
    }
    sidebarOpen = !sidebarOpen;
}

// Starts/Stops listening to the microphone
function toggleListening() {
    if (!recognition) {
        alert("Speech recognition is not supported in this browser. Try Google Chrome!");
        return;
    }
    if (isListening) {
        userManuallyStopped = true;
        stopListening();
    } else {
        userManuallyStopped = false;
        startListening();
    }
}

function startListening() {
    isListening = true;
    const selectedLang = document.getElementById("speech-lang").value;
    recognition.lang = selectedLang; 

    document.getElementById("listen-btn").innerText = "🛑 Stop Listening";
    document.getElementById("listen-btn").style.backgroundColor = "#dc3545";
    document.getElementById("speech-transcript").innerText = "Listening...";
    recognition.start();
}

function stopListening() {
    isListening = false;
    if (recognition) {
        recognition.stop();
    }
}
