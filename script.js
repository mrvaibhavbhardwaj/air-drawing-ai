import {
    HandLandmarker,
    FilesetResolver
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/vision_bundle.js";

// DOM Elements
const video = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx = canvasElement.getContext("2d");
const loading = document.getElementById("loading");
const colorsContainer = document.getElementById("colors");
const modeLabel = document.getElementById("mode-label");
const sizeVal = document.getElementById("size-val");
const clearBtn = document.getElementById("clear-btn");
const saveBtn = document.getElementById("save-btn");

// State
let handLandmarker = undefined;
let webcamRunning = false;
let lastVideoTime = -1;
let brushSize = 6;
let isDrawing = false;
let lastDrawPoint = null;

// Palette
const PALETTE = [
    { name: "Coral", hex: "#507FFF" }, // OpenCV BGR mapped to Hex roughly
    { name: "Sky", hex: "#FFB450" },
    { name: "Mint", hex: "#8CDC78" },
    { name: "Sun", hex: "#00D2FF" },
    { name: "Lilac", hex: "#DC82FF" },
    { name: "White", hex: "#FFFFFF" }
];
let currentColor = PALETTE[0].hex;

// One Euro Filter Implementation
class OneEuroFilter {
    constructor(minCutoff = 0.5, beta = 1.5, dCutoff = 1.0) {
        this.minCutoff = minCutoff;
        this.beta = beta;
        this.dCutoff = dCutoff;
        this.reset();
    }
    
    reset() {
        this.xPrev = null;
        this.yPrev = null;
        this.dxPrev = 0;
        this.dyPrev = 0;
        this.tPrev = performance.now() / 1000;
    }

    filter(x, y) {
        const t = performance.now() / 1000;
        if (this.xPrev === null) {
            this.xPrev = x;
            this.yPrev = y;
            this.tPrev = t;
            return { x, y };
        }

        const te = t - this.tPrev;
        if (te <= 0) return { x: this.xPrev, y: this.yPrev };

        const smoothingFactor = (te, cutoff) => {
            const r = 2 * Math.PI * cutoff * te;
            return r / (r + 1);
        };

        const ad = smoothingFactor(te, this.dCutoff);
        const dx = (x - this.xPrev) / te;
        const dy = (y - this.yPrev) / te;
        
        const dxHat = ad * dx + (1 - ad) * this.dxPrev;
        const dyHat = ad * dy + (1 - ad) * this.dyPrev;

        const speed = Math.sqrt(dxHat * dxHat + dyHat * dyHat);
        const cutoff = this.minCutoff + this.beta * speed;

        const a = smoothingFactor(te, cutoff);
        const xHat = a * x + (1 - a) * this.xPrev;
        const yHat = a * y + (1 - a) * this.yPrev;

        this.xPrev = xHat;
        this.yPrev = yHat;
        this.dxPrev = dxHat;
        this.dyPrev = dyHat;
        this.tPrev = t;

        return { x: xHat, y: yHat };
    }
}

const euroFilter = new OneEuroFilter();

// Initialize App
async function initApp() {
    setupToolbar();
    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);
    
    try {
        const vision = await FilesetResolver.forVisionTasks(
            "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
        );
        
        handLandmarker = await HandLandmarker.createFromOptions(vision, {
            baseOptions: {
                modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
                delegate: "GPU"
            },
            runningMode: "VIDEO",
            numHands: 1,
            minHandDetectionConfidence: 0.7,
            minTrackingConfidence: 0.7
        });

        startCamera();
    } catch (e) {
        console.error(e);
        alert("Failed to load AI models.");
    }
}

function resizeCanvas() {
    canvasElement.width = window.innerWidth;
    canvasElement.height = window.innerHeight;
}

function setupToolbar() {
    PALETTE.forEach((color, i) => {
        const div = document.createElement("div");
        div.className = `color-swatch ${i === 0 ? "active" : ""}`;
        div.style.backgroundColor = color.hex;
        div.onclick = () => {
            document.querySelectorAll(".color-swatch").forEach(el => el.classList.remove("active"));
            div.classList.add("active");
            currentColor = color.hex;
        };
        colorsContainer.appendChild(div);
    });

    clearBtn.onclick = () => canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    
    saveBtn.onclick = () => {
        const link = document.createElement('a');
        link.download = 'air-drawing.png';
        // Create composite image with black background (since strokes are bright)
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = canvasElement.width;
        tempCanvas.height = canvasElement.height;
        const tempCtx = tempCanvas.getContext('2d');
        tempCtx.fillStyle = '#0f172a'; // Match CSS bg
        tempCtx.fillRect(0, 0, tempCanvas.width, tempCanvas.height);
        tempCtx.drawImage(canvasElement, 0, 0);
        
        link.href = tempCanvas.toDataURL();
        link.click();
    };

    // Hotkeys
    window.addEventListener('keydown', (e) => {
        if (e.key === 'c') clearBtn.click();
        if (e.key === 's') saveBtn.click();
        if (e.key === '=' || e.key === '+') {
            brushSize = Math.min(40, brushSize + 2);
            sizeVal.innerText = brushSize;
        }
        if (e.key === '-' || e.key === '_') {
            brushSize = Math.max(2, brushSize - 2);
            sizeVal.innerText = brushSize;
        }
    });
}

function startCamera() {
    navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720, facingMode: "user" } }).then((stream) => {
        video.srcObject = stream;
        video.addEventListener("loadeddata", predictWebcam);
        loading.classList.add("hidden");
    }).catch(err => {
        alert("Camera access denied.");
    });
}

function getFingerStates(landmarks) {
    const isUp = (tip, pip) => landmarks[tip].y < landmarks[pip].y;
    return {
        index: isUp(8, 6),
        middle: isUp(12, 10),
        ring: isUp(16, 14),
        pinky: isUp(20, 18)
    };
}

async function predictWebcam() {
    let startTimeMs = performance.now();
    if (lastVideoTime !== video.currentTime) {
        lastVideoTime = video.currentTime;
        const results = handLandmarker.detectForVideo(video, startTimeMs);

        if (results.landmarks && results.landmarks.length > 0) {
            const landmarks = results.landmarks[0];
            
            // Mirror X coordinate since video is mirrored
            const rawX = (1 - landmarks[8].x) * canvasElement.width;
            const rawY = landmarks[8].y * canvasElement.height;
            
            const fingers = getFingerStates(landmarks);
            
            const indexUp = fingers.index;
            const middleUp = fingers.middle;
            const ringUp = fingers.ring;
            const pinkyUp = fingers.pinky;

            const isDraw = indexUp && !middleUp && !ringUp && !pinkyUp;
            const isErase = indexUp && middleUp && ringUp && !pinkyUp;
            const isPause = indexUp && middleUp && !ringUp && !pinkyUp;

            if (isDraw) {
                modeLabel.innerText = "Mode: Drawing 🎨";
                const smoothPoint = euroFilter.filter(rawX, rawY);
                drawStroke(smoothPoint.x, smoothPoint.y, false);
            } else if (isErase) {
                modeLabel.innerText = "Mode: Eraser 🧽";
                const smoothPoint = euroFilter.filter(rawX, rawY);
                drawStroke(smoothPoint.x, smoothPoint.y, true);
            } else if (isPause) {
                modeLabel.innerText = "Mode: Paused (Toolbar/Select) ✌️";
                endStroke();
            } else {
                modeLabel.innerText = "Mode: Idle";
                endStroke();
            }

            drawCursor(rawX, rawY, isDraw, isErase);

        } else {
            endStroke();
            modeLabel.innerText = "Mode: No Hand Detected";
            // Clear cursor frame by drawing a clean frame? 
            // In a web app, the video is separate from canvas. 
            // We should use an overlay canvas for cursor so it doesn't stay painted on drawing canvas.
            // But to keep it simple, we don't draw cursor if no hand.
        }
    }
    
    window.requestAnimationFrame(predictWebcam);
}

// Overlay canvas for cursor
const cursorCanvas = document.createElement("canvas");
cursorCanvas.style.position = "absolute";
cursorCanvas.style.top = "0";
cursorCanvas.style.left = "0";
cursorCanvas.style.width = "100%";
cursorCanvas.style.height = "100%";
cursorCanvas.style.pointerEvents = "none";
cursorCanvas.style.zIndex = "10";
document.querySelector(".canvas-container").appendChild(cursorCanvas);
const cursorCtx = cursorCanvas.getContext("2d");

function drawStroke(x, y, isEraser) {
    if (lastDrawPoint) {
        canvasCtx.beginPath();
        canvasCtx.lineCap = "round";
        canvasCtx.lineJoin = "round";
        if (isEraser) {
            canvasCtx.globalCompositeOperation = "destination-out";
            canvasCtx.lineWidth = brushSize * 8;
            canvasCtx.strokeStyle = "rgba(0,0,0,1)";
        } else {
            canvasCtx.globalCompositeOperation = "source-over";
            canvasCtx.lineWidth = brushSize;
            canvasCtx.strokeStyle = currentColor;
        }
        canvasCtx.moveTo(lastDrawPoint.x, lastDrawPoint.y);
        canvasCtx.lineTo(x, y);
        canvasCtx.stroke();
    }
    lastDrawPoint = { x, y };
}

function endStroke() {
    lastDrawPoint = null;
    euroFilter.reset();
}

function drawCursor(x, y, isDrawing, isEraser) {
    cursorCanvas.width = canvasElement.width;
    cursorCanvas.height = canvasElement.height;
    cursorCtx.clearRect(0, 0, cursorCanvas.width, cursorCanvas.height);
    
    cursorCtx.beginPath();
    if (isDrawing && !isEraser) {
        cursorCtx.arc(x, y, 8, 0, 2 * Math.PI);
        cursorCtx.fillStyle = currentColor;
        cursorCtx.fill();
    } else if (isEraser) {
        cursorCtx.arc(x, y, brushSize * 4, 0, 2 * Math.PI);
        cursorCtx.strokeStyle = "#ff4444";
        cursorCtx.lineWidth = 2;
        cursorCtx.stroke();
    } else {
        cursorCtx.arc(x, y, 10, 0, 2 * Math.PI);
        cursorCtx.strokeStyle = currentColor;
        cursorCtx.lineWidth = 2;
        cursorCtx.stroke();
        
        cursorCtx.beginPath();
        cursorCtx.arc(x, y, 3, 0, 2 * Math.PI);
        cursorCtx.fillStyle = currentColor;
        cursorCtx.fill();
    }
}

// Start
initApp();
