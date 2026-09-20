/**
 * Thai Character Classifier - Interactive Frontend Engine
 * Solution 3 (Pai + Pooh)
 */

document.addEventListener('DOMContentLoaded', () => {
    // --- State Variables ---
    let currentTab = 'draw';
    let selectedModel = 'custom_cnn';
    let isDrawing = false;
    let strokeHistory = [];
    let currentStroke = [];
    let uploadedImageBase64 = null;
    let selectedSampleImageBase64 = null;

    // --- DOM Elements ---
    const canvas = document.getElementById('drawingCanvas');
    const ctx = canvas.getContext('2d');
    const canvasPlaceholder = document.getElementById('canvasPlaceholder');
    const brushSizeInput = document.getElementById('brushSize');
    const brushSizeVal = document.getElementById('brushSizeVal');
    const clearBtn = document.getElementById('clearBtn');
    const undoBtn = document.getElementById('undoBtn');

    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');
    const modelCards = document.querySelectorAll('.model-card');

    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('fileInput');
    const previewContainer = document.getElementById('previewContainer');
    const uploadPreviewImg = document.getElementById('uploadPreviewImg');
    const removeUploadBtn = document.getElementById('removeUploadBtn');

    const samplesGrid = document.getElementById('samplesGrid');
    const refreshSamplesBtn = document.getElementById('refreshSamplesBtn');

    const predictBtn = document.getElementById('predictBtn');
    const resultsEmpty = document.getElementById('resultsEmpty');
    const resultsLoading = document.getElementById('resultsLoading');
    const resultsContent = document.getElementById('resultsContent');
    const latencyTag = document.getElementById('latencyTag');
    const latencyVal = document.getElementById('latencyVal');

    const heroGlyph = document.getElementById('heroGlyph');
    const heroScore = document.getElementById('heroScore');
    const heroThaiName = document.getElementById('heroThaiName');
    const heroEnName = document.getElementById('heroEnName');
    const heroTypeBadge = document.getElementById('heroTypeBadge');
    const heroCodeBadge = document.getElementById('heroCodeBadge');
    const heroMeaning = document.getElementById('heroMeaning');
    const predictionsList = document.getElementById('predictionsList');
    const tensorPreviewImg = document.getElementById('tensorPreviewImg');
    const deviceText = document.getElementById('deviceText');

    // --- Initialize Canvas ---
    function setupCanvas() {
        const dpr = window.devicePixelRatio || 1;
        const rect = canvas.getBoundingClientRect();
        canvas.width = rect.width * dpr;
        canvas.height = rect.height * dpr;
        ctx.scale(dpr, dpr);

        // Fill with white background
        ctx.fillStyle = '#FFFFFF';
        ctx.fillRect(0, 0, rect.width, rect.height);

        ctx.strokeStyle = '#000000';
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        ctx.lineWidth = parseInt(brushSizeInput.value, 10);
    }

    setupCanvas();
    window.addEventListener('resize', () => {
        // Redraw on resize
        const currentData = canvas.toDataURL();
        setupCanvas();
        const img = new Image();
        img.onload = () => ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        img.src = currentData;
    });

    // --- Drawing Handlers ---
    function getCanvasPos(e) {
        const rect = canvas.getBoundingClientRect();
        const clientX = e.touches ? e.touches[0].clientX : e.clientX;
        const clientY = e.touches ? e.touches[0].clientY : e.clientY;
        return {
            x: clientX - rect.left,
            y: clientY - rect.top,
        };
    }

    function startDrawing(e) {
        isDrawing = true;
        canvasPlaceholder.style.opacity = '0';
        const pos = getCanvasPos(e);
        ctx.beginPath();
        ctx.moveTo(pos.x, pos.y);
        currentStroke = [{ x: pos.x, y: pos.y }];
    }

    function draw(e) {
        if (!isDrawing) return;
        e.preventDefault();
        const pos = getCanvasPos(e);
        ctx.lineTo(pos.x, pos.y);
        ctx.stroke();
        currentStroke.push({ x: pos.x, y: pos.y });
    }

    function stopDrawing() {
        if (!isDrawing) return;
        isDrawing = false;
        ctx.closePath();
        if (currentStroke.length > 0) {
            // Save state snapshot for undo
            const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
            strokeHistory.push(imgData);
        }
    }

    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mousemove', draw);
    window.addEventListener('mouseup', stopDrawing);

    canvas.addEventListener('touchstart', startDrawing, { passive: false });
    canvas.addEventListener('touchmove', draw, { passive: false });
    window.addEventListener('touchend', stopDrawing);

    brushSizeInput.addEventListener('input', (e) => {
        ctx.lineWidth = parseInt(e.target.value, 10);
        brushSizeVal.textContent = `${e.target.value}px`;
    });

    clearBtn.addEventListener('click', () => {
        const rect = canvas.getBoundingClientRect();
        ctx.fillStyle = '#FFFFFF';
        ctx.fillRect(0, 0, rect.width, rect.height);
        strokeHistory = [];
        canvasPlaceholder.style.opacity = '0.7';
    });

    undoBtn.addEventListener('click', () => {
        if (strokeHistory.length > 0) {
            strokeHistory.pop();
            const rect = canvas.getBoundingClientRect();
            if (strokeHistory.length > 0) {
                const prev = strokeHistory[strokeHistory.length - 1];
                ctx.putImageData(prev, 0, 0);
            } else {
                ctx.fillStyle = '#FFFFFF';
                ctx.fillRect(0, 0, rect.width, rect.height);
                canvasPlaceholder.style.opacity = '0.7';
            }
        }
    });

    // --- Tab Switcher ---
    tabBtns.forEach((btn) => {
        btn.addEventListener('click', () => {
            const targetTab = btn.getAttribute('data-tab');
            currentTab = targetTab;

            tabBtns.forEach((b) => b.classList.remove('active'));
            tabContents.forEach((c) => c.classList.remove('active'));

            btn.classList.add('active');
            document.getElementById(`tab${targetTab.charAt(0).toUpperCase() + targetTab.slice(1)}`).classList.add('active');

            if (targetTab === 'samples' && samplesGrid.children.length <= 3) {
                loadSamples();
            }
        });
    });

    // --- Model Switcher Cards ---
    modelCards.forEach((card) => {
        card.addEventListener('click', () => {
            modelCards.forEach((c) => c.classList.remove('active'));
            card.classList.add('active');
            selectedModel = card.getAttribute('data-model');
        });
    });

    // --- File Upload & Drag-and-Drop ---
    dropzone.addEventListener('click', (e) => {
        if (e.target !== removeUploadBtn && !removeUploadBtn.contains(e.target)) {
            fileInput.click();
        }
    });

    dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.classList.add('dragover');
    });

    dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));

    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
            handleFileUpload(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileUpload(e.target.files[0]);
        }
    });

    function handleFileUpload(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please select a valid image file (.jpg, .png, etc.).');
            return;
        }

        const reader = new FileReader();
        reader.onload = (e) => {
            uploadedImageBase64 = e.target.result;
            uploadPreviewImg.src = uploadedImageBase64;
            previewContainer.classList.remove('hidden');
        };
        reader.readAsDataURL(file);
    }

    removeUploadBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        uploadedImageBase64 = null;
        uploadPreviewImg.src = '';
        previewContainer.classList.add('hidden');
        fileInput.value = '';
    });

    // --- Load Dataset Random Samples ---
    async function loadSamples() {
        samplesGrid.innerHTML = '<div class="sample-skeleton"></div><div class="sample-skeleton"></div><div class="sample-skeleton"></div>';
        try {
            const res = await fetch('/api/sample_characters');
            const data = await res.json();
            samplesGrid.innerHTML = '';

            data.samples.forEach((sample) => {
                const card = document.createElement('div');
                card.className = 'sample-card';
                card.innerHTML = `
                    <img src="${sample.image_b64}" alt="${sample.character}">
                    <span class="sample-glyph">${sample.character}</span>
                    <span class="sample-desc">${sample.name_th}</span>
                `;
                card.addEventListener('click', () => {
                    selectedSampleImageBase64 = sample.image_b64;
                    document.querySelectorAll('.sample-card').forEach((c) => c.style.borderColor = 'var(--border-glass)');
                    card.style.borderColor = 'var(--primary-indigo)';
                    executePrediction(selectedSampleImageBase64);
                });
                samplesGrid.appendChild(card);
            });
        } catch (err) {
            samplesGrid.innerHTML = '<p style="color:var(--text-dim);grid-column:1/-1;">Could not load samples.</p>';
        }
    }

    refreshSamplesBtn.addEventListener('click', loadSamples);

    // --- Inference Client ---
    predictBtn.addEventListener('click', () => {
        let payloadImage = null;

        if (currentTab === 'draw') {
            payloadImage = canvas.toDataURL('image/png');
        } else if (currentTab === 'upload') {
            if (!uploadedImageBase64) {
                alert('Please upload an image first.');
                return;
            }
            payloadImage = uploadedImageBase64;
        } else if (currentTab === 'samples') {
            if (!selectedSampleImageBase64) {
                alert('Please click on a sample character.');
                return;
            }
            payloadImage = selectedSampleImageBase64;
        }

        executePrediction(payloadImage);
    });

    async function executePrediction(imageBase64) {
        resultsEmpty.classList.add('hidden');
        resultsContent.classList.add('hidden');
        resultsLoading.classList.remove('hidden');

        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    image: imageBase64,
                    model: selectedModel,
                    top_k: 5,
                }),
            });

            const data = await response.json();
            resultsLoading.classList.add('hidden');

            if (data.error) {
                alert(`Prediction Error: ${data.error}`);
                resultsEmpty.classList.remove('hidden');
                return;
            }

            renderResults(data);
        } catch (err) {
            resultsLoading.classList.add('hidden');
            resultsEmpty.classList.remove('hidden');
            alert(`Failed to communicate with server: ${err.message}`);
        }
    }

    function renderResults(data) {
        resultsContent.classList.remove('hidden');

        // Latency
        if (data.latency_ms !== undefined) {
            latencyVal.textContent = data.latency_ms;
            latencyTag.classList.remove('hidden');
        }

        // Hero Card
        const top = data.top_prediction;
        if (top) {
            heroGlyph.textContent = top.character;
            heroScore.textContent = `${top.confidence_pct}%`;
            heroThaiName.textContent = top.name_th;
            heroEnName.textContent = top.name_en;
            heroTypeBadge.textContent = top.type;
            heroCodeBadge.textContent = `TIS-620: ${top.class_number} (0x${top.class_number.toString(16).toUpperCase()})`;
            heroMeaning.textContent = `Meaning / Note: ${top.meaning}`;
        }

        // Top-5 Candidates
        predictionsList.innerHTML = '';
        data.predictions.forEach((pred, index) => {
            const row = document.createElement('div');
            row.className = 'prediction-row';
            row.innerHTML = `
                <span class="prediction-rank">#${index + 1}</span>
                <span class="prediction-char">${pred.character}</span>
                <div class="prediction-bar-container">
                    <span class="prediction-name">${pred.name_th} (${pred.name_en})</span>
                    <div class="prediction-bar-track">
                        <div class="prediction-bar-fill" style="width: 0%;"></div>
                    </div>
                </div>
                <span class="prediction-pct">${pred.confidence_pct}%</span>
            `;
            predictionsList.appendChild(row);

            // Animate bar fill
            setTimeout(() => {
                const fill = row.querySelector('.prediction-bar-fill');
                if (fill) fill.style.width = `${pred.confidence_pct}%`;
            }, 50 * (index + 1));
        });

        // Tensor preview
        if (data.preprocessed_preview) {
            tensorPreviewImg.src = data.preprocessed_preview;
        }
    }

    // --- Poll Hardware Status ---
    async function checkServerStatus() {
        try {
            const res = await fetch('/api/models');
            const info = await res.json();
            deviceText.textContent = `${info.device.toUpperCase()} (${info.device_name || 'Host'})`;
        } catch (e) {
            deviceText.textContent = 'Server Offline';
        }
    }

    checkServerStatus();
});
