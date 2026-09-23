/**
 * Thai Character Classifier - Interactive Frontend Engine (ver2)
 * Solution 3 (Pai + Pooh)
 */

document.addEventListener('DOMContentLoaded', () => {
    // --- State Variables ---
    let currentTab = 'draw';
    let selectedModel = 'custom_cnn_v1';
    let brushMode = 'smooth'; // 'smooth' or 'pixel'
    let canvasRes = 280;
    let isDrawing = false;
    let strokeHistory = [];
    let uploadedImageBase64 = null;
    let selectedSampleImageBase64 = null;

    // --- DOM Elements ---
    const canvas = document.getElementById('drawingCanvas');
    const ctx = canvas.getContext('2d');
    const canvasContainer = document.getElementById('canvasContainer');
    const gridOverlay = document.getElementById('gridOverlay');
    const canvasPlaceholder = document.getElementById('canvasPlaceholder');
    const brushSizeInput = document.getElementById('brushSize');
    const brushSizeVal = document.getElementById('brushSizeVal');
    const paddingSlider = document.getElementById('paddingSlider');
    const paddingVal = document.getElementById('paddingVal');
    const focalCleanerToggle = document.getElementById('focalCleanerToggle');

    const brushSmoothBtn = document.getElementById('brushSmoothBtn');
    const brushPixelBtn = document.getElementById('brushPixelBtn');
    const canvasResSelect = document.getElementById('canvasResSelect');

    const clearBtn = document.getElementById('clearBtn');
    const undoBtn = document.getElementById('undoBtn');
    const predictBtn = document.getElementById('predictBtn');

    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('fileInput');
    const previewContainer = document.getElementById('previewContainer');
    const uploadPreviewImg = document.getElementById('uploadPreviewImg');
    const removeUploadBtn = document.getElementById('removeUploadBtn');
    const predictUploadBtn = document.getElementById('predictUploadBtn');

    const samplesGrid = document.getElementById('samplesGrid');
    const refreshSamplesBtn = document.getElementById('refreshSamplesBtn');

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
    const cropPreviewImg = document.getElementById('cropPreviewImg');
    const tensorPreviewImg = document.getElementById('tensorPreviewImg');
    const deviceText = document.getElementById('deviceText');

    const modelSelectorChips = document.getElementById('modelSelectorChips');
    const activeModelName = document.getElementById('activeModelName');
    const activeModelBadge = document.getElementById('activeModelBadge');
    const activeModelRes = document.getElementById('activeModelRes');

    // Modal elements
    const checkpointModal = document.getElementById('checkpointModal');
    const openCheckpointModalBtn = document.getElementById('openCheckpointModalBtn');
    const closeCheckpointModalBtn = document.getElementById('closeCheckpointModalBtn');
    const cancelModalBtn = document.getElementById('cancelModalBtn');
    const submitLoadCkptBtn = document.getElementById('submitLoadCkptBtn');
    const customCkptInput = document.getElementById('customCkptInput');
    const customModelNameInput = document.getElementById('customModelNameInput');
    const modalStatusMsg = document.getElementById('modalStatusMsg');
    const presetSol3v2 = document.getElementById('presetSol3v2');
    const presetSol3v1 = document.getElementById('presetSol3v1');
    const presetSol2 = document.getElementById('presetSol2');

    // --- Initialize Canvas ---
    function setupCanvas() {
        canvas.width = canvasRes;
        canvas.height = canvasRes;
        ctx.imageSmoothingEnabled = (brushMode === 'smooth');

        // Fill background with clean white
        ctx.fillStyle = '#FFFFFF';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        ctx.strokeStyle = '#000000';
        ctx.fillStyle = '#000000';
        ctx.lineCap = (brushMode === 'pixel') ? 'square' : 'round';
        ctx.lineJoin = (brushMode === 'pixel') ? 'miter' : 'round';
        ctx.lineWidth = parseInt(brushSizeInput.value, 10);

        // Show grid overlay if low resolution (32x32 or 64x64)
        if (canvasRes <= 64) {
            gridOverlay.style.backgroundImage = `
                linear-gradient(to right, rgba(255,255,255,0.08) 1px, transparent 1px),
                linear-gradient(to bottom, rgba(255,255,255,0.08) 1px, transparent 1px)
            `;
            gridOverlay.style.backgroundSize = `${100 / canvasRes}% ${100 / canvasRes}%`;
            gridOverlay.style.display = 'block';
        } else {
            gridOverlay.style.display = 'none';
        }
    }

    setupCanvas();

    // --- Coordinate Calculation ---
    function getCanvasPos(e) {
        const rect = canvas.getBoundingClientRect();
        const clientX = e.touches ? e.touches[0].clientX : e.clientX;
        const clientY = e.touches ? e.touches[0].clientY : e.clientY;
        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;

        let x = (clientX - rect.left) * scaleX;
        let y = (clientY - rect.top) * scaleY;

        if (brushMode === 'pixel') {
            x = Math.floor(x);
            y = Math.floor(y);
        }

        return { x, y };
    }

    // --- Drawing Handlers ---
    function startDrawing(e) {
        isDrawing = true;
        canvasPlaceholder.style.opacity = '0';
        const pos = getCanvasPos(e);

        if (brushMode === 'pixel') {
            drawPixelBlock(pos.x, pos.y);
        } else {
            ctx.beginPath();
            ctx.moveTo(pos.x, pos.y);
        }
    }

    function drawPixelBlock(x, y) {
        const size = Math.max(1, Math.round(parseInt(brushSizeInput.value, 10) * (canvasRes / 280)));
        ctx.fillStyle = '#000000';
        ctx.fillRect(x - Math.floor(size / 2), y - Math.floor(size / 2), size, size);
    }

    function draw(e) {
        if (!isDrawing) return;
        e.preventDefault();
        const pos = getCanvasPos(e);

        if (brushMode === 'pixel') {
            drawPixelBlock(pos.x, pos.y);
        } else {
            ctx.lineTo(pos.x, pos.y);
            ctx.stroke();
        }
    }

    function stopDrawing() {
        if (!isDrawing) return;
        isDrawing = false;
        if (brushMode === 'smooth') {
            ctx.closePath();
        }
        // Save history snapshot for undo
        strokeHistory.push(ctx.getImageData(0, 0, canvas.width, canvas.height));
    }

    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mousemove', draw);
    window.addEventListener('mouseup', stopDrawing);

    canvas.addEventListener('touchstart', startDrawing, { passive: false });
    canvas.addEventListener('touchmove', draw, { passive: false });
    window.addEventListener('touchend', stopDrawing);

    // --- Controls Listeners ---
    brushSizeInput.addEventListener('input', (e) => {
        ctx.lineWidth = parseInt(e.target.value, 10);
        brushSizeVal.textContent = `${e.target.value}px`;
    });

    paddingSlider.addEventListener('input', (e) => {
        paddingVal.textContent = `${e.target.value}%`;
    });

    brushSmoothBtn.addEventListener('click', () => {
        brushMode = 'smooth';
        brushSmoothBtn.classList.add('active');
        brushPixelBtn.classList.remove('active');
        ctx.imageSmoothingEnabled = true;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
    });

    brushPixelBtn.addEventListener('click', () => {
        brushMode = 'pixel';
        brushPixelBtn.classList.add('active');
        brushSmoothBtn.classList.remove('active');
        ctx.imageSmoothingEnabled = false;
        ctx.lineCap = 'square';
        ctx.lineJoin = 'miter';
    });

    canvasResSelect.addEventListener('change', (e) => {
        canvasRes = parseInt(e.target.value, 10);
        setupCanvas();
        strokeHistory = [];
        canvasPlaceholder.style.opacity = '0.7';
    });

    clearBtn.addEventListener('click', () => {
        ctx.fillStyle = '#FFFFFF';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        strokeHistory = [];
        canvasPlaceholder.style.opacity = '0.7';
    });

    undoBtn.addEventListener('click', () => {
        if (strokeHistory.length > 0) {
            strokeHistory.pop();
            if (strokeHistory.length > 0) {
                const prev = strokeHistory[strokeHistory.length - 1];
                ctx.putImageData(prev, 0, 0);
            } else {
                ctx.fillStyle = '#FFFFFF';
                ctx.fillRect(0, 0, canvas.width, canvas.height);
                canvasPlaceholder.style.opacity = '0.7';
            }
        }
    });

    // --- Tabs Switching ---
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            tabBtns.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));
            btn.classList.add('active');
            currentTab = btn.getAttribute('data-tab');
            document.getElementById(`tab-${currentTab}`).classList.add('active');
        });
    });

    // --- Upload Dropzone Handlers ---
    dropzone.addEventListener('click', () => fileInput.click());
    dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.classList.add('drag-over');
    });
    dropzone.addEventListener('dragleave', () => dropzone.classList.remove('drag-over'));
    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('drag-over');
        if (e.dataTransfer.files.length > 0) {
            handleUploadedFile(e.dataTransfer.files[0]);
        }
    });
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleUploadedFile(e.target.files[0]);
        }
    });

    function handleUploadedFile(file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            uploadedImageBase64 = e.target.result;
            uploadPreviewImg.src = uploadedImageBase64;
            dropzone.style.display = 'none';
            previewContainer.style.display = 'block';
        };
        reader.readAsDataURL(file);
    }

    removeUploadBtn.addEventListener('click', () => {
        uploadedImageBase64 = null;
        fileInput.value = '';
        dropzone.style.display = 'flex';
        previewContainer.style.display = 'none';
    });

    // --- Fetch Models & Catalog ---
    async function loadModelsCatalog() {
        try {
            const res = await fetch('/api/models');
            const data = await res.json();

            if (data.device) {
                deviceText.textContent = `${data.device.toUpperCase()} (${data.device_name || 'Active'})`;
            }

            modelSelectorChips.innerHTML = '';
            data.models.forEach(m => {
                const chip = document.createElement('button');
                chip.className = `model-chip ${m.id === selectedModel ? 'active' : ''} ${m.trained ? '' : 'disabled'}`;
                chip.innerHTML = `
                    <span class="chip-name">${m.name}</span>
                    <span class="chip-badge">${m.badge}</span>
                `;
                chip.addEventListener('click', () => {
                    selectedModel = m.id;
                    activeModelName.textContent = m.name;
                    activeModelBadge.textContent = m.badge;
                    activeModelRes.textContent = `Res: ${m.resolution || '32x32'}`;
                    document.querySelectorAll('.model-chip').forEach(c => c.classList.remove('active'));
                    chip.classList.add('active');
                });
                modelSelectorChips.appendChild(chip);

                if (m.id === selectedModel) {
                    activeModelName.textContent = m.name;
                    activeModelBadge.textContent = m.badge;
                    activeModelRes.textContent = `Res: ${m.resolution || '32x32'}`;
                }
            });
        } catch (err) {
            console.error('Failed to load models catalog:', err);
        }
    }

    // --- Fetch Cleaned Dataset Samples ---
    async function loadDatasetSamples() {
        samplesGrid.innerHTML = '<div class="spinner"></div>';
        try {
            const res = await fetch('/api/sample_characters');
            const data = await res.json();
            samplesGrid.innerHTML = '';

            data.samples.forEach(s => {
                const item = document.createElement('div');
                item.className = 'sample-card';
                item.innerHTML = `
                    <div class="sample-img-box">
                        <img src="${s.image_b64}" alt="${s.character}">
                    </div>
                    <div class="sample-info">
                        <span class="sample-char">${s.character}</span>
                        <span class="sample-code">Class ${s.class_number}</span>
                    </div>
                `;
                item.addEventListener('click', () => {
                    selectedSampleImageBase64 = s.image_b64;
                    document.querySelectorAll('.sample-card').forEach(c => c.classList.remove('active'));
                    item.classList.add('active');
                    runInference(s.image_b64);
                });
                samplesGrid.appendChild(item);
            });
        } catch (err) {
            samplesGrid.innerHTML = `<p class="error-msg">Failed to load samples: ${err.message}</p>`;
        }
    }

    refreshSamplesBtn.addEventListener('click', loadDatasetSamples);

    // --- Inference Request Execution ---
    async function runInference(imageData) {
        resultsEmpty.style.display = 'none';
        resultsContent.style.display = 'none';
        resultsLoading.style.display = 'flex';

        const padRatio = parseFloat(paddingSlider.value) / 100.0;
        const useFocal = focalCleanerToggle.checked;

        try {
            const res = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    image: imageData,
                    model: selectedModel,
                    top_k: 5,
                    pad_ratio: padRatio,
                    use_focal_cleaner: useFocal,
                }),
            });

            const data = await res.json();
            resultsLoading.style.display = 'none';

            if (data.error) {
                alert(`Inference Error: ${data.error}`);
                resultsEmpty.style.display = 'flex';
                return;
            }

            renderResults(data);
        } catch (err) {
            resultsLoading.style.display = 'none';
            resultsEmpty.style.display = 'flex';
            alert(`Network error: ${err.message}`);
        }
    }

    function renderResults(data) {
        resultsContent.style.display = 'block';

        if (data.latency_ms !== undefined) {
            latencyTag.style.display = 'inline-flex';
            latencyVal.textContent = `${data.latency_ms} ms`;
        }

        const top = data.top_prediction;
        if (top) {
            heroGlyph.textContent = top.character;
            heroScore.textContent = `${top.confidence_pct}%`;
            heroThaiName.textContent = top.name_th;
            heroEnName.textContent = top.name_en;
            heroTypeBadge.textContent = top.type;
            heroCodeBadge.textContent = `TIS-620: ${top.class_number} (0x${top.class_number.toString(16).toUpperCase()})`;
            heroMeaning.textContent = `Meaning: ${top.meaning}`;
        }

        if (data.crop_preview) {
            cropPreviewImg.src = data.crop_preview;
        }
        if (data.preprocessed_preview) {
            tensorPreviewImg.src = data.preprocessed_preview;
        }

        predictionsList.innerHTML = '';
        data.predictions.forEach((p, idx) => {
            const row = document.createElement('div');
            row.className = `prediction-row ${idx === 0 ? 'top-rank' : ''}`;
            row.innerHTML = `
                <div class="row-rank">#${idx + 1}</div>
                <div class="row-glyph">${p.character}</div>
                <div class="row-names">
                    <span class="row-thai">${p.name_th}</span>
                    <span class="row-en">${p.name_en} • Class ${p.class_number}</span>
                </div>
                <div class="row-bar-wrap">
                    <div class="row-bar-fill" style="width: ${p.confidence_pct}%"></div>
                </div>
                <div class="row-pct">${p.confidence_pct}%</div>
            `;
            predictionsList.appendChild(row);
        });
    }

    predictBtn.addEventListener('click', () => {
        const dataUrl = canvas.toDataURL('image/png');
        runInference(dataUrl);
    });

    predictUploadBtn.addEventListener('click', () => {
        if (uploadedImageBase64) {
            runInference(uploadedImageBase64);
        }
    });

    // --- Custom Checkpoint Modal Logic ---
    openCheckpointModalBtn.addEventListener('click', () => {
        modalStatusMsg.style.display = 'none';
        checkpointModal.style.display = 'flex';
    });

    function closeModal() {
        checkpointModal.style.display = 'none';
    }

    closeCheckpointModalBtn.addEventListener('click', closeModal);
    cancelModalBtn.addEventListener('click', closeModal);

    presetSol3v2.addEventListener('click', () => {
        customCkptInput.value = 'c:\\workspace\\vscode\\kmitl\\3-1\\dlmed\\DL-in-Medical-Image-Project-1\\Solution 3 (Pai + Pooh)\\ver2\\checkpoints\\custom_cnn\\best_model.pt';
        customModelNameInput.value = 'CustomGlyphCNN-v2';
    });

    presetSol3v1.addEventListener('click', () => {
        customCkptInput.value = 'c:\\workspace\\vscode\\kmitl\\3-1\\dlmed\\DL-in-Medical-Image-Project-1\\Solution 3 (Pai + Pooh)\\ver1\\checkpoints\\custom_cnn\\best_model.pt';
        customModelNameInput.value = 'CustomGlyphCNN-v1';
    });

    presetSol2.addEventListener('click', () => {
        customCkptInput.value = 'c:\\workspace\\vscode\\kmitl\\3-1\\dlmed\\DL-in-Medical-Image-Project-1\\Solution 2 (Eungul + Ninenine)\\best_thai_character_model.pth';
        customModelNameInput.value = 'Solution2-EfficientNet';
    });

    submitLoadCkptBtn.addEventListener('click', async () => {
        const ckptPath = customCkptInput.value.trim();
        const alias = customModelNameInput.value.trim();

        if (!ckptPath) {
            modalStatusMsg.className = 'modal-status error';
            modalStatusMsg.textContent = 'Please specify a valid checkpoint path.';
            modalStatusMsg.style.display = 'block';
            return;
        }

        submitLoadCkptBtn.disabled = true;
        submitLoadCkptBtn.textContent = 'Inspecting & Loading...';

        try {
            const res = await fetch('/api/load_custom_checkpoint', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ checkpoint_path: ckptPath, model_name: alias }),
            });
            const data = await res.json();

            if (data.success) {
                modalStatusMsg.className = 'modal-status success';
                modalStatusMsg.textContent = data.message;
                modalStatusMsg.style.display = 'block';
                selectedModel = data.model_id;
                await loadModelsCatalog();
                setTimeout(closeModal, 1200);
            } else {
                modalStatusMsg.className = 'modal-status error';
                modalStatusMsg.textContent = data.error || 'Failed to load checkpoint.';
                modalStatusMsg.style.display = 'block';
            }
        } catch (err) {
            modalStatusMsg.className = 'modal-status error';
            modalStatusMsg.textContent = `Error: ${err.message}`;
            modalStatusMsg.style.display = 'block';
        } finally {
            submitLoadCkptBtn.disabled = false;
            submitLoadCkptBtn.textContent = 'Load & Activate Checkpoint';
        }
    });

    // --- Initial Boot ---
    loadModelsCatalog();
    loadDatasetSamples();
});
