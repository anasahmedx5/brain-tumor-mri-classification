// NeuroScan AI - Brain Tumor MRI Classification Frontend Logic

document.addEventListener('DOMContentLoaded', () => {
  // Elements
  const apiStatusChip = document.getElementById('apiStatusChip');
  const apiStatusText = document.getElementById('apiStatusText');
  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('fileInput');
  const btnBrowse = document.getElementById('btnBrowse');
  const dropzonePrompt = document.getElementById('dropzonePrompt');
  const previewViewport = document.getElementById('previewViewport');
  const mriPreviewImg = document.getElementById('mriPreviewImg');
  const previewFileName = document.getElementById('previewFileName');
  const previewDimensions = document.getElementById('previewDimensions');
  const previewFileSize = document.getElementById('previewFileSize');
  const btnAnalyze = document.getElementById('btnAnalyze');
  const btnClear = document.getElementById('btnClear');
  const btnLoader = document.getElementById('btnLoader');
  const autoAnalyzeToggle = document.getElementById('autoAnalyzeToggle');
  const samplesGrid = document.getElementById('samplesGrid');

  // Results elements
  const resultsPlaceholder = document.getElementById('resultsPlaceholder');
  const resultsLoading = document.getElementById('resultsLoading');
  const resultsContent = document.getElementById('resultsContent');
  const liveLatencyPill = document.getElementById('liveLatencyPill');
  const headerLatencyText = document.getElementById('headerLatencyText');
  const elapsedTimer = document.getElementById('elapsedTimer');

  // Prediction card
  const predictedLabel = document.getElementById('predictedLabel');
  const clinicalStatusTag = document.getElementById('clinicalStatusTag');
  const confidencePercent = document.getElementById('confidencePercent');
  const confidenceRaw = document.getElementById('confidenceRaw');
  const gaugeBar = document.getElementById('gaugeBar');
  const latencyVal = document.getElementById('latencyVal');
  const latencySpeedBadge = document.getElementById('latencySpeedBadge');
  const frameworkVal = document.getElementById('frameworkVal');

  // JSON Accordion
  const btnToggleJson = document.getElementById('btnToggleJson');
  const jsonAccordionContent = document.getElementById('jsonAccordionContent');
  const accordionArrow = document.getElementById('accordionArrow');
  const rawJsonBlock = document.getElementById('rawJsonBlock');
  const btnCopyJson = document.getElementById('btnCopyJson');

  // Error Banner
  const errorBanner = document.getElementById('errorBanner');
  const errorTitle = document.getElementById('errorTitle');
  const errorMessage = document.getElementById('errorMessage');
  const btnDismissError = document.getElementById('btnDismissError');

  // State
  let currentFile = null;
  let timerInterval = null;
  const classesList = ['glioma', 'meningioma', 'notumor', 'pituitary'];

  // Base API URL: default to current origin (works when served by FastAPI or configured host)
  const API_BASE = window.location.origin.includes('http') ? window.location.origin : 'http://127.0.0.1:8000';

  // 1. Check API Health
  async function checkHealth() {
    try {
      const res = await fetch(`${API_BASE}/health`);
      if (res.ok) {
        const data = await res.json();
        apiStatusChip.classList.remove('error');
        apiStatusText.textContent = `FastAPI Online (${data.framework.toUpperCase()})`;
      } else {
        throw new Error(`Status ${res.status}`);
      }
    } catch (err) {
      apiStatusChip.classList.add('error');
      apiStatusText.textContent = 'FastAPI Offline';
      console.warn('API Health check failed:', err);
    }
  }
  checkHealth();

  // 2. File Selection & Drag-and-Drop
  btnBrowse.addEventListener('click', (e) => {
    e.stopPropagation();
    fileInput.click();
  });

  dropzone.addEventListener('click', (e) => {
    if (previewViewport.classList.contains('hidden')) {
      fileInput.click();
    }
  });

  fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFileSelected(e.target.files[0]);
    }
  });

  ['dragenter', 'dragover'].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.add('dragover');
    });
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.remove('dragover');
    });
  });

  dropzone.addEventListener('drop', (e) => {
    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      handleFileSelected(files[0]);
    }
  });

  function handleFileSelected(file) {
    if (!file.type.startsWith('image/')) {
      showError('Invalid File Type', 'Please upload a valid image file (JPEG or PNG).');
      return;
    }

    currentFile = file;
    hideError();

    // Clear active sample highlights
    document.querySelectorAll('.sample-card').forEach(c => c.classList.remove('active'));

    // Read and display preview
    const reader = new FileReader();
    reader.onload = (e) => {
      mriPreviewImg.src = e.target.result;
      
      mriPreviewImg.onload = () => {
        previewDimensions.textContent = `${mriPreviewImg.naturalWidth} \u00d7 ${mriPreviewImg.naturalHeight} px`;
      };

      previewFileName.textContent = file.name;
      previewFileSize.textContent = `${(file.size / 1024).toFixed(1)} KB`;

      dropzonePrompt.classList.add('hidden');
      previewViewport.classList.remove('hidden');

      btnAnalyze.disabled = false;
      btnClear.disabled = false;

      // Auto-analyze if enabled
      if (autoAnalyzeToggle.checked) {
        runInference();
      }
    };
    reader.readAsDataURL(file);
  }

  // 3. Quick-Test Samples
  samplesGrid.addEventListener('click', async (e) => {
    const card = e.target.closest('.sample-card');
    if (!card) return;

    const filename = card.getAttribute('data-filename');
    document.querySelectorAll('.sample-card').forEach(c => c.classList.remove('active'));
    card.classList.add('active');

    try {
      // Fetch sample image as Blob
      const imgUrl = `/data/${filename}`;
      const res = await fetch(imgUrl);
      if (!res.ok) throw new Error(`Could not load sample ${filename}`);

      const blob = await res.blob();
      const file = new File([blob], filename, { type: 'image/jpeg' });
      handleFileSelected(file);
    } catch (err) {
      showError('Sample Load Error', `Failed to load sample image "${filename}": ${err.message}`);
    }
  });

  // 4. Run Inference API Call
  btnAnalyze.addEventListener('click', () => {
    runInference();
  });

  async function runInference() {
    if (!currentFile) return;

    // UI Loading state
    hideError();
    resultsPlaceholder.classList.add('hidden');
    resultsContent.classList.add('hidden');
    resultsLoading.classList.remove('hidden');
    btnAnalyze.disabled = true;
    btnLoader.classList.remove('hidden');

    let startTime = performance.now();
    elapsedTimer.textContent = '0.0s elapsed';
    clearInterval(timerInterval);
    timerInterval = setInterval(() => {
      const elapsed = ((performance.now() - startTime) / 1000).toFixed(1);
      elapsedTimer.textContent = `${elapsed}s elapsed`;
    }, 100);

    const formData = new FormData();
    formData.append('file', currentFile);

    try {
      const response = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        body: formData
      });

      clearInterval(timerInterval);

      if (!response.ok) {
        let errDetail = 'Inference failed';
        try {
          const errJson = await response.json();
          errDetail = errJson.detail || errDetail;
        } catch (_) {}
        throw new Error(errDetail);
      }

      const data = await response.json();
      displayResults(data);
    } catch (err) {
      clearInterval(timerInterval);
      resultsLoading.classList.add('hidden');
      resultsPlaceholder.classList.remove('hidden');
      showError('Prediction Failed', err.message || 'Error communicating with FastAPI server.');
    } finally {
      btnAnalyze.disabled = false;
      btnLoader.classList.add('hidden');
    }
  }

  // 5. Display Prediction Outputs matching project contract
  function displayResults(data) {
    resultsLoading.classList.add('hidden');
    resultsPlaceholder.classList.add('hidden');
    resultsContent.classList.remove('hidden');

    // Expected JSON structure:
    // {
    //   "framework": "pytorch",
    //   "label": "pituitary",
    //   "confidence": 0.99,
    //   "probabilities": { "glioma": 0.0, "meningioma": 0.01, "notumor": 0.0, "pituitary": 0.99 },
    //   "latency_ms": 38
    // }

    const label = data.label || 'Unknown';
    const confidence = typeof data.confidence === 'number' ? data.confidence : 0;
    const probabilities = data.probabilities || {};
    const latencyMs = data.latency_ms !== undefined ? data.latency_ms : 0;
    const framework = data.framework || 'PyTorch';

    // 1. Predicted Label & Tag
    predictedLabel.textContent = formatClassName(label);
    updateClinicalTag(label);

    // 2. Confidence Circle & Text
    const confPct = Math.round(confidence * 100);
    confidencePercent.textContent = `${confPct}%`;
    confidenceRaw.textContent = confidence.toFixed(2);

    // Circumference = 2 * PI * 42 ≈ 263.89
    const circumference = 264;
    const offset = circumference - (confidence * circumference);
    gaugeBar.style.strokeDashoffset = offset;

    // 3. Probabilities Distribution Bars
    classesList.forEach(cls => {
      const row = document.getElementById(`row-${cls}`);
      const fill = document.getElementById(`fill-${cls}`);
      const scorePct = document.getElementById(`score-pct-${cls}`);
      const scoreRaw = document.getElementById(`score-raw-${cls}`);

      const probVal = probabilities[cls] !== undefined ? probabilities[cls] : 0;
      const probPercent = (probVal * 100).toFixed(1);

      if (scorePct) scorePct.textContent = `${probPercent}%`;
      if (scoreRaw) scoreRaw.textContent = `(${probVal.toFixed(2)})`;
      if (fill) fill.style.width = `${Math.min(100, Math.max(0, probPercent))}%`;

      if (row) {
        if (cls.toLowerCase() === label.toLowerCase()) {
          row.classList.add('winner');
        } else {
          row.classList.remove('winner');
        }
      }
    });

    // 4. Latency & Performance
    latencyVal.textContent = `${latencyMs} ms`;
    headerLatencyText.textContent = `${latencyMs} ms`;
    liveLatencyPill.classList.remove('hidden');

    if (latencyMs < 50) {
      latencySpeedBadge.textContent = 'Ultra Fast';
      latencySpeedBadge.style.background = 'rgba(16, 185, 129, 0.15)';
      latencySpeedBadge.style.color = '#34d399';
    } else if (latencyMs < 120) {
      latencySpeedBadge.textContent = 'Standard';
      latencySpeedBadge.style.background = 'rgba(6, 182, 212, 0.15)';
      latencySpeedBadge.style.color = '#22d3ee';
    } else {
      latencySpeedBadge.textContent = 'Heavy';
      latencySpeedBadge.style.background = 'rgba(245, 158, 11, 0.15)';
      latencySpeedBadge.style.color = '#fbbf24';
    }

    frameworkVal.textContent = framework.toUpperCase();

    // 5. Raw JSON Inspector
    rawJsonBlock.textContent = JSON.stringify(data, null, 2);
  }

  function formatClassName(name) {
    const map = {
      'glioma': 'Glioma Tumor',
      'meningioma': 'Meningioma Tumor',
      'notumor': 'No Tumor (Healthy)',
      'pituitary': 'Pituitary Tumor'
    };
    return map[name.toLowerCase()] || name.toUpperCase();
  }

  function updateClinicalTag(name) {
    clinicalStatusTag.className = 'clinical-tag';
    if (name.toLowerCase() === 'notumor') {
      clinicalStatusTag.textContent = 'Benign / Negative';
      clinicalStatusTag.style.background = 'rgba(16, 185, 129, 0.2)';
      clinicalStatusTag.style.color = '#34d399';
    } else {
      clinicalStatusTag.textContent = 'Positive Finding';
      clinicalStatusTag.style.background = 'rgba(244, 63, 94, 0.2)';
      clinicalStatusTag.style.color = '#fb7185';
    }
  }

  // 6. Clear Button
  btnClear.addEventListener('click', () => {
    currentFile = null;
    fileInput.value = '';
    mriPreviewImg.src = '';
    dropzonePrompt.classList.remove('hidden');
    previewViewport.classList.add('hidden');
    btnAnalyze.disabled = true;
    btnClear.disabled = true;

    resultsContent.classList.add('hidden');
    resultsLoading.classList.add('hidden');
    resultsPlaceholder.classList.remove('hidden');
    liveLatencyPill.classList.add('hidden');
    hideError();

    document.querySelectorAll('.sample-card').forEach(c => c.classList.remove('active'));
  });

  // 7. Accordion Toggle
  btnToggleJson.addEventListener('click', () => {
    const isHidden = jsonAccordionContent.classList.contains('hidden');
    if (isHidden) {
      jsonAccordionContent.classList.remove('hidden');
      accordionArrow.classList.add('rotated');
    } else {
      jsonAccordionContent.classList.add('hidden');
      accordionArrow.classList.remove('rotated');
    }
  });

  // 8. Copy JSON
  btnCopyJson.addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(rawJsonBlock.textContent);
      const originalText = btnCopyJson.textContent;
      btnCopyJson.textContent = 'Copied!';
      setTimeout(() => {
        btnCopyJson.textContent = originalText;
      }, 1800);
    } catch (err) {
      console.error('Failed to copy JSON:', err);
    }
  });

  // 9. Error handling
  function showError(title, msg) {
    errorTitle.textContent = title;
    errorMessage.textContent = msg;
    errorBanner.classList.remove('hidden');
  }

  function hideError() {
    errorBanner.classList.add('hidden');
  }

  btnDismissError.addEventListener('click', hideError);
});
