(() => {
  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('fileInput');
  const dzIdle = document.getElementById('dzIdle');
  const dzPreview = document.getElementById('dzPreview');
  const previewImg = document.getElementById('previewImg');
  const scanLine = document.getElementById('scanLine');
  const scanLabel = document.getElementById('scanLabel');

  const analyzeBtn = document.getElementById('analyzeBtn');
  const resetBtn = document.getElementById('resetBtn');
  const errorBox = document.getElementById('errorBox');
  const resultBox = document.getElementById('resultBox');

  const resultVerdict = document.getElementById('resultVerdict');
  const meterFill = document.getElementById('meterFill');
  const meterThreshold = document.getElementById('meterThreshold');
  const metricFake = document.getElementById('metricFake');
  const metricReal = document.getElementById('metricReal');
  const metricConfidence = document.getElementById('metricConfidence');

  let currentFile = null;

  function resetUI() {
    currentFile = null;
    fileInput.value = '';
    dzIdle.hidden = false;
    dzPreview.hidden = true;
    previewImg.src = '';
    scanLine.classList.remove('active');
    scanLabel.classList.remove('active');
    analyzeBtn.disabled = true;
    analyzeBtn.textContent = 'Run Scan';
    resetBtn.hidden = true;
    errorBox.hidden = true;
    resultBox.hidden = true;
  }

  function showError(msg) {
    errorBox.textContent = msg;
    errorBox.hidden = false;
    resultBox.hidden = true;
  }

  function setFile(file) {
    if (!file) return;
    if (!file.type.startsWith('image/')) {
      showError('That file does not look like an image. Please choose a JPG, PNG, WEBP, or BMP.');
      return;
    }
    currentFile = file;
    errorBox.hidden = true;
    resultBox.hidden = true;

    const reader = new FileReader();
    reader.onload = (e) => {
      previewImg.src = e.target.result;
      dzIdle.hidden = true;
      dzPreview.hidden = false;
      analyzeBtn.disabled = false;
      resetBtn.hidden = false;
    };
    reader.readAsDataURL(file);
  }

  dropzone.addEventListener('click', () => {
    if (!currentFile) fileInput.click();
  });
  dropzone.setAttribute('tabindex', '0');
  dropzone.setAttribute('role', 'button');
  dropzone.setAttribute('aria-label', 'Upload a face image');
  dropzone.addEventListener('keydown', (e) => {
    if ((e.key === 'Enter' || e.key === ' ') && !currentFile) {
      e.preventDefault();
      fileInput.click();
    }
  });

  fileInput.addEventListener('change', () => setFile(fileInput.files[0]));

  ['dragenter', 'dragover'].forEach(evt =>
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.add('dragover');
    })
  );
  ['dragleave', 'drop'].forEach(evt =>
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.remove('dragover');
    })
  );
  dropzone.addEventListener('drop', (e) => {
    const file = e.dataTransfer.files && e.dataTransfer.files[0];
    setFile(file);
  });

  resetBtn.addEventListener('click', resetUI);

  analyzeBtn.addEventListener('click', async () => {
    if (!currentFile) return;

    analyzeBtn.disabled = true;
    analyzeBtn.textContent = 'Scanning…';
    resetBtn.hidden = true;
    errorBox.hidden = true;
    resultBox.hidden = true;
    scanLine.classList.add('active');
    scanLabel.classList.add('active');

    const formData = new FormData();
    formData.append('image', currentFile);

    try {
      const res = await fetch('/predict', { method: 'POST', body: formData });
      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.error || 'Something went wrong while analyzing the image.');
      }

      renderResult(data);
    } catch (err) {
      showError(err.message || 'Network error — could not reach the scanner.');
    } finally {
      scanLine.classList.remove('active');
      scanLabel.classList.remove('active');
      analyzeBtn.disabled = false;
      analyzeBtn.textContent = 'Run Scan';
      resetBtn.hidden = false;
    }
  });

  function renderResult(data) {
    const isFake = data.label === 'fake';
    const fakePct = Math.round(data.fake_probability * 100);
    const realPct = Math.round(data.real_probability * 100);
    const confPct = Math.round(data.confidence * 100);
    const thresholdPct = Math.round(data.threshold * 100);

    resultVerdict.textContent = isFake ? 'Likely Fake' : 'Likely Real';
    resultVerdict.className = 'result-value ' + (isFake ? 'fake' : 'real');

    meterFill.style.left = `calc(${fakePct}% - 1.5px)`;
    meterThreshold.style.left = `${thresholdPct}%`;

    metricFake.textContent = `${fakePct}%`;
    metricReal.textContent = `${realPct}%`;
    metricConfidence.textContent = `${confPct}%`;

    resultBox.hidden = false;
  }

  resetUI();
})();
