/* ==========================================================================
   PRODUCT VISION AI - SINGLE-UNIT SAMPLE REGISTRATION STATION (CAMERA)
   High-performance webcam stream, optical viewfinder focus, and Gemini Vision AI
   (Local AI Fallback + Auto-Scan on Hold + No noisy background boxes & No live counting)
   ========================================================================== */

// DOM Elements
const startCameraBtn = document.getElementById('startCameraBtn');
const stopCameraBtn = document.getElementById('stopCameraBtn');
const snapshotBtn = document.getElementById('snapshotBtn');
const identifySampleBtn = document.getElementById('identifySampleBtn');
const autoScanToggle = document.getElementById('autoScanToggle');

const video = document.getElementById('cameraFeed');
const canvas = document.getElementById('cameraCanvas');
const placeholder = document.getElementById('cameraPlaceholder');
const cameraHud = document.getElementById('cameraHud');
const cameraViewfinder = document.getElementById('cameraViewfinder');
const viewfinderCountdown = document.getElementById('viewfinderCountdown');
const viewfinderInstruction = document.getElementById('viewfinderInstruction');
const streamStatusText = document.getElementById('streamStatusText');
const liveDotIndicator = document.getElementById('liveDotIndicator');
const fpsCounter = document.getElementById('fpsCounter');
const hudTimerBadge = document.getElementById('hudTimerBadge');

// Sidebar and Modal Mounts
const activeSampleWrapper = document.getElementById('activeSampleWrapper');
const cameraRecentSamplesList = document.getElementById('cameraRecentSamplesList');
const openLibraryBtn = document.getElementById('openLibraryBtn');
const sampleLibraryModal = document.getElementById('sampleLibraryModal');
const closeLibraryModalBtn = document.getElementById('closeLibraryModalBtn');
const sampleLibraryList = document.getElementById('sampleLibraryList');

// Gemini Key Modal Elements
const btnOpenGeminiModal = document.getElementById('btnOpenGeminiModal');
const geminiKeyStatusText = document.getElementById('geminiKeyStatusText');
const geminiKeyModal = document.getElementById('geminiKeyModal');
const closeGeminiModalBtn = document.getElementById('closeGeminiModalBtn');
const btnCancelGeminiModal = document.getElementById('btnCancelGeminiModal');
const btnSaveGeminiKey = document.getElementById('btnSaveGeminiKey');
const geminiApiKeyInput = document.getElementById('geminiApiKeyInput');
const geminiKeyModalMsg = document.getElementById('geminiKeyModalMsg');
const btnToggleKeyEye = document.getElementById('btnToggleKeyEye');

let stream = null;
let animFrameId = null;
let sessionSeconds = 0;
let sessionTimer = null;
let lastFpsTime = performance.now();
let framesThisSecond = 0;

let isIdentifying = false;
let autoScanInterval = null;
let holdSteadyScore = 0;
let lastSamplePixels = null;

// Toast Notification System
function showToast(message, type = 'info') {
  let container = document.getElementById('toastContainer');
  if (!container) {
    container = document.createElement('div');
    container.id = 'toastContainer';
    container.className = 'toast-container';
    document.body.appendChild(container);
  }

  const toast = document.createElement('div');
  toast.className = `toast toast-${type}`;
  toast.innerHTML = `
    <div style="flex: 1; font-size: 0.9rem;">${message}</div>
    <span style="cursor: pointer; opacity: 0.7; font-weight: bold; margin-left: 8px;">✕</span>
  `;

  toast.querySelector('span').addEventListener('click', () => toast.remove());
  container.appendChild(toast);

  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transform = 'translateX(100%)';
    toast.style.transition = 'all 0.3s ease';
    setTimeout(() => toast.remove(), 300);
  }, 4000);
}

// Format duration HH:MM:SS
function formatTime(seconds) {
  const h = String(Math.floor(seconds / 3600)).padStart(2, '0');
  const m = String(Math.floor((seconds % 3600) / 60)).padStart(2, '0');
  const s = String(seconds % 60).padStart(2, '0');
  return `${h}:${m}:${s}`;
}

// ==========================================================================
// Gemini API Key Management
// ==========================================================================
async function checkGeminiStatus() {
  try {
    const res = await fetch('/api/vision/status');
    if (!res.ok) return;
    const data = await res.json();
    const isConfigured = data?.status?.gemini?.configured;

    if (geminiKeyStatusText && btnOpenGeminiModal) {
      if (isConfigured) {
        geminiKeyStatusText.textContent = 'Gemini: Đã kết nối';
      } else {
        geminiKeyStatusText.textContent = 'Cấu hình Gemini';
      }
    }
  } catch (e) {
    console.debug('Gemini status check skipped:', e);
  }
}

function openGeminiModal() {
  if (!geminiKeyModal) return;
  geminiKeyModal.style.display = 'flex';
  if (geminiKeyModalMsg) geminiKeyModalMsg.style.display = 'none';
  const saved = localStorage.getItem('gemini_api_key') || '';
  if (geminiApiKeyInput) geminiApiKeyInput.value = saved;
}

function closeGeminiModal() {
  if (geminiKeyModal) geminiKeyModal.style.display = 'none';
}

async function saveGeminiKeyAction() {
  const key = (geminiApiKeyInput?.value || '').trim();
  if (!key) {
    showToast('Vui lòng dán Gemini API Key.', 'warning');
    return;
  }

  btnSaveGeminiKey.disabled = true;
  btnSaveGeminiKey.textContent = 'Đang kiểm tra...';

  try {
    const res = await fetch('/api/vision/gemini/save-key', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ api_key: key, persist_to_env: true }),
    });
    const data = await res.json();

    if (res.ok && data.success) {
      localStorage.setItem('gemini_api_key', key);
      showToast('Đã lưu và kích hoạt Google Gemini API Key thành công!', 'success');
      closeGeminiModal();
      checkGeminiStatus();
    } else {
      if (geminiKeyModalMsg) {
        geminiKeyModalMsg.className = 'alert-box alert-error';
        geminiKeyModalMsg.textContent = data.detail || data.message || 'API Key không hợp lệ.';
        geminiKeyModalMsg.style.display = 'block';
      }
    }
  } catch (err) {
    if (geminiKeyModalMsg) {
      geminiKeyModalMsg.className = 'alert-box alert-error';
      geminiKeyModalMsg.textContent = 'Không thể kết nối máy chủ để lưu key: ' + err.message;
      geminiKeyModalMsg.style.display = 'block';
    }
  } finally {
    btnSaveGeminiKey.disabled = false;
    btnSaveGeminiKey.textContent = 'Lưu & Kích Hoạt Key';
  }
}

function toggleKeyVisibility() {
  if (!geminiApiKeyInput) return;
  geminiApiKeyInput.type = geminiApiKeyInput.type === 'password' ? 'text' : 'password';
}

// ==========================================================================
// Smooth Real-Time Camera Render Loop (Clean, 60fps, No noisy YOLO boxes)
// ==========================================================================
function processCameraFrames() {
  if (!stream || video.paused || video.ended) return;

  if (canvas && video.videoWidth > 0) {
    if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
    }
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Calculate real FPS
    framesThisSecond++;
    const now = performance.now();
    if (now - lastFpsTime >= 1000) {
      if (fpsCounter) {
        fpsCounter.textContent = Math.round((framesThisSecond * 1000) / (now - lastFpsTime));
      }
      framesThisSecond = 0;
      lastFpsTime = now;
    }
  }

  animFrameId = requestAnimationFrame(processCameraFrames);
}

// ==========================================================================
// Auto-Scan on Hold (Tự Động Chụp & Quét Khi Người Dùng Giữ Yên Sản Phẩm)
// ==========================================================================
const diffCanvas = document.createElement('canvas');
diffCanvas.width = 48;
diffCanvas.height = 48;
const diffCtx = diffCanvas.getContext('2d', { willReadFrequently: true });

function startAutoScanMonitor() {
  stopAutoScanMonitor();
  holdSteadyScore = 0;
  lastSamplePixels = null;

  autoScanInterval = setInterval(() => {
    if (!stream || !canvas || isIdentifying) return;
    if (autoScanToggle && !autoScanToggle.checked) {
      if (viewfinderCountdown) viewfinderCountdown.style.display = 'none';
      return;
    }

    const cw = canvas.width;
    const ch = canvas.height;
    if (!cw || !ch) return;

    // Sample the center viewfinder region (60% width and height)
    const cropW = Math.round(cw * 0.60);
    const cropH = Math.round(ch * 0.60);
    const cropX = Math.round((cw - cropW) / 2);
    const cropY = Math.round((ch - cropH) / 2);

    diffCtx.drawImage(canvas, cropX, cropY, cropW, cropH, 0, 0, 48, 48);
    const imgData = diffCtx.getImageData(0, 0, 48, 48);
    const pixels = imgData.data;

    if (!lastSamplePixels) {
      lastSamplePixels = new Uint8Array(pixels);
      return;
    }

    // Compute pixel diff between consecutive 350ms samples
    let diffSum = 0;
    let nonZeroCount = 0;
    for (let i = 0; i < pixels.length; i += 4) {
      const dR = Math.abs(pixels[i] - lastSamplePixels[i]);
      const dG = Math.abs(pixels[i + 1] - lastSamplePixels[i + 1]);
      const dB = Math.abs(pixels[i + 2] - lastSamplePixels[i + 2]);
      const delta = (dR + dG + dB) / 3;
      diffSum += delta;
      if (pixels[i] > 30 || pixels[i + 1] > 30 || pixels[i + 2] > 30) {
        nonZeroCount++;
      }
      lastSamplePixels[i] = pixels[i];
      lastSamplePixels[i + 1] = pixels[i + 1];
      lastSamplePixels[i + 2] = pixels[i + 2];
    }

    const avgDiff = diffSum / (48 * 48);

    // If an object is present and motion is steady/low (user holding product)
    if (avgDiff > 0.8 && avgDiff < 14.0 && nonZeroCount > 300) {
      holdSteadyScore += 350;

      if (viewfinderCountdown) {
        if (holdSteadyScore >= 700 && holdSteadyScore < 1400) {
          viewfinderCountdown.textContent = '2';
          viewfinderCountdown.style.display = 'flex';
        } else if (holdSteadyScore >= 1400 && holdSteadyScore < 2000) {
          viewfinderCountdown.textContent = '1';
          viewfinderCountdown.style.display = 'flex';
        }
      }

      if (holdSteadyScore >= 2000) {
        if (viewfinderCountdown) viewfinderCountdown.style.display = 'none';
        holdSteadyScore = 0;
        identifySampleFromCamera();
      }
    } else {
      // Rapid motion or background reset
      holdSteadyScore = Math.max(0, holdSteadyScore - 400);
      if (viewfinderCountdown && holdSteadyScore < 600) {
        viewfinderCountdown.style.display = 'none';
      }
    }
  }, 350);
}

function stopAutoScanMonitor() {
  if (autoScanInterval) {
    clearInterval(autoScanInterval);
    autoScanInterval = null;
  }
  if (viewfinderCountdown) {
    viewfinderCountdown.style.display = 'none';
  }
}

// Start Camera Stream
async function startCamera() {
  try {
    startCameraBtn.disabled = true;
    startCameraBtn.textContent = 'Đang mở camera...';

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error(
        'Trình duyệt không hỗ trợ mediaDevices hoặc trang web không chạy trên http://localhost:8000.'
      );
    }

    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      });
    } catch (constraintErr) {
      console.warn('Fallback to default camera constraints:', constraintErr);
      stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: false,
      });
    }

    video.srcObject = stream;
    await video.play();

    // Ensure video metadata loaded
    await new Promise((resolve) => {
      if (video.videoWidth > 0 && video.videoHeight > 0) {
        resolve();
      } else {
        video.onloadedmetadata = () => resolve();
        setTimeout(resolve, 800);
      }
    });

    // Notify backend
    try {
      await fetch('/api/camera/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'start' }),
      });
    } catch (e) {
      console.warn('Backend camera notification warning:', e);
    }

    // Toggle UI display
    placeholder.style.display = 'none';
    canvas.style.display = 'block';
    cameraHud.style.display = 'flex';
    streamStatusText.textContent = 'Camera: Đang hoạt động (Trạm nhận diện)';
    liveDotIndicator.style.opacity = '1';

    stopCameraBtn.disabled = false;
    snapshotBtn.disabled = false;
    if (identifySampleBtn) identifySampleBtn.disabled = false;
    if (cameraViewfinder) cameraViewfinder.style.display = 'flex';
    startCameraBtn.style.display = 'none';

    // Start timer
    sessionSeconds = 0;
    sessionTimer = setInterval(() => {
      sessionSeconds++;
      if (hudTimerBadge) hudTimerBadge.textContent = `Thời gian: ${formatTime(sessionSeconds)}`;
    }, 1000);

    // Launch render loop
    lastFpsTime = performance.now();
    framesThisSecond = 0;
    if (animFrameId) cancelAnimationFrame(animFrameId);
    animFrameId = requestAnimationFrame(processCameraFrames);

    // Launch Auto-Scan on Hold
    startAutoScanMonitor();

    if (typeof window.setAiActive === 'function') window.setAiActive(true);

    showToast('Camera đã bật! Đưa sản phẩm vào khung ngắm để nhận diện.', 'success');
  } catch (error) {
    console.error('Không thể truy cập camera:', error);
    let msg = error.message || 'Cần cấp quyền camera trong trình duyệt';
    if (error.name === 'NotAllowedError' || error.name === 'PermissionDeniedError') {
      msg = 'Trình duyệt chưa được cấp quyền camera. Vui lòng bấm vào biểu tượng camera trên thanh địa chỉ và chọn "Cho phép (Allow)"!';
    } else if (error.name === 'NotFoundError' || error.name === 'DevicesNotFoundError') {
      msg = 'Không tìm thấy thiết bị webcam nào được kết nối với máy tính.';
    } else if (error.name === 'NotReadableError' || error.name === 'TrackStartError') {
      msg = 'Camera đang bị ứng dụng khác chiếm dụng (Zoom, Teams, OBS...). Vui lòng đóng ứng dụng đó và thử lại!';
    }
    showToast(msg, 'error');
    startCameraBtn.disabled = false;
    startCameraBtn.style.display = 'inline-flex';
    startCameraBtn.innerHTML = `
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <polygon points="5 3 19 12 5 21 5 3"></polygon>
      </svg>
      Bật Camera
    `;
  }
}

// Stop Camera Stream
function stopCamera() {
  stopAutoScanMonitor();

  if (stream) {
    stream.getTracks().forEach((track) => track.stop());
    stream = null;
  }

  if (animFrameId) {
    cancelAnimationFrame(animFrameId);
    animFrameId = null;
  }

  if (sessionTimer) {
    clearInterval(sessionTimer);
    sessionTimer = null;
  }

  if (canvas) {
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    canvas.style.display = 'none';
  }

  placeholder.style.display = 'flex';
  cameraHud.style.display = 'none';
  if (cameraViewfinder) cameraViewfinder.style.display = 'none';

  streamStatusText.textContent = 'Camera: Đã ngắt kết nối';
  liveDotIndicator.style.opacity = '0.3';

  startCameraBtn.disabled = false;
  startCameraBtn.style.display = 'inline-flex';
  startCameraBtn.innerHTML = `
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <polygon points="5 3 19 12 5 21 5 3"></polygon>
    </svg>
    Bật Camera
  `;

  stopCameraBtn.disabled = true;
  snapshotBtn.disabled = true;
  if (identifySampleBtn) identifySampleBtn.disabled = true;

  if (typeof window.setAiActive === 'function') window.setAiActive(false);

  fetch('/api/camera/stop', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ action: 'stop' }),
  }).catch(() => {});

  showToast('Đã dừng phát Camera.', 'info');
}

// Full Snapshot Capture
snapshotBtn.addEventListener('click', () => {
  if (!canvas || !stream) {
    showToast('Chưa có luồng camera để chụp ảnh', 'warning');
    return;
  }

  const tempCanvas = document.createElement('canvas');
  tempCanvas.width = canvas.width;
  tempCanvas.height = canvas.height;
  const tempCtx = tempCanvas.getContext('2d');
  tempCtx.drawImage(canvas, 0, 0);

  const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
  const dataUrl = tempCanvas.toDataURL('image/png');
  const a = document.createElement('a');
  a.href = dataUrl;
  a.download = `product_sample_${timestamp}.png`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);

  showToast('Đã lưu ảnh chụp khung hình camera!', 'success');
});

// ==========================================================================
// SINGLE-UNIT SAMPLE REGISTRATION & PERSISTENCE
// ==========================================================================

function renderActiveSampleCard(sample) {
  if (!activeSampleWrapper) return;
  if (!sample) {
    activeSampleWrapper.innerHTML = `
      <div class="kv-panel">
        <div class="kv-panel-header">Mục tiêu đếm đang chọn</div>
        <div class="kv-list">
          <div class="kv-row">
            <span class="kv-key">Mẫu:</span>
            <span class="kv-val">Chưa chọn</span>
          </div>
          <div class="kv-row">
            <span class="kv-key">Phân loại:</span>
            <span class="kv-val">-</span>
          </div>
          <div class="kv-row">
            <span class="kv-key">Hãng:</span>
            <span class="kv-val">-</span>
          </div>
          <div class="kv-row">
            <span class="kv-key">Chế độ lọc:</span>
            <span class="kv-val">Đếm tổng quát</span>
          </div>
        </div>
        <div class="empty-state" style="margin-top: 10px;">
          Đặt sản phẩm vào khung ngắm rồi nhấn "Chụp mẫu" để lưu thông tin.
        </div>
      </div>
    `;
    return;
  }

  const thumbSrc = sample.sample_image_url || '/static/assets/logo.png';
  const brand = sample.brand || 'Chưa rõ';
  const visualClass = sample.visual_class || 'Sản phẩm';
  const specs = sample.specs || '';

  activeSampleWrapper.innerHTML = `
    <div class="kv-panel">
      <div class="kv-panel-header" style="display: flex; justify-content: space-between; align-items: center;">
        <span>Mục tiêu đếm đang chọn</span>
        <button id="btnCardClearTarget" class="link-btn" style="font-size: 11px;">Bỏ chọn</button>
      </div>
      <div style="display: flex; gap: 10px; margin-bottom: 12px; align-items: center;">
        <img src="${thumbSrc}" alt="${sample.product_name}" class="sample-thumb-mini" style="width: 36px; height: 36px;" />
        <div>
          <div style="font-weight: 600; font-size: 13px;">${sample.product_name}</div>
          <div style="font-size: 11px; color: var(--text-muted);">${brand}</div>
        </div>
      </div>
      <div class="kv-list">
        <div class="kv-row">
          <span class="kv-key">Phân loại:</span>
          <span class="kv-val">${visualClass}</span>
        </div>
        <div class="kv-row">
          <span class="kv-key">Hãng:</span>
          <span class="kv-val">${brand}</span>
        </div>
        ${specs ? `
        <div class="kv-row">
          <span class="kv-key">Quy cách:</span>
          <span class="kv-val">${specs}</span>
        </div>` : ''}
        <div class="kv-row">
          <span class="kv-key">Chế độ lọc:</span>
          <span class="kv-val text-accent">Chỉ đếm mẫu này</span>
        </div>
      </div>
      <div style="margin-top: 14px;">
        <a href="/upload" class="btn btn-primary btn-sm btn-wide">
          Chuyển sang đếm qua video
        </a>
      </div>
    </div>
  `;

  const btnCardClear = document.getElementById('btnCardClearTarget');
  if (btnCardClear) {
    btnCardClear.addEventListener('click', clearActiveTarget);
  }
}

function setActiveTarget(sample) {
  try {
    localStorage.setItem('activeTargetProduct', JSON.stringify(sample));
  } catch (e) {
    console.warn('Could not save to localStorage:', e);
  }
  renderActiveSampleCard(sample);
  loadRecentSamplesPreview();
}

function clearActiveTarget() {
  localStorage.removeItem('activeTargetProduct');
  renderActiveSampleCard(null);
  loadRecentSamplesPreview();
  showToast('Đã bỏ chọn mục tiêu sản phẩm.', 'info');
}

// Single-Unit Identification via Viewfinder Crop
async function identifySampleFromCamera() {
  if (isIdentifying) return;
  if (!canvas || !stream) {
    showToast('Vui lòng bật camera trước khi chụp nhận diện mẫu.', 'warning');
    return;
  }

  isIdentifying = true;
  const viewfinderBox = cameraViewfinder ? cameraViewfinder.querySelector('.viewfinder-box') : null;

  identifySampleBtn.disabled = true;
  const originalBtnHtml = identifySampleBtn.innerHTML;
  identifySampleBtn.innerHTML = 'Đang phân tích mẫu...';

  if (viewfinderBox) {
    viewfinderBox.classList.add('is-scanning');
  }
  if (viewfinderInstruction) {
    viewfinderInstruction.textContent = 'Đang quét vùng lấy nét và phân tích...';
  }

  try {
    const cw = canvas.width || 640;
    const ch = canvas.height || 480;
    const cropW = Math.max(80, Math.round(cw * 0.60));
    const cropH = Math.max(80, Math.round(ch * 0.60));
    const cropX = Math.round((cw - cropW) / 2);
    const cropY = Math.round((ch - cropH) / 2);

    const cropCanvas = document.createElement('canvas');
    cropCanvas.width = cropW;
    cropCanvas.height = cropH;
    const cropCtx = cropCanvas.getContext('2d');
    cropCtx.drawImage(canvas, cropX, cropY, cropW, cropH, 0, 0, cropW, cropH);

    const cropBase64 = cropCanvas.toDataURL('image/jpeg', 0.92);

    const res = await fetch('/api/vision/identify-sample', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: cropBase64 }),
    });

    if (!res.ok) {
      const errJson = await res.json().catch(() => ({}));
      throw new Error(errJson.detail || `Lỗi HTTP ${res.status}`);
    }

    const resData = await res.json();
    if (resData.success && resData.sample) {
      setActiveTarget(resData.sample);

      if (viewfinderBox) {
        viewfinderBox.classList.remove('is-scanning');
        viewfinderBox.classList.add('is-success');
      }
      if (viewfinderInstruction) {
        viewfinderInstruction.textContent = `Đã nhận diện: ${resData.sample.product_name}`;
      }

      const toastMsg = resData.notice || `Nhận diện thành công: "${resData.sample.product_name}"`;
      showToast(toastMsg, 'success');

      setTimeout(() => {
        if (viewfinderBox) viewfinderBox.classList.remove('is-success');
        if (viewfinderInstruction) {
          viewfinderInstruction.textContent = 'Đặt 1 sản phẩm vào giữa khung ngắm rồi nhấn "Chụp mẫu" (hoặc giữ yên 2 giây)';
        }
      }, 2500);
    } else {
      throw new Error('Không nhận được dữ liệu mẫu từ máy chủ.');
    }
  } catch (err) {
    console.error('Lỗi nhận diện mẫu:', err);
    if (viewfinderBox) viewfinderBox.classList.remove('is-scanning');
    if (viewfinderInstruction) {
      viewfinderInstruction.textContent = 'Đặt 1 sản phẩm vào giữa khung ngắm rồi nhấn "Chụp mẫu" (hoặc giữ yên 2 giây)';
    }
    showToast(`Lỗi nhận diện: ${err.message}`, 'error');
  } finally {
    isIdentifying = false;
    identifySampleBtn.disabled = false;
    identifySampleBtn.innerHTML = originalBtnHtml;
  }
}

// Load Recent Samples Preview in Sidebar
async function loadRecentSamplesPreview() {
  if (!cameraRecentSamplesList) return;

  try {
    const res = await fetch('/api/vision/samples');
    if (!res.ok) throw new Error('Không thể tải');
    const data = await res.json();
    const samples = (data.samples || []).slice(0, 3);

    if (samples.length === 0) {
      cameraRecentSamplesList.innerHTML = `
        <div class="empty-state">
          Chưa có mẫu nào được lưu.
        </div>
      `;
      return;
    }

    let activeTargetId = null;
    try {
      const activeSaved = JSON.parse(localStorage.getItem('activeTargetProduct') || '{}');
      activeTargetId = activeSaved.id;
    } catch (_) {}

    cameraRecentSamplesList.innerHTML = samples.map((s) => {
      const isActive = s.id === activeTargetId;
      const thumb = s.sample_image_url || '/static/assets/logo.png';
      return `
        <div class="sample-row-flat ${isActive ? 'is-active' : ''}">
          <div style="display: flex; align-items: center; gap: 8px; overflow: hidden;">
            <img src="${thumb}" alt="${s.product_name}" class="sample-thumb-mini" style="width: 24px; height: 24px;" />
            <div style="overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: 12px; font-weight: 500;">
              ${s.product_name}
            </div>
          </div>
          <button type="button" class="btn btn-secondary btn-sm" onclick='${isActive ? "clearActiveTarget()" : `selectSampleFromModal(${JSON.stringify(s).replace(/'/g, "&#39;")})`}'>
            ${isActive ? 'Bỏ chọn' : 'Chọn'}
          </button>
        </div>
      `;
    }).join('');
  } catch (err) {
    cameraRecentSamplesList.innerHTML = '';
  }
}

// Modal Library Management
async function loadSampleLibrary() {
  if (!sampleLibraryList) return;
  sampleLibraryList.innerHTML = '<div class="empty-state">Đang tải danh sách mẫu...</div>';

  try {
    const res = await fetch('/api/vision/samples');
    if (!res.ok) {
      throw new Error(`Lỗi tải dữ liệu (${res.status})`);
    }
    const data = await res.json();
    const samples = data.samples || [];

    if (samples.length === 0) {
      sampleLibraryList.innerHTML = `
        <div class="empty-state">
          Chưa có mẫu nào được lưu. Hãy đưa sản phẩm vào khung ngắm camera và nhấn Chụp mẫu.
        </div>
      `;
      return;
    }

    let activeTargetId = null;
    try {
      const activeSaved = JSON.parse(localStorage.getItem('activeTargetProduct') || '{}');
      activeTargetId = activeSaved.id;
    } catch (_) {}

    sampleLibraryList.innerHTML = `
      <div class="sample-list-flat">
        ${samples.map((s) => {
          const isActive = s.id === activeTargetId;
          const thumb = s.sample_image_url || '/static/assets/logo.png';
          return `
            <div class="sample-row-flat ${isActive ? 'is-active' : ''}" id="sampleItem-${s.id}">
              <div style="display: flex; align-items: center; gap: 10px; overflow: hidden; flex: 1;">
                <img src="${thumb}" alt="${s.product_name}" class="sample-thumb-mini" />
                <div style="overflow: hidden;">
                  <div style="font-weight: 600; font-size: 13px;">${s.product_name}</div>
                  <div style="font-size: 11px; color: var(--text-muted); display: flex; gap: 8px;">
                    <span>Hãng: ${s.brand || 'Chưa rõ'}</span>
                    <span>Loại: ${s.visual_class || 'Sản phẩm'}</span>
                    ${s.specs ? `<span>${s.specs}</span>` : ''}
                  </div>
                </div>
              </div>
              <div style="display: flex; gap: 6px;">
                <button class="btn btn-secondary btn-sm" onclick='${isActive ? "clearActiveTarget()" : `selectSampleFromModal(${JSON.stringify(s).replace(/'/g, "&#39;")})`}'>
                  ${isActive ? 'Bỏ chọn' : 'Chọn'}
                </button>
                <button class="btn btn-danger btn-sm" onclick="deleteSampleFromModal(${s.id})" title="Xóa mẫu">
                  Xóa
                </button>
              </div>
            </div>
          `;
        }).join('')}
      </div>
    `;
  } catch (err) {
    sampleLibraryList.innerHTML = `<div style="color: var(--status-danger); padding: 12px; font-size: 13px;">Lỗi tải: ${err.message}</div>`;
  }
}

window.openLibraryModal = function() {
  if (sampleLibraryModal) {
    sampleLibraryModal.style.display = 'flex';
    loadSampleLibrary();
  }
};

window.selectSampleFromModal = function(sample) {
  setActiveTarget(sample);
  if (sampleLibraryModal) sampleLibraryModal.style.display = 'none';
  showToast(`Đã chọn mục tiêu: "${sample.product_name}".`, 'success');
};

window.deleteSampleFromModal = async function(sampleId) {
  if (!confirm('Bạn có chắc chắn muốn xóa mẫu sản phẩm này khỏi thư viện vĩnh viễn?')) return;
  try {
    const res = await fetch(`/api/vision/samples/${sampleId}`, { method: 'DELETE' });
    if (!res.ok) throw new Error(`Lỗi HTTP ${res.status}`);
    showToast('Đã xóa mẫu sản phẩm khỏi thư viện.', 'info');

    try {
      const active = JSON.parse(localStorage.getItem('activeTargetProduct') || '{}');
      if (active.id === sampleId) {
        clearActiveTarget();
      }
    } catch (_) {}

    loadSampleLibrary();
    loadRecentSamplesPreview();
  } catch (e) {
    showToast(`Không thể xóa mẫu: ${e.message}`, 'error');
  }
};

// Event Listeners
startCameraBtn.addEventListener('click', startCamera);
stopCameraBtn.addEventListener('click', stopCamera);

if (identifySampleBtn) {
  identifySampleBtn.addEventListener('click', identifySampleFromCamera);
}

if (openLibraryBtn) {
  openLibraryBtn.addEventListener('click', window.openLibraryModal);
}

if (closeLibraryModalBtn) {
  closeLibraryModalBtn.addEventListener('click', () => {
    if (sampleLibraryModal) sampleLibraryModal.style.display = 'none';
  });
}

if (sampleLibraryModal) {
  sampleLibraryModal.addEventListener('click', (e) => {
    if (e.target === sampleLibraryModal) {
      sampleLibraryModal.style.display = 'none';
    }
  });
}

// Gemini Key Modal Listeners
if (btnOpenGeminiModal) {
  btnOpenGeminiModal.addEventListener('click', openGeminiModal);
}
if (closeGeminiModalBtn) {
  closeGeminiModalBtn.addEventListener('click', closeGeminiModal);
}
if (btnCancelGeminiModal) {
  btnCancelGeminiModal.addEventListener('click', closeGeminiModal);
}
if (btnSaveGeminiKey) {
  btnSaveGeminiKey.addEventListener('click', saveGeminiKeyAction);
}
if (btnToggleKeyEye) {
  btnToggleKeyEye.addEventListener('click', toggleKeyVisibility);
}

if (geminiKeyModal) {
  geminiKeyModal.addEventListener('click', (e) => {
    if (e.target === geminiKeyModal) closeGeminiModal();
  });
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
  try {
    const saved = localStorage.getItem('activeTargetProduct');
    if (saved) {
      const parsed = JSON.parse(saved);
      renderActiveSampleCard(parsed);
    } else {
      renderActiveSampleCard(null);
    }
  } catch (e) {
    renderActiveSampleCard(null);
  }
  loadRecentSamplesPreview();
  checkGeminiStatus();
});
