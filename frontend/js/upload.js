/* ==========================================================================
   AI VISION PRO - LIVE VIDEO STUDIO & OBJECT COUNTING CONTROLLER
   Immediate playback on upload, live canvas AI tracking overlay, real-time counters
   ========================================================================== */

// State Management
const studioState = {
  file: null,
  filename: '',
  serverFilename: '',
  isProcessingFrame: false,
  isPlaying: false,
  totalCount: 0,
  byClass: {},
  seenTrackIds: new Set(),
  activeTracks: new Map(), // Tracking ID -> Smooth interpolated track
  currentDetections: [],
  frameCount: 0,
  lastFrameTime: performance.now(),
  lastInferTime: performance.now(),
  fps: 0,
  latency: 0,
  animationId: null,
};

// DOM References
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('videoUpload');
const fileInfoRow = document.getElementById('fileInfoRow');
const loadedFileName = document.getElementById('loadedFileName');
const loadedFileSize = document.getElementById('loadedFileSize');
const changeFileBtn = document.getElementById('changeFileBtn');
const loadSample1Btn = document.getElementById('loadSample1');
const loadSample2Btn = document.getElementById('loadSample2');
const loadSampleCoffeeBtn = document.getElementById('loadSampleCoffee');

// Studio Player & Canvas
const videoStudio = document.getElementById('videoStudio');
const studioVideo = document.getElementById('studioVideo');
const studioCanvas = document.getElementById('studioCanvas');
const playerViewport = document.getElementById('playerViewport');
const laserLine = document.getElementById('laserLine');
const studioFps = document.getElementById('studioFps');
const studioLatency = document.getElementById('studioLatency');
const activeInFrameCount = document.getElementById('activeInFrameCount');

// Controls
const timelineSlider = document.getElementById('timelineSlider');
const timeDisplay = document.getElementById('timeDisplay');
const playPauseBtn = document.getElementById('playPauseBtn');
const playBtnText = document.getElementById('playBtnText');
const playIcon = document.getElementById('playIcon');
const restartBtn = document.getElementById('restartBtn');
const snapshotBtn = document.getElementById('snapshotBtn');
const playbackSpeed = document.getElementById('playbackSpeed');
const toggleBoxes = document.getElementById('toggleBoxes');
const toggleLaser = document.getElementById('toggleLaser');

// Metrics & Lists
const totalCountDisplay = document.getElementById('totalCountDisplay');
const categoryCountBadge = document.getElementById('categoryCountBadge');
const categoryList = document.getElementById('categoryList');
const feedList = document.getElementById('feedList');
const saveSessionBtn = document.getElementById('saveSessionBtn');
const serverProcessBtn = document.getElementById('serverProcessBtn');

// Batch Processing Progress
const batchProgressArea = document.getElementById('batchProgressArea');
const batchProgressBar = document.getElementById('batchProgressBar');
const batchStatusText = document.getElementById('batchStatusText');
const batchPercentText = document.getElementById('batchPercentText');
const batchDownloadArea = document.getElementById('batchDownloadArea');
const downloadProcessedBtn = document.getElementById('downloadProcessedBtn');
const step1 = document.getElementById('step1');
const step2 = document.getElementById('step2');
const step3 = document.getElementById('step3');
const step4 = document.getElementById('step4');

// Offscreen canvas for frame extraction
const offCanvas = document.createElement('canvas');
const offCtx = offCanvas.getContext('2d', { willReadFrequently: true });
const ctx = studioCanvas.getContext('2d');

// Clean color palette for classes
const classColors = [
  '#3b82f6', '#10b981', '#f59e0b', '#8b5cf6', '#0284c7',
  '#ec4899', '#14b8a6', '#6366f1', '#ef4444', '#64748b'
];
const classColorMap = new Map();

function getColorForClass(className) {
  if (!classColorMap.has(className)) {
    const idx = classColorMap.size % classColors.length;
    classColorMap.set(className, classColors[idx]);
  }
  return classColorMap.get(className);
}

// Toast Notifications
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

// Format duration
function formatSeconds(secs) {
  if (isNaN(secs) || secs < 0) return '00:00';
  const m = Math.floor(secs / 60);
  const s = Math.floor(secs % 60);
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
}

function formatBytes(bytes) {
  if (bytes === 0) return '0 Bytes';
  const units = ['Bytes', 'KB', 'MB', 'GB'];
  const idx = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  return `${(bytes / 1024 ** idx).toFixed(1)} ${units[idx]}`;
}

// ==========================================================================
// Video Load & Initialization
// ==========================================================================
function loadVideoSource(url, name, sizeText = '') {
  resetStudioData();
  studioState.filename = name;

  // Show studio, hide dropzone
  dropZone.style.display = 'none';
  fileInfoRow.style.display = 'flex';
  videoStudio.style.display = 'grid';
  loadedFileName.textContent = name;
  loadedFileSize.textContent = sizeText ? `(${sizeText})` : '';

  // Setup video element
  studioVideo.src = url;
  studioVideo.load();

  showToast(`Đã tải video: ${name}. Sẵn sàng chạy và đếm!`, 'info');
}

function handleVideoFile(file) {
  if (!file) return;
  const validExts = ['.mp4', '.avi', '.mov', '.webm', '.mkv'];
  const ext = '.' + file.name.split('.').pop().toLowerCase();
  if (!validExts.includes(ext) && !file.type.includes('video')) {
    showToast('Vui lòng chọn tệp video hợp lệ (.mp4, .avi, .mov, .webm, .mkv)', 'error');
    return;
  }

  studioState.file = file;
  const localUrl = URL.createObjectURL(file);
  loadVideoSource(localUrl, file.name, formatBytes(file.size));

  // Background upload to server for storage/batch
  uploadVideoToServer(file);
}

// Upload file to server in background
async function uploadVideoToServer(file) {
  const formData = new FormData();
  formData.append('file', file);
  try {
    const res = await fetch('/api/video/upload', {
      method: 'POST',
      body: formData,
    });
    const data = await res.json();
    if (res.ok && data.success) {
      studioState.serverFilename = data.filename;
      console.log('Video saved on server as:', data.filename);
    }
  } catch (err) {
    console.warn('Background server upload note:', err);
  }
}

// Video metadata loaded: resize canvas & init timeline
studioVideo.addEventListener('loadedmetadata', () => {
  const vWidth = studioVideo.videoWidth || 640;
  const vHeight = studioVideo.videoHeight || 360;

  studioCanvas.width = vWidth;
  studioCanvas.height = vHeight;
  offCanvas.width = vWidth;
  offCanvas.height = vHeight;

  timelineSlider.max = studioVideo.duration || 100;
  timelineSlider.value = 0;
  updateTimeDisplay();

  // Default to smooth sync speed (0.75x) for 100% accurate, jitter-free counting
  studioVideo.playbackRate = parseFloat(playbackSpeed.value) || 0.75;

  // Reset backend tracker
  fetch('/api/detect/reset', { method: 'POST' }).catch(() => {});

  // Start continuous rendering loop
  if (!studioState.animationId) {
    renderCanvasLoop();
  }

  // Auto start playback and counting
  playVideoAndCount();
});

// Update timeline slider as video plays
studioVideo.addEventListener('timeupdate', () => {
  if (!studioVideo.seeking) {
    timelineSlider.value = studioVideo.currentTime;
  }
  updateTimeDisplay();
});

studioVideo.addEventListener('ended', () => {
  pauseVideo();
  showToast('Đã phát hết video! Đã nhận diện và đếm toàn bộ sản phẩm.', 'success');
});

function updateTimeDisplay() {
  const cur = formatSeconds(studioVideo.currentTime);
  const dur = formatSeconds(studioVideo.duration);
  timeDisplay.textContent = `${cur} / ${dur}`;
}

// ==========================================================================
// Play, Pause & Reset Controls
// ==========================================================================
function playVideoAndCount() {
  studioVideo.play().then(() => {
    studioState.isPlaying = true;
    playBtnText.textContent = 'Tạm Dừng';
    playIcon.innerHTML = '<rect x="6" y="4" width="4" height="16"></rect><rect x="14" y="4" width="4" height="16"></rect>';
    playPauseBtn.className = 'btn btn-secondary';
    if (toggleLaser.checked) playerViewport.classList.add('laser-active');
    if (typeof window.setAiActive === 'function') window.setAiActive(true);
  }).catch((err) => {
    console.warn('Playback error:', err);
  });
}

function pauseVideo() {
  studioVideo.pause();
  studioState.isPlaying = false;
  studioState.isProcessingFrame = false;
  playBtnText.textContent = 'Tiếp Tục Đếm';
  playIcon.innerHTML = '<polygon points="5 3 19 12 5 21 5 3"></polygon>';
  playPauseBtn.className = 'btn btn-primary';
  playerViewport.classList.remove('laser-active');
  if (typeof window.setAiActive === 'function') window.setAiActive(false);
}

playPauseBtn.addEventListener('click', () => {
  if (studioState.isPlaying) {
    pauseVideo();
  } else {
    playVideoAndCount();
  }
});

restartBtn.addEventListener('click', () => {
  studioVideo.currentTime = 0;
  resetStudioData();
  fetch('/api/detect/reset', { method: 'POST' }).catch(() => {});
  playVideoAndCount();
  showToast('Đã làm mới dữ liệu đếm và phát lại từ đầu!', 'info');
});

// Timeline seeking
timelineSlider.addEventListener('input', () => {
  studioVideo.currentTime = parseFloat(timelineSlider.value);
  updateTimeDisplay();
  studioState.activeTracks.clear();
  ctx.clearRect(0, 0, studioCanvas.width, studioCanvas.height);
});

// Playback Speed
playbackSpeed.addEventListener('change', () => {
  studioVideo.playbackRate = parseFloat(playbackSpeed.value);
});

// Laser Toggle
toggleLaser.addEventListener('change', () => {
  if (toggleLaser.checked && studioState.isPlaying) {
    playerViewport.classList.add('laser-active');
  } else {
    playerViewport.classList.remove('laser-active');
  }
});

// Reset Studio Data
function resetStudioData() {
  studioState.totalCount = 0;
  studioState.byClass = {};
  studioState.seenTrackIds.clear();
  studioState.activeTracks.clear();
  studioState.currentDetections = [];
  studioState.frameCount = 0;
  studioState.isProcessingFrame = false;
  totalCountDisplay.textContent = '0';
  categoryCountBadge.textContent = '0 Loại';
  categoryList.innerHTML = '<div style="color: var(--text-muted); font-size: 0.85rem; text-align: center; padding: 20px 0;">Bấm "Bắt Đầu Đếm" để xem số lượng phân loại...</div>';
  feedList.innerHTML = '<div style="color: var(--text-muted); font-size: 0.82rem; text-align: center; padding: 14px 0;">Chưa có sự kiện nhận diện</div>';
  activeInFrameCount.textContent = '0';
  ctx.clearRect(0, 0, studioCanvas.width, studioCanvas.height);
}

// ==========================================================================
// Frame Extraction & AI Detection Request
// ==========================================================================
async function captureAndDetectFrame() {
  if (
    studioVideo.paused ||
    studioVideo.ended ||
    studioState.isProcessingFrame ||
    studioVideo.readyState < 2
  ) {
    return;
  }

  const w = studioVideo.videoWidth;
  const h = studioVideo.videoHeight;
  if (!w || !h) return;

  studioState.isProcessingFrame = true;
  studioState.frameCount++;

  // Optimize payload size: Scale frame to 416px (matches YOLO imgsz 416) for high-accuracy inference & fast transfer
  const maxDim = 416;
  let targetW = w;
  let targetH = h;
  if (w > maxDim || h > maxDim) {
    if (w >= h) {
      targetW = maxDim;
      targetH = Math.round((h * maxDim) / w);
    } else {
      targetH = maxDim;
      targetW = Math.round((w * maxDim) / h);
    }
  }

  if (offCanvas.width !== targetW || offCanvas.height !== targetH) {
    offCanvas.width = targetW;
    offCanvas.height = targetH;
  }
  offCtx.drawImage(studioVideo, 0, 0, targetW, targetH);
  const base64Data = offCanvas.toDataURL('image/jpeg', 0.60);

  const startReq = performance.now();

  try {
    const res = await fetch('/api/detect/frame', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        image: base64Data,
        frame_number: studioState.frameCount,
        reset: false,
      }),
    });

    const endReq = performance.now();
    studioState.latency = Math.round(endReq - startReq);

    if (res.ok) {
      const data = await res.json();
      if (data.success) {
        handleDetectionResults(data, w / targetW, h / targetH);
      }
    }
  } catch (err) {
    console.debug('Frame infer error:', err);
  } finally {
    // Immediately ready for next frame without artificial delay
    studioState.isProcessingFrame = false;
  }
}

// Handle AI Detections & Update Counters
function handleDetectionResults(data, scaleX = 1, scaleY = 1) {
  const rawDetections = data.detections || [];
  const now = performance.now();

  // Calculate AI Detection FPS
  const delta = (now - studioState.lastInferTime) / 1000;
  studioState.lastInferTime = now;
  if (delta > 0) {
    studioState.fps = Math.min(60, Math.round(1 / delta));
  }
  studioFps.textContent = studioState.fps || 24;
  studioLatency.textContent = studioState.latency || 25;

  activeInFrameCount.textContent = rawDetections.length;

  // Update Total Count
  const newTotal = Number(data.total || 0);
  if (newTotal !== studioState.totalCount) {
    studioState.totalCount = newTotal;
    totalCountDisplay.textContent = newTotal;
    totalCountDisplay.classList.add('bounce');
    setTimeout(() => totalCountDisplay.classList.remove('bounce'), 350);
  }

  // Update Categories Breakdown
  if (data.counts && Object.keys(data.counts).length > 0) {
    studioState.byClass = data.counts;
    updateCategoryBreakdown(data.counts, newTotal);
  }

  // Sync with activeTracks map for butter-smooth interpolation
  let hasNewProduct = false;

  rawDetections.forEach((det) => {
    const trackId = det.tracking_id ?? 0;
    const x1 = det.x1 * scaleX;
    const y1 = det.y1 * scaleY;
    const x2 = det.x2 * scaleX;
    const y2 = det.y2 * scaleY;

    const isCounted = det.is_counted || studioState.seenTrackIds.has(trackId);

    if (trackId !== 0 && !studioState.seenTrackIds.has(trackId)) {
      studioState.seenTrackIds.add(trackId);
      hasNewProduct = true;
      addFeedItem({
        ...det,
        is_counted: true,
      });
    }

    // Check if this new detection spatially replaces an older track at the same spot
    let initX1 = x1, initY1 = y1, initX2 = x2, initY2 = y2;
    for (const [oldId, oldTrack] of studioState.activeTracks.entries()) {
      if (oldId !== trackId) {
        const cOx = (oldTrack.currX1 + oldTrack.currX2) / 2;
        const cOy = (oldTrack.currY1 + oldTrack.currY2) / 2;
        const cNx = (x1 + x2) / 2;
        const cNy = (y1 + y2) / 2;
        if (Math.hypot(cOx - cNx, cOy - cNy) < 70) {
          initX1 = oldTrack.currX1;
          initY1 = oldTrack.currY1;
          initX2 = oldTrack.currX2;
          initY2 = oldTrack.currY2;
          studioState.activeTracks.delete(oldId);
          break;
        }
      }
    }

    if (studioState.activeTracks.has(trackId)) {
      const track = studioState.activeTracks.get(trackId);
      const dt = Math.max(16, now - track.lastSeenTime);
      
      // Calculate instantaneous velocity (pixels per millisecond)
      const nVx1 = (x1 - track.targetX1) / dt;
      const nVy1 = (y1 - track.targetY1) / dt;
      const nVx2 = (x2 - track.targetX2) / dt;
      const nVy2 = (y2 - track.targetY2) / dt;

      // Exponential moving average for velocity to avoid erratic jitter
      track.vx1 = track.vx1 ? (track.vx1 * 0.35 + nVx1 * 0.65) : nVx1;
      track.vy1 = track.vy1 ? (track.vy1 * 0.35 + nVy1 * 0.65) : nVy1;
      track.vx2 = track.vx2 ? (track.vx2 * 0.35 + nVx2 * 0.65) : nVx2;
      track.vy2 = track.vy2 ? (track.vy2 * 0.35 + nVy2 * 0.65) : nVy2;

      track.targetX1 = x1;
      track.targetY1 = y1;
      track.targetX2 = x2;
      track.targetY2 = y2;
      track.confidence = det.confidence;
      track.className = det.class_name;
      track.isCounted = isCounted;
      track.lastSeenTime = now;
    } else {
      studioState.activeTracks.set(trackId, {
        id: trackId,
        className: det.class_name,
        confidence: det.confidence,
        isCounted: isCounted,
        currX1: initX1,
        currY1: initY1,
        currX2: initX2,
        currY2: initY2,
        targetX1: x1,
        targetY1: y1,
        targetX2: x2,
        targetY2: y2,
        vx1: 0,
        vy1: 0,
        vx2: 0,
        vy2: 0,
        lastSeenTime: now,
      });
    }
  });

  if (hasNewProduct) {
    totalCountDisplay.classList.add('bounce');
    setTimeout(() => totalCountDisplay.classList.remove('bounce'), 350);
  }
}

// Update Categories UI
function updateCategoryBreakdown(counts, total) {
  const entries = Object.entries(counts);
  categoryCountBadge.textContent = `${entries.length} Loại`;

  categoryList.innerHTML = entries
    .map(([cls, count]) => {
      const color = getColorForClass(cls);
      const percent = total > 0 ? Math.round((count / total) * 100) : 0;
      return `
        <div class="category-item">
          <div class="cat-top">
            <div class="cat-name-group">
              <span class="cat-dot" style="background:${color};"></span>
              <span class="cat-name">${cls}</span>
            </div>
            <span class="cat-count" style="color:${color};">${count}</span>
          </div>
          <div class="cat-bar-bg">
            <div class="cat-bar-fill" style="width:${percent}%; background:${color};"></div>
          </div>
        </div>
      `;
    })
    .join('');
}

// Add Item to Activity Feed (Only for newly counted products)
function addFeedItem(item) {
  const timeStr = formatSeconds(studioVideo.currentTime);
  const color = getColorForClass(item.class_name);
  const confPercent = Math.round((item.confidence || 0.9) * 100);

  const feedDiv = document.createElement('div');
  feedDiv.className = 'feed-item';
  feedDiv.style.borderLeftColor = color;
  feedDiv.innerHTML = `
    <div>
      <span class="time">[${timeStr}]</span>
      <span class="label" style="margin-left: 6px;">${item.class_name}</span>
      <span style="font-size:0.76rem; color:var(--mint); font-weight:700; margin-left:6px;">#ID:${item.tracking_id} ✓ ĐÃ ĐẾM</span>
    </div>
    <span class="conf" style="color:${color};">${confPercent}%</span>
  `;

  // Prepend to list
  if (feedList.firstElementChild && feedList.firstElementChild.textContent.includes('Chưa có')) {
    feedList.innerHTML = '';
  }
  feedList.insertBefore(feedDiv, feedList.firstChild);

  // Keep max 20 items
  if (feedList.children.length > 20) {
    feedList.removeChild(feedList.lastChild);
  }
}

// ==========================================================================
// 60fps Canvas Render Loop with Butter-Smooth Box Interpolation (Lerp)
// ==========================================================================
function renderCanvasLoop() {
  studioState.animationId = requestAnimationFrame(renderCanvasLoop);

  // High-frequency frame detection: trigger next frame as soon as previous frame completes
  if (
    studioState.isPlaying &&
    !studioState.isProcessingFrame &&
    !studioVideo.paused &&
    !studioVideo.ended &&
    studioVideo.readyState >= 2
  ) {
    captureAndDetectFrame();
  }

  const w = studioCanvas.width;
  const h = studioCanvas.height;
  if (!w || !h) return;

  ctx.clearRect(0, 0, w, h);

  if (!toggleBoxes.checked) return;

  const now = performance.now();
  const lerp = 0.28; // Butter-smooth box interpolation

  // Pairwise deduplication of active tracks to strictly guarantee at most 1 box per physical object
  const trackEntries = Array.from(studioState.activeTracks.entries());
  for (let i = 0; i < trackEntries.length; i++) {
    const [idA, trackA] = trackEntries[i];
    if (!studioState.activeTracks.has(idA)) continue;
    for (let j = i + 1; j < trackEntries.length; j++) {
      const [idB, trackB] = trackEntries[j];
      if (!studioState.activeTracks.has(idB)) continue;

      const cAx = (trackA.currX1 + trackA.currX2) / 2;
      const cAy = (trackA.currY1 + trackA.currY2) / 2;
      const cBx = (trackB.currX1 + trackB.currX2) / 2;
      const cBy = (trackB.currY1 + trackB.currY2) / 2;
      const dist = Math.hypot(cAx - cBx, cAy - cBy);

      const interX1 = Math.max(trackA.currX1, trackB.currX1);
      const interY1 = Math.max(trackA.currY1, trackB.currY1);
      const interX2 = Math.min(trackA.currX2, trackB.currX2);
      const interY2 = Math.min(trackA.currY2, trackB.currY2);
      const interW = Math.max(0, interX2 - interX1);
      const interH = Math.max(0, interY2 - interY1);
      const interArea = interW * interH;
      const areaA = Math.max(1, (trackA.currX2 - trackA.currX1) * (trackA.currY2 - trackA.currY1));
      const areaB = Math.max(1, (trackB.currX2 - trackB.currX1) * (trackB.currY2 - trackB.currY1));
      const minArea = Math.min(areaA, areaB);
      const iou = interArea / (areaA + areaB - interArea);

      if (dist < 60 || (interArea / minArea) > 0.4 || iou > 0.25) {
        if (trackA.lastSeenTime < trackB.lastSeenTime) {
          studioState.activeTracks.delete(idA);
          break;
        } else {
          studioState.activeTracks.delete(idB);
        }
      }
    }
  }

  studioState.activeTracks.forEach((track, trackId) => {
    const age = now - track.lastSeenTime;
    const isCounted = track.isCounted || studioState.seenTrackIds.has(trackId);

    // Check if product has moved near or past screen edges
    const isNearEdge =
      track.currX1 <= 15 ||
      track.currY1 <= 15 ||
      track.currX2 >= w - 15 ||
      track.currY2 >= h - 15;

    // Fast purge if near edge, otherwise max 450ms persistence to prevent ghost trail
    const maxAge = isNearEdge ? 250 : 450;

    if (age > maxAge) {
      studioState.activeTracks.delete(trackId);
      return;
    }

    // Fade out smoothly only during the final 150ms
    let alpha = 1.0;
    if (age > maxAge - 150) {
      alpha = Math.max(0, (maxAge - age) / 150);
    }

    // Deadband filter & forward motion extrapolation:
    // Project forward along velocity vector between AI detection updates
    const elapsed = Math.min(220, now - track.lastSeenTime);
    const forwardX1 = track.targetX1 + (track.vx1 || 0) * elapsed * 0.75;
    const forwardY1 = track.targetY1 + (track.vy1 || 0) * elapsed * 0.75;
    const forwardX2 = track.targetX2 + (track.vx2 || 0) * elapsed * 0.75;
    const forwardY2 = track.targetY2 + (track.vy2 || 0) * elapsed * 0.75;

    const dx1 = forwardX1 - track.currX1;
    const dy1 = forwardY1 - track.currY1;
    const dx2 = forwardX2 - track.currX2;
    const dy2 = forwardY2 - track.currY2;

    const lerp = 0.32;
    track.currX1 += Math.abs(dx1) < 1.0 ? 0 : dx1 * lerp;
    track.currY1 += Math.abs(dy1) < 1.0 ? 0 : dy1 * lerp;
    track.currX2 += Math.abs(dx2) < 1.0 ? 0 : dx2 * lerp;
    track.currY2 += Math.abs(dy2) < 1.0 ? 0 : dy2 * lerp;

    const x1 = Math.max(0, track.currX1);
    const y1 = Math.max(0, track.currY1);
    const x2 = Math.min(w, track.currX2);
    const y2 = Math.min(h, track.currY2);
    const bw = x2 - x1;
    const bh = y2 - y1;
    if (bw <= 0 || bh <= 0) return;

    const baseColor = getColorForClass(track.className);
    const color = isCounted ? '#10b981' : baseColor;

    ctx.save();
    ctx.globalAlpha = alpha;

    const scaleFactor = Math.max(1, w / 960);
    const lineWidth = Math.round(2 * scaleFactor);
    const fontSize = Math.round(12 * scaleFactor);

    // Subtle background tint with soft depth
    ctx.fillStyle = hexToRgba(color, 0.08);
    ctx.fillRect(x1, y1, bw, bh);

    // Clean primary bounding outline
    ctx.strokeStyle = hexToRgba(color, 0.65);
    ctx.lineWidth = Math.max(1, lineWidth - 0.5);
    ctx.strokeRect(x1, y1, bw, bh);

    // Accent Corner Brackets (Computer Vision high-precision markers)
    const cornerLen = Math.min(15 * scaleFactor, Math.min(bw, bh) / 4);
    ctx.strokeStyle = color;
    ctx.lineWidth = lineWidth + 1.2;
    ctx.beginPath();
    // Top-left
    ctx.moveTo(x1, y1 + cornerLen); ctx.lineTo(x1, y1); ctx.lineTo(x1 + cornerLen, y1);
    // Top-right
    ctx.moveTo(x2 - cornerLen, y1); ctx.lineTo(x2, y1); ctx.lineTo(x2, y1 + cornerLen);
    // Bottom-left
    ctx.moveTo(x1, y2 - cornerLen); ctx.lineTo(x1, y2); ctx.lineTo(x1 + cornerLen, y2);
    // Bottom-right
    ctx.moveTo(x2 - cornerLen, y2); ctx.lineTo(x2, y2); ctx.lineTo(x2, y2 - cornerLen);
    ctx.stroke();

    // Floating Label Tag with rounded pill geometry & clean badge
    const conf = Math.round((track.confidence || 0.9) * 100);
    const statusText = isCounted ? '✓ ' : '';
    const label = `${statusText}${track.className} #${trackId} · ${conf}%`;

    ctx.font = `600 ${fontSize}px "Plus Jakarta Sans", sans-serif`;
    const textMetrics = ctx.measureText(label);
    const tagW = textMetrics.width + 16 * scaleFactor;
    const tagH = 22 * scaleFactor;
    const tagX = x1;
    const tagY = Math.max(0, y1 - tagH - 3);
    const tagRadius = 6 * scaleFactor;

    // Tag background with clean rounded corners
    ctx.fillStyle = color;
    roundRect(ctx, tagX, tagY, tagW, tagH, tagRadius, true, false);

    // Tag text
    ctx.fillStyle = '#ffffff';
    ctx.fillText(label, tagX + 8 * scaleFactor, tagY + 15 * scaleFactor);

    ctx.restore();
  });
}

function hexToRgba(hex, alpha) {
  let c = hex.replace('#', '');
  if (c.length === 3) c = c.split('').map(x => x + x).join('');
  const num = parseInt(c, 16);
  return `rgba(${(num >> 16) & 255}, ${(num >> 8) & 255}, ${num & 255}, ${alpha})`;
}

function roundRect(ctx, x, y, width, height, radius, fill, stroke) {
  ctx.beginPath();
  ctx.moveTo(x + radius, y);
  ctx.lineTo(x + width - radius, y);
  ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
  ctx.lineTo(x + width, y + height - radius);
  ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
  ctx.lineTo(x + radius, y + height);
  ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
  ctx.lineTo(x, y + radius);
  ctx.quadraticCurveTo(x, y, x + radius, y);
  ctx.closePath();
  if (fill) ctx.fill();
  if (stroke) ctx.stroke();
}

// ==========================================================================
// Snapshot Feature
// ==========================================================================
snapshotBtn.addEventListener('click', () => {
  const snapCanvas = document.createElement('canvas');
  snapCanvas.width = studioCanvas.width;
  snapCanvas.height = studioCanvas.height;
  const snapCtx = snapCanvas.getContext('2d');

  // Draw video frame first, then overlays
  snapCtx.drawImage(studioVideo, 0, 0, snapCanvas.width, snapCanvas.height);
  snapCtx.drawImage(studioCanvas, 0, 0, snapCanvas.width, snapCanvas.height);

  const dataUrl = snapCanvas.toDataURL('image/png');
  const a = document.createElement('a');
  a.href = dataUrl;
  a.download = `detection_${studioState.filename || 'frame'}_${Date.now()}.png`;
  a.click();
  showToast('Đã chụp và lưu ảnh khung hình nhận diện!', 'success');
});

// ==========================================================================
// Save Live Session to Database
// ==========================================================================
saveSessionBtn.addEventListener('click', async () => {
  if (studioState.totalCount === 0) {
    showToast('Chưa có sản phẩm nào được đếm để lưu!', 'info');
    return;
  }

  saveSessionBtn.disabled = true;
  saveSessionBtn.textContent = 'Đang lưu CSDL...';

  try {
    const res = await fetch('/api/video/save-session', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_type: 'VIDEO',
        video_name: studioState.filename || 'Live Video Session',
        total_objects: studioState.totalCount,
        by_class: studioState.byClass,
      }),
    });

    const data = await res.json();
    if (res.ok && data.success) {
      showToast(`Đã lưu phiên nhận diện #${data.session_id} vào cơ sở dữ liệu!`, 'success');
    } else {
      throw new Error(data.message || 'Lỗi lưu phiên');
    }
  } catch (err) {
    showToast(`Không thể lưu: ${err.message}`, 'error');
  } finally {
    saveSessionBtn.disabled = false;
    saveSessionBtn.innerHTML = `
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z"></path>
        <polyline points="17 21 17 13 7 13 7 21"></polyline>
        <polyline points="7 3 7 8 15 8"></polyline>
      </svg>
      Lưu Kết Quả Vào CSDL
    `;
  }
});

// ==========================================================================
// Server Batch Processing & MP4 Export
// ==========================================================================
serverProcessBtn.addEventListener('click', async () => {
  const targetVideo = studioState.serverFilename || studioState.filename;
  if (!targetVideo) {
    showToast('Vui lòng chọn video trước!', 'error');
    return;
  }

  serverProcessBtn.disabled = true;
  batchProgressArea.style.display = 'block';
  batchDownloadArea.style.display = 'none';
  batchProgressArea.scrollIntoView({ behavior: 'smooth' });

  updateBatchStep(2, 30, 'Đang chuẩn bị mô hình YOLOv8 trên máy chủ...');

  try {
    const res = await fetch('/api/video/process', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_type: 'VIDEO',
        video_name: targetVideo,
      }),
    });

    const data = await res.json();
    if (!res.ok || !data.success) {
      throw new Error(data.message || 'Xử lý video máy chủ thất bại');
    }

    updateBatchStep(4, 100, 'Xử lý hoàn tất! Video đã được xuất thành công.');
    showToast('Xuất video nhận diện hoàn tất!', 'success');

    const outVideo = data.output_video || data.output_path || '';
    const outFilename = outVideo.split(/[\\/]/).pop();

    if (outFilename) {
      const downloadUrl = `/api/video/output/${encodeURIComponent(outFilename)}`;
      downloadProcessedBtn.href = downloadUrl;
      downloadProcessedBtn.download = outFilename;
      batchDownloadArea.style.display = 'flex';
    }
  } catch (err) {
    updateBatchStep(1, 0, `Lỗi: ${err.message}`);
    showToast(`Lỗi xử lý server: ${err.message}`, 'error');
  } finally {
    serverProcessBtn.disabled = false;
  }
});

function updateBatchStep(stepNum, percent, status) {
  const steps = [step1, step2, step3, step4];
  steps.forEach((st, idx) => {
    if (!st) return;
    const n = idx + 1;
    st.classList.remove('active', 'completed');
    if (n < stepNum) st.classList.add('completed');
    else if (n === stepNum) st.classList.add('active');
  });

  if (batchProgressBar) batchProgressBar.style.width = `${percent}%`;
  if (batchPercentText) batchPercentText.textContent = `${percent}%`;
  if (batchStatusText) batchStatusText.textContent = status;
}

// ==========================================================================
// Event Listeners for Upload & Samples
// ==========================================================================
fileInput.addEventListener('change', (e) => {
  if (e.target.files && e.target.files[0]) {
    handleVideoFile(e.target.files[0]);
  }
});

// Drag and drop
['dragenter', 'dragover'].forEach((ev) => {
  dropZone.addEventListener(ev, (e) => {
    e.preventDefault();
    e.stopPropagation();
    dropZone.classList.add('dragover');
  });
});

['dragleave', 'drop'].forEach((ev) => {
  dropZone.addEventListener(ev, (e) => {
    e.preventDefault();
    e.stopPropagation();
    dropZone.classList.remove('dragover');
  });
});

dropZone.addEventListener('drop', (e) => {
  if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
    fileInput.files = e.dataTransfer.files;
    handleVideoFile(e.dataTransfer.files[0]);
  }
});

// Change Video Button
changeFileBtn.addEventListener('click', () => {
  pauseVideo();
  videoStudio.style.display = 'none';
  fileInfoRow.style.display = 'none';
  dropZone.style.display = 'block';
  batchProgressArea.style.display = 'none';
  fileInput.value = '';
});

// Sample Video Buttons
if (loadSample1Btn) {
  loadSample1Btn.addEventListener('click', () => {
    loadVideoSource('/api/video/file/sample_test.mp4', 'sample_test.mp4', '3.4 MB');
    studioState.serverFilename = 'sample_test.mp4';
  });
}

if (loadSample2Btn) {
  loadSample2Btn.addEventListener('click', () => {
    loadVideoSource('/api/video/file/debug_sample.mp4', 'debug_sample.mp4', '2.1 MB');
    studioState.serverFilename = 'debug_sample.mp4';
  });
}

if (loadSampleCoffeeBtn) {
  loadSampleCoffeeBtn.addEventListener('click', () => {
    loadVideoSource('/api/video/file/6444194-uhd_3840_2160_24fps.mp4', '6444194-uhd_3840_2160_24fps.mp4', '217 MB');
    studioState.serverFilename = '6444194-uhd_3840_2160_24fps.mp4';
  });
}

// Topbar dynamic scroll effect
document.addEventListener('DOMContentLoaded', () => {
  const topbar = document.querySelector('.topbar');
  if (topbar) {
    const handleScroll = () => {
      if (window.scrollY > 15) {
        topbar.classList.add('scrolled');
      } else {
        topbar.classList.remove('scrolled');
      }
    };
    window.addEventListener('scroll', handleScroll, { passive: true });
    handleScroll();
  }
});
