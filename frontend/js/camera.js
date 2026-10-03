/* ==========================================================================
   AI VISION PRO - REALTIME CAMERA & AI DETECTOR HUD
   Webcam integration, live canvas tracker, dynamic bounding boxes, snapshot
   ========================================================================== */

// DOM Elements
const startCameraBtn = document.getElementById('startCameraBtn');
const stopCameraBtn = document.getElementById('stopCameraBtn');
const resetCountBtn = document.getElementById('resetCountBtn');
const snapshotBtn = document.getElementById('snapshotBtn');
const video = document.getElementById('cameraFeed');
const canvas = document.getElementById('cameraCanvas');
const placeholder = document.getElementById('cameraPlaceholder');
const cameraHud = document.getElementById('cameraHud');
const streamStatusText = document.getElementById('streamStatusText');
const liveDotIndicator = document.getElementById('liveDotIndicator');

const detectedObjects = document.getElementById('detectedObjects');
const countPanel = document.getElementById('countPanel');
const totalCount = document.getElementById('totalCount');
const activeObjectsBadge = document.getElementById('activeObjectsBadge');
const fpsCounter = document.getElementById('fpsCounter');
const hudTimerBadge = document.getElementById('hudTimerBadge');

let stream = null;
let animFrameId = null;
let sessionSeconds = 0;
let sessionTimer = null;
let lastFrameTime = performance.now();
let frameCount = 0;

// State Variables
let realCameraDetections = [];
let accumulatedCounts = {};
const cameraTracks = new Map(); // Persistent smooth tracks: trackId -> Track
const seenCountedTrackIds = new Set();

// Toast System
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

// Offscreen canvas for camera frame extraction
const camOffCanvas = document.createElement('canvas');
const camOffCtx = camOffCanvas.getContext('2d', { willReadFrequently: true });

let isInferringCamera = false;

async function sendCameraFrameToAI() {
  if (!video || video.paused || video.ended || isInferringCamera) return;
  const vw = video.videoWidth;
  const vh = video.videoHeight;
  if (!vw || !vh) return;

  isInferringCamera = true;
  frameCount++;

  // Scale frame to 384px width/height for fast YOLO inference & minimal lag
  const maxDim = 384;
  let targetW = vw;
  let targetH = vh;
  if (vw > maxDim || vh > maxDim) {
    if (vw >= vh) {
      targetW = maxDim;
      targetH = Math.round((vh * maxDim) / vw);
    } else {
      targetH = maxDim;
      targetW = Math.round((vw * maxDim) / vh);
    }
  }

  if (camOffCanvas.width !== targetW || camOffCanvas.height !== targetH) {
    camOffCanvas.width = targetW;
    camOffCanvas.height = targetH;
  }
  camOffCtx.drawImage(video, 0, 0, targetW, targetH);
  const base64 = camOffCanvas.toDataURL('image/jpeg', 0.60);

  try {
    const res = await fetch('/api/detect/frame', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        image: base64,
        frame_number: frameCount,
        reset: false,
      }),
    });
    const data = await res.json();
    if (res.ok && data.success) {
      const cw = canvas.width || vw;
      const ch = canvas.height || vh;
      const scaleX = cw / targetW;
      const scaleY = ch / targetH;
      const rawDetections = data.detections || [];
      const now = performance.now();

      // Update or add tracks to persistent cameraTracks store
      rawDetections.forEach((d) => {
        const trackId = d.tracking_id ?? 0;
        const isCounted = d.is_counted || seenCountedTrackIds.has(trackId);
        if (isCounted && trackId !== 0) {
          seenCountedTrackIds.add(trackId);
        }

        const x1 = d.x1 * scaleX;
        const y1 = d.y1 * scaleY;
        const x2 = d.x2 * scaleX;
        const y2 = d.y2 * scaleY;

        if (cameraTracks.has(trackId)) {
          const track = cameraTracks.get(trackId);
          const dt = Math.max(16, now - track.lastSeen);
          const nVx1 = (x1 - track.targetX1) / dt;
          const nVy1 = (y1 - track.targetY1) / dt;
          const nVx2 = (x2 - track.targetX2) / dt;
          const nVy2 = (y2 - track.targetY2) / dt;

          track.vx1 = track.vx1 ? (track.vx1 * 0.35 + nVx1 * 0.65) : nVx1;
          track.vy1 = track.vy1 ? (track.vy1 * 0.35 + nVy1 * 0.65) : nVy1;
          track.vx2 = track.vx2 ? (track.vx2 * 0.35 + nVx2 * 0.65) : nVx2;
          track.vy2 = track.vy2 ? (track.vy2 * 0.35 + nVy2 * 0.65) : nVy2;

          track.targetX1 = x1;
          track.targetY1 = y1;
          track.targetX2 = x2;
          track.targetY2 = y2;
          track.confidence = d.confidence || track.confidence;
          track.class_name = d.class_name;
          track.is_counted = isCounted;
          track.lastSeen = now;
        } else {
          // Check if this detection spatially replaces an older track at the same spot
          let initX1 = x1, initY1 = y1, initX2 = x2, initY2 = y2;
          for (const [oldId, oldTrack] of cameraTracks.entries()) {
            if (oldId !== trackId) {
              const cOx = (oldTrack.currX1 + oldTrack.currX2) / 2;
              const cOy = (oldTrack.currY1 + oldTrack.currY2) / 2;
              const cNx = (x1 + x2) / 2;
              const cNy = (y1 + y2) / 2;
              if (Math.hypot(cOx - cNx, cOy - cNy) < 60) {
                initX1 = oldTrack.currX1;
                initY1 = oldTrack.currY1;
                initX2 = oldTrack.currX2;
                initY2 = oldTrack.currY2;
                cameraTracks.delete(oldId);
                break;
              }
            }
          }

          cameraTracks.set(trackId, {
            id: trackId,
            class_name: d.class_name,
            confidence: d.confidence || 0.9,
            is_counted: isCounted,
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
            lastSeen: now,
          });
        }
      });

      if (data.counts && Object.keys(data.counts).length > 0) {
        accumulatedCounts = data.counts;
      }
      if (data.total !== undefined && totalCount) {
        totalCount.textContent = data.total;
      }
    }
  } catch (e) {
    console.debug('Camera AI infer error:', e);
  } finally {
    setTimeout(() => {
      isInferringCamera = false;
    }, 40);
  }
}

// Draw Clean Bounding Boxes on Canvas with Anti-Jitter and Persistence
function drawDetections(ctx, width, height) {
  ctx.drawImage(video, 0, 0, width, height);

  // Trigger backend AI inference
  sendCameraFrameToAI();

  const now = performance.now();
  const lerp = 0.28; // Soft, stable lerp smoothing
  const activeList = [];

  // Pairwise deduplication of active tracks to strictly guarantee 1 box per item
  const trackEntries = Array.from(cameraTracks.entries());
  for (let i = 0; i < trackEntries.length; i++) {
    const [idA, trackA] = trackEntries[i];
    if (!cameraTracks.has(idA)) continue;
    for (let j = i + 1; j < trackEntries.length; j++) {
      const [idB, trackB] = trackEntries[j];
      if (!cameraTracks.has(idB)) continue;

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
        if (trackA.lastSeen < trackB.lastSeen) {
          cameraTracks.delete(idA);
          break;
        } else {
          cameraTracks.delete(idB);
        }
      }
    }
  }

  cameraTracks.forEach((track, trackId) => {
    const age = now - track.lastSeen;

    // Check if product has moved out near screen edges
    const isNearEdge =
      track.currX1 <= 15 ||
      track.currY1 <= 15 ||
      track.currX2 >= width - 15 ||
      track.currY2 >= height - 15;

    // Fast purge if near edge, otherwise max 450ms persistence to prevent ghost trail
    const maxAge = isNearEdge ? 250 : 450;

    if (age > maxAge) {
      cameraTracks.delete(trackId);
      return;
    }

    // Deadband filter & forward velocity projection
    const elapsed = Math.min(220, now - track.lastSeen);
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

    const x = Math.max(0, track.currX1);
    const y = Math.max(0, track.currY1);
    const bw = Math.min(width, track.currX2) - x;
    const bh = Math.min(height, track.currY2) - y;
    if (bw <= 0 || bh <= 0) return;

    activeList.push({
      class_name: track.class_name,
      tracking_id: track.id,
      confidence: track.confidence,
      is_counted: track.is_counted,
    });

    // Fade out smoothly only during the final 150ms before removal
    let alpha = 1.0;
    if (age > maxAge - 150) {
      alpha = Math.max(0, (maxAge - age) / 150);
    }

    ctx.save();
    ctx.globalAlpha = alpha;

    const isCounted = track.is_counted || seenCountedTrackIds.has(trackId);
    const color = isCounted ? '#10b981' : '#3b82f6';

    // Subtle background tint with soft depth
    ctx.fillStyle = isCounted ? 'rgba(16, 185, 129, 0.08)' : 'rgba(59, 130, 246, 0.08)';
    ctx.fillRect(x, y, bw, bh);

    // Clean primary bounding outline
    ctx.strokeStyle = isCounted ? 'rgba(16, 185, 129, 0.65)' : 'rgba(59, 130, 246, 0.65)';
    ctx.lineWidth = 1.5;
    ctx.strokeRect(x, y, bw, bh);

    // Accent Corner Brackets (Computer Vision high-precision markers)
    const cornerLen = Math.min(14, Math.min(bw, bh) / 4);
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    // Top-left
    ctx.moveTo(x, y + cornerLen); ctx.lineTo(x, y); ctx.lineTo(x + cornerLen, y);
    // Top-right
    ctx.moveTo(x + bw - cornerLen, y); ctx.lineTo(x + bw, y); ctx.lineTo(x + bw, y + cornerLen);
    // Bottom-left
    ctx.moveTo(x, y + bh - cornerLen); ctx.lineTo(x, y + bh); ctx.lineTo(x + cornerLen, y + bh);
    // Bottom-right
    ctx.moveTo(x + bw - cornerLen, y + bh); ctx.lineTo(x + bw, y + bh); ctx.lineTo(x + bw, y + bh - cornerLen);
    ctx.stroke();

    // Clean floating label tag badge
    const confText = `${Math.round(track.confidence * 100)}%`;
    const statusMark = isCounted ? '✓ ' : '';
    const label = `${statusMark}${track.class_name} #${track.id} · ${confText}`;

    ctx.font = '600 12px "Plus Jakarta Sans", sans-serif';
    const textMetrics = ctx.measureText(label);
    const tagW = textMetrics.width + 16;
    const tagH = 22;
    const tagY = Math.max(0, y - tagH - 3);

    // Tag background with rounded pill geometry
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.roundRect ? ctx.roundRect(x, tagY, tagW, tagH, 6) : ctx.rect(x, tagY, tagW, tagH);
    ctx.fill();

    // Tag text
    ctx.fillStyle = '#ffffff';
    ctx.fillText(label, x + 8, tagY + 15);

    ctx.restore();
  });

  realCameraDetections = activeList;

  // Calculate FPS
  frameCount++;
  const fpsNow = performance.now();
  if (fpsNow - lastFrameTime >= 1000) {
    const fps = Math.round((frameCount * 1000) / (fpsNow - lastFrameTime));
    if (fpsCounter) fpsCounter.textContent = fps;
    frameCount = 0;
    lastFrameTime = fpsNow;
  }
}

// Render Side Panel Detection Cards
function updateSidebarUI() {
  if (activeObjectsBadge) {
    activeObjectsBadge.textContent = `${realCameraDetections.length} Vật thể`;
  }

  // Active detected objects list
  if (detectedObjects) {
    if (realCameraDetections.length > 0) {
      detectedObjects.innerHTML = realCameraDetections
        .map((t) => {
          return `
            <div class="detected-card">
              <div>
                <div class="title">${t.class_name}</div>
                <div class="sub">TRACK_ID: #${t.tracking_id}</div>
              </div>
              <div style="display: flex; align-items: center; gap: 6px;">
                <button class="btn-gemini-inspect" onclick="inspectTrackGemini(${t.tracking_id})" title="Soi chi tiết với Gemini Vision AI">
                  ✨ Gemini
                </button>
                <span class="confidence-chip">${Math.round(t.confidence * 100)}%</span>
              </div>
            </div>
          `;
        })
        .join('');
    } else {
      detectedObjects.innerHTML = `
        <div style="color: var(--text-muted); font-size: 0.86rem; text-align: center; padding: 24px 0;">
          Chưa có vật thể nào trong khung hình.
        </div>
      `;
    }
  }

  // Counter breakdown
  if (countPanel) {
    const entries = Object.entries(accumulatedCounts);
    if (entries.length > 0) {
      countPanel.innerHTML = entries
        .map(([name, count]) => {
          return `
            <div class="breakdown-row">
              <div style="display:flex; align-items:center; gap:8px;">
                <span style="width:8px; height:8px; border-radius:50%; background:var(--primary);"></span>
                <span style="font-weight:600;">${name}</span>
              </div>
              <strong style="color:var(--primary); font-weight:700;">${count}</strong>
            </div>
          `;
        })
        .join('');
    }
  }
}

// Frame loop
function processCameraFrames() {
  if (!stream || video.paused || video.ended) return;

  if (canvas && video.videoWidth > 0) {
    if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
    }
    const ctx = canvas.getContext('2d');
    drawDetections(ctx, canvas.width, canvas.height);
    updateSidebarUI();
  }

  animFrameId = requestAnimationFrame(processCameraFrames);
}

// Start Camera Stream
async function startCamera() {
  try {
    startCameraBtn.disabled = true;
    startCameraBtn.textContent = 'Đang mở camera...';

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error(
        'Trình duyệt không hỗ trợ mediaDevices hoặc trang web không chạy trên http://localhost:8000 / http://127.0.0.1:8000.'
      );
    }

    // Try high resolution, fallback to basic constraints if not supported
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      });
    } catch (constraintErr) {
      console.warn('Fallback to basic video constraints:', constraintErr);
      stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: false,
      });
    }

    video.srcObject = stream;
    await video.play();

    // Ensure video has actual dimensions before starting canvas render
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
    streamStatusText.textContent = 'Camera: Đang nhận diện trực tiếp';
    liveDotIndicator.style.opacity = '1';

    stopCameraBtn.disabled = false;
    snapshotBtn.disabled = false;
    startCameraBtn.style.display = 'none';

    // Start timer
    sessionSeconds = 0;
    sessionTimer = setInterval(() => {
      sessionSeconds++;
      if (hudTimerBadge) hudTimerBadge.textContent = `Thời gian: ${formatTime(sessionSeconds)}`;
    }, 1000);

    // Launch render loop
    lastFrameTime = performance.now();
    frameCount = 0;
    if (animFrameId) cancelAnimationFrame(animFrameId);
    animFrameId = requestAnimationFrame(processCameraFrames);

    if (typeof window.setAiActive === 'function') window.setAiActive(true);

    showToast('Camera đã bật và hệ thống theo dõi AI đang hoạt động!', 'success');
  } catch (error) {
    console.error('Không thể truy cập camera:', error);
    let msg = error.message || 'Cần cấp quyền camera trong trình duyệt';
    if (error.name === 'NotAllowedError' || error.name === 'PermissionDeniedError') {
      msg = 'Trình duyệt chưa được cấp quyền camera. Vui lòng nhấn vào biểu tượng ổ khóa/camera trên thanh địa chỉ trình duyệt và chọn "Cho phép (Allow)"!';
    } else if (error.name === 'NotFoundError' || error.name === 'DevicesNotFoundError') {
      msg = 'Không tìm thấy thiết bị webcam nào được kết nối với máy tính.';
    } else if (error.name === 'NotReadableError' || error.name === 'TrackStartError') {
      msg = 'Camera đang bị ứng dụng khác sử dụng (Zoom, Teams, Zalo, OBS...). Vui lòng tắt các ứng dụng đó rồi nhấn thử lại!';
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
async function stopCamera() {
  if (typeof window.setAiActive === 'function') window.setAiActive(false);

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

  video.pause();
  video.srcObject = null;

  // Backend stop notification
  try {
    await fetch('/api/camera/stop', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action: 'stop' }),
    });
  } catch (e) {
    console.warn('Backend camera stop call:', e);
  }

  // Reset UI
  placeholder.style.display = 'flex';
  canvas.style.display = 'none';
  cameraHud.style.display = 'none';
  streamStatusText.textContent = 'Camera: Đã dừng kết nối';
  liveDotIndicator.style.opacity = '0.3';

  startCameraBtn.style.display = 'inline-flex';
  startCameraBtn.disabled = false;
  startCameraBtn.innerHTML = `
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <polygon points="5 3 19 12 5 21 5 3"></polygon>
    </svg>
    Bật Camera
  `;

  stopCameraBtn.disabled = true;
  snapshotBtn.disabled = true;
  cameraTracks.clear();
  seenCountedTrackIds.clear();
  realCameraDetections = [];
  updateSidebarUI();

  showToast('Đã dừng camera và lưu phiên làm việc.', 'info');
}

// Reset Counter & Tracking
async function resetAllCounts() {
  cameraTracks.clear();
  seenCountedTrackIds.clear();
  accumulatedCounts = {};
  realCameraDetections = [];
  if (totalCount) totalCount.textContent = '0';
  if (countPanel) {
    countPanel.innerHTML = `
      <div class="breakdown-row" style="color: var(--text-muted);">
        <span>Chưa có dữ liệu</span>
        <span>-</span>
      </div>
    `;
  }
  updateSidebarUI();

  try {
    const res = await fetch('/api/detect/reset', { method: 'POST' });
    if (res.ok) {
      showToast('Đã đặt lại toàn bộ đếm và mã theo dõi về 0!', 'success');
    }
  } catch (e) {
    console.error('Reset error:', e);
  }
}

// Capture Snapshot
function captureSnapshot() {
  if (!canvas || !stream) return;
  const link = document.createElement('a');
  link.download = `snapshot_${Date.now()}.png`;
  link.href = canvas.toDataURL('image/png');
  link.click();
  showToast('Đã chụp và tải ảnh khung hình!', 'success');
}

// Event Bindings
startCameraBtn.addEventListener('click', startCamera);
stopCameraBtn.addEventListener('click', stopCamera);
if (resetCountBtn) resetCountBtn.addEventListener('click', resetAllCounts);
snapshotBtn.addEventListener('click', captureSnapshot);

// Topbar dynamic scroll effect & VisionHub Init
document.addEventListener('DOMContentLoaded', () => {
  if (window.VisionHub) {
    window.VisionHub.init('visionHubMount');
  }

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

// Gemini Inspection for specific tracked item
window.inspectTrackGemini = function (trackId) {
  const track = cameraTracks.get(trackId);
  if (!canvas) {
    if (window.VisionHub) window.VisionHub.triggerFullSceneInspection();
    return;
  }

  const cropCanvas = document.createElement('canvas');
  const cropCtx = cropCanvas.getContext('2d');
  const pad = 12;

  let sx = 0, sy = 0, sw = canvas.width, sh = canvas.height;
  if (track) {
    sx = Math.max(0, track.currX1 - pad);
    sy = Math.max(0, track.currY1 - pad);
    sw = Math.min(canvas.width - sx, Math.max(30, (track.currX2 - track.currX1) + pad * 2));
    sh = Math.min(canvas.height - sy, Math.max(30, (track.currY2 - track.currY1) + pad * 2));
  }

  cropCanvas.width = sw;
  cropCanvas.height = sh;
  cropCtx.drawImage(canvas, sx, sy, sw, sh, 0, 0, sw, sh);
  const cropBase64 = cropCanvas.toDataURL('image/jpeg', 0.9);

  if (window.VisionHub) {
    window.VisionHub.inspectProductCrop(cropBase64);
  }
};
