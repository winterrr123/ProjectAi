/**
 * ==============================================================================
 * PRODUCT VISION - VISION AI HUB CONTROLLER
 * Seamless Integration for Gemini Flash Vision & YOLO-World Open-Vocabulary AI
 * ==============================================================================
 */

window.VisionHub = (function () {
  let activeEngine = 'standard';
  let yoloClasses = [];
  let geminiConfigured = false;
  let geminiApiKey = localStorage.getItem('gemini_api_key') || '';

  async function fetchStatus() {
    try {
      const res = await fetch('/api/vision/status');
      if (!res.ok) return;
      const data = await res.json();
      if (data.success && data.status) {
        activeEngine = data.status.active_engine || 'standard';
        yoloClasses = data.status.yolo_world?.classes || [];
        geminiConfigured = data.status.gemini?.configured || false;
      }
    } catch (e) {
      console.warn('Vision status fetch error:', e);
    }
  }

  async function setEngine(engineName) {
    try {
      const res = await fetch('/api/vision/set-engine', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ engine: engineName }),
      });
      const data = await res.json();
      if (data.success) {
        activeEngine = engineName;
        renderHub();
        if (typeof showToast === 'function') {
          const names = {
            standard: 'YOLOv8 Siêu Tốc (Mặc Định)',
            yolo_world: 'YOLO-World (Nhận Diện Vạn Vật)',
            hybrid: 'Hybrid (YOLO + Gemini AI)',
          };
          showToast(`Đã kích hoạt chế độ: ${names[engineName] || engineName}`, 'info');
        }
      }
    } catch (e) {
      console.error('Failed to set engine:', e);
    }
  }

  async function updateYoloClasses(newClasses) {
    try {
      const res = await fetch('/api/vision/set-classes', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ classes: newClasses }),
      });
      const data = await res.json();
      if (data.success) {
        yoloClasses = data.classes || newClasses;
        renderHub();
        if (typeof showToast === 'function') {
          showToast(`Đã nạp ${yoloClasses.length} nhãn nhận diện mới vào AI!`, 'success');
        }
      } else {
        alert(data.message || 'Lỗi cập nhật nhãn.');
      }
    } catch (e) {
      console.error('Failed to update classes:', e);
    }
  }

  async function saveGeminiKey(key) {
    if (!key.trim()) return;
    try {
      const res = await fetch('/api/vision/gemini/save-key', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ api_key: key.trim(), persist_to_env: true }),
      });
      const data = await res.json();
      if (data.success) {
        geminiApiKey = key.trim();
        localStorage.setItem('gemini_api_key', geminiApiKey);
        geminiConfigured = true;
        renderHub();
        if (typeof showToast === 'function') {
          showToast('Đã lưu và kích hoạt Gemini API Key thành công!', 'success');
        }
      } else {
        alert('Lỗi: ' + (data.message || 'API Key không hợp lệ.'));
      }
    } catch (e) {
      alert('Không thể kết nối máy chủ để lưu API Key: ' + e);
    }
  }

  function renderHub(targetElementId = 'visionHubMount') {
    const mount = document.getElementById(targetElementId);
    if (!mount) return;

    let subpanelHtml = '';

    if (activeEngine === 'yolo_world') {
      subpanelHtml = `
        <div class="vision-subpanel">
          <div style="display:flex; justify-content:space-between; align-items:center;">
            <span style="font-size:0.8rem; color:var(--text-secondary); font-weight:600;">
              Từ khóa nhận diện vạn vật (Zero-Shot - gõ bất kỳ sản phẩm nào):
            </span>
            <span style="font-size:0.75rem; color:var(--primary); font-weight:700;">${yoloClasses.length} Nhãn kích hoạt</span>
          </div>

          <div class="classes-tags-wrapper" id="yoloClassesTagList">
            ${yoloClasses
              .map(
                (cls, idx) => `
              <span class="class-tag">
                ${cls}
                <span class="tag-remove" data-idx="${idx}" title="Xóa nhãn này">×</span>
              </span>
            `
              )
              .join('')}
          </div>

          <div class="class-add-row">
            <input type="text" id="newClassInput" class="class-add-input" placeholder="Ví dụ: lon coca, hộp sữa vinamilk, bánh oreo..." />
            <button class="btn btn-secondary" id="addClassBtn" style="padding: 6px 14px; font-size: 0.8rem;">
              + Thêm nhãn
            </button>
            <button class="btn btn-primary" id="resetDefaultClassesBtn" style="padding: 6px 12px; font-size: 0.78rem;" title="Đặt lại nhãn thông dụng">
              Đặt lại mẫu
            </button>
          </div>
        </div>
      `;
    } else if (activeEngine === 'hybrid') {
      subpanelHtml = `
        <div class="vision-subpanel">
          <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;">
            <div>
              <strong style="font-size:0.85rem; color:var(--text-primary);">Google Gemini Vision Cloud AI</strong>
              <p style="font-size:0.78rem; color:var(--text-muted); margin:0;">
                Nhận diện chi tiết thương hiệu, mã vạch, quy cách và đọc chữ tiếng Việt trên bao bì.
              </p>
            </div>
            <span style="font-size:0.78rem; padding:3px 10px; border-radius:99px; font-weight:700; ${
              geminiConfigured
                ? 'background:var(--success-light); color:var(--success); border:1px solid var(--success-border);'
                : 'background:var(--warning-light); color:var(--warning); border:1px solid var(--warning-border);'
            }">
              ${geminiConfigured ? '✓ Đã Kết Nối API' : 'Chưa Nhập API Key'}
            </span>
          </div>

          <div class="class-add-row" style="margin-top:4px;">
            <input type="password" id="geminiKeyInput" class="class-add-input" style="max-width:380px;" 
              placeholder="Dán Google Gemini API Key (AI Studio) tại đây..." value="${geminiApiKey}" />
            <button class="btn btn-primary" id="saveGeminiKeyBtn" style="padding: 6px 14px; font-size: 0.8rem;">
              Lưu Key
            </button>
            <button class="btn btn-secondary" id="testSceneBtn" style="padding: 6px 12px; font-size: 0.78rem;">
              ✨ Quét Toàn Cảnh
            </button>
          </div>
        </div>
      `;
    }

    mount.innerHTML = `
      <div class="vision-hub-card" style="display: none !important;">
        <div class="vision-hub-header">
          <div class="vision-hub-title">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="var(--primary)" stroke-width="2.2">
              <circle cx="12" cy="12" r="10"></circle>
              <polygon points="12 8 8 12 12 16 12 8"></polygon>
              <polygon points="12 8 16 12 12 16 12 8"></polygon>
            </svg>
            <span>Công Cụ AI Thị Giác:</span>
          </div>

          <div class="vision-segmented-control">
            <button class="vision-segment-btn ${activeEngine === 'standard' ? 'active' : ''}" data-engine="standard">
              ⚡ YOLOv8 Siêu Tốc
            </button>
            <button class="vision-segment-btn ${activeEngine === 'yolo_world' ? 'active' : ''}" data-engine="yolo_world">
              🌐 YOLO-World (Vạn Vật)
            </button>
            <button class="vision-segment-btn ${activeEngine === 'hybrid' ? 'active' : ''}" data-engine="hybrid">
              ✨ Gemini AI (Cloud Vision)
            </button>
          </div>
        </div>
        ${subpanelHtml}
      </div>
    `;

    // Bind segmented buttons
    mount.querySelectorAll('.vision-segment-btn').forEach((btn) => {
      btn.addEventListener('click', () => {
        const eng = btn.getAttribute('data-engine');
        if (eng && eng !== activeEngine) {
          setEngine(eng);
        }
      });
    });

    // Bind YOLO-World events
    if (activeEngine === 'yolo_world') {
      mount.querySelectorAll('.tag-remove').forEach((btn) => {
        btn.addEventListener('click', (e) => {
          const idx = parseInt(e.target.getAttribute('data-idx'), 10);
          if (!isNaN(idx)) {
            const nextClasses = [...yoloClasses];
            nextClasses.splice(idx, 1);
            updateYoloClasses(nextClasses);
          }
        });
      });

      const addBtn = mount.querySelector('#addClassBtn');
      const input = mount.querySelector('#newClassInput');
      const addAction = () => {
        const val = (input.value || '').trim();
        if (val) {
          const parts = val.split(/[,;]+/).map((s) => s.trim()).filter(Boolean);
          const next = Array.from(new Set([...yoloClasses, ...parts]));
          input.value = '';
          updateYoloClasses(next);
        }
      };

      if (addBtn && input) {
        addBtn.addEventListener('click', addAction);
        input.addEventListener('keydown', (e) => {
          if (e.key === 'Enter') {
            e.preventDefault();
            addAction();
          }
        });
      }

      const resetBtn = mount.querySelector('#resetDefaultClassesBtn');
      if (resetBtn) {
        resetBtn.addEventListener('click', () => {
          const defaults = [
            'gói cà phê',
            'chai nước',
            'hộp sữa',
            'lon nước ngọt',
            'bánh kẹo',
            'gói snack',
            'điện thoại',
            'sách vở',
            'balo',
            'thùng hộp',
          ];
          updateYoloClasses(defaults);
        });
      }
    }

    // Bind Gemini events
    if (activeEngine === 'hybrid') {
      const saveKeyBtn = mount.querySelector('#saveGeminiKeyBtn');
      const keyInput = mount.querySelector('#geminiKeyInput');
      if (saveKeyBtn && keyInput) {
        saveKeyBtn.addEventListener('click', () => {
          saveGeminiKey(keyInput.value);
        });
      }

      const testSceneBtn = mount.querySelector('#testSceneBtn');
      if (testSceneBtn) {
        testSceneBtn.addEventListener('click', () => {
          triggerFullSceneInspection();
        });
      }
    }
  }

  // Open Gemini Inspection Modal
  function showGeminiModal(loading = true, data = null, imageCropBase64 = null) {
    let overlay = document.getElementById('geminiModalOverlay');
    if (!overlay) {
      overlay = document.createElement('div');
      overlay.id = 'geminiModalOverlay';
      overlay.className = 'gemini-modal-overlay';
      document.body.appendChild(overlay);
    }

    if (loading) {
      overlay.innerHTML = `
        <div class="gemini-modal-content" style="max-width: 440px; text-align: center; padding: 36px 24px;">
          <div class="loading-spinner" style="margin: 0 auto 16px;"></div>
          <h3 style="font-size: 1.1rem; color: var(--text-primary); margin-bottom: 6px;">Đang Phân Tích Gemini Vision...</h3>
          <p style="font-size: 0.84rem; color: var(--text-muted);">
            Trí tuệ nhân tạo đang quét thương hiệu, đọc chữ tiếng Việt và kiểm tra quy cách bao bì...
          </p>
        </div>
      `;
      overlay.style.display = 'flex';
      return;
    }

    if (!data) {
      overlay.style.display = 'none';
      return;
    }

    const info = data.data || {};
    const imgHtml = imageCropBase64
      ? `<img src="${imageCropBase64}" class="gemini-crop-preview" alt="Crop" />`
      : '';

    overlay.innerHTML = `
      <div class="gemini-modal-content">
        <div class="gemini-modal-header">
          <div style="display:flex; align-items:center; gap:8px;">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="var(--purple)" stroke-width="2.2">
              <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"></polygon>
            </svg>
            <strong style="font-size: 1rem; color: var(--text-primary);">Kết Quả Phân Tích Gemini AI</strong>
          </div>
          <button id="closeGeminiModal" style="background:none; border:none; font-size:1.4rem; cursor:pointer; color:var(--text-muted);">&times;</button>
        </div>

        <div class="gemini-modal-body">
          <div class="gemini-product-hero">
            ${imgHtml}
            <div style="flex:1;">
              <span class="badge" style="background:var(--purple-light); color:var(--purple); border:1px solid var(--purple-border); margin-bottom:4px; display:inline-block;">
                ${info.brand || 'Thương hiệu'}
              </span>
              <h2 style="font-size:1.25rem; font-weight:800; color:var(--text-primary); line-height:1.25;">
                ${info.product_name || 'Sản Phẩm Đã Nhận Diện'}
              </h2>
              <p style="font-size:0.84rem; color:var(--text-secondary); margin-top:4px;">
                ${info.category || 'Phân loại'} &nbsp;•&nbsp; ${info.specs || 'Tiêu chuẩn'}
              </p>
            </div>
          </div>

          <div class="gemini-specs-grid">
            <div class="gemini-spec-item">
              <div class="gemini-spec-label">Nhãn Hiệu / Brand</div>
              <div class="gemini-spec-val" style="color:var(--primary);">${info.brand || 'Chưa rõ'}</div>
            </div>
            <div class="gemini-spec-item">
              <div class="gemini-spec-label">Quy Cách / Trọng Lượng</div>
              <div class="gemini-spec-val">${info.specs || 'Nguyên bản'}</div>
            </div>
            <div class="gemini-spec-item">
              <div class="gemini-spec-label">Tình Trạng Bao Bì</div>
              <div class="gemini-spec-val" style="color:var(--success);">${info.packaging_condition || 'Nguyên vẹn'}</div>
            </div>
            <div class="gemini-spec-item">
              <div class="gemini-spec-label">Mã SKU / Barcode</div>
              <div class="gemini-spec-val" style="font-family:var(--font-mono); font-size:0.8rem;">
                ${info.barcode_or_sku || 'N/A'}
              </div>
            </div>
          </div>

          <div style="background:var(--bg-subtle); border:1px solid var(--border-default); border-radius:var(--radius-sm); padding:12px 14px;">
            <div class="gemini-spec-label">Mô Tả Chi Tiết Sản Phẩm</div>
            <p style="font-size:0.85rem; color:var(--text-secondary); line-height:1.5;">
              ${info.description || 'Không có mô tả chi tiết.'}
            </p>
          </div>

          <div style="display:flex; justify-content:space-between; align-items:center; pt:6px;">
            <span style="font-size:0.75rem; color:var(--text-dim);">
              Độ tin cậy AI: <strong>${Math.round((info.confidence || 0.95) * 100)}%</strong> ${data.cached ? '(Từ bộ nhớ đệm)' : ''}
            </span>
            <button class="btn btn-primary" id="confirmGeminiModalBtn" style="padding:7px 18px; font-size:0.82rem;">
              Xác Nhận & Đóng
            </button>
          </div>
        </div>
      </div>
    `;

    overlay.style.display = 'flex';

    const closeHandler = () => {
      overlay.style.display = 'none';
    };
    overlay.querySelector('#closeGeminiModal')?.addEventListener('click', closeHandler);
    overlay.querySelector('#confirmGeminiModalBtn')?.addEventListener('click', closeHandler);
    overlay.addEventListener('click', (e) => {
      if (e.target === overlay) closeHandler();
    });
  }

  // Inspect specific cropped product box
  async function inspectProductCrop(cropBase64) {
    if (!cropBase64) return;
    showGeminiModal(true);

    try {
      const res = await fetch('/api/vision/identify', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image: cropBase64,
          api_key: geminiApiKey || undefined,
        }),
      });

      const data = await res.json();
      if (res.ok && data.success) {
        showGeminiModal(false, data, cropBase64);
      } else {
        alert(data.message || data.detail || 'Không thể nhận diện sản phẩm với Gemini.');
        showGeminiModal(false);
      }
    } catch (e) {
      alert('Lỗi kết nối Gemini API: ' + e);
      showGeminiModal(false);
    }
  }

  // Inspect full scene from canvas or video
  async function triggerFullSceneInspection() {
    // Try to find active canvas or video
    const canvas = document.getElementById('cameraCanvas') || document.getElementById('studioCanvas');
    if (!canvas) {
      alert('Không tìm thấy khung hình video để phân tích.');
      return;
    }

    const fullB64 = canvas.toDataURL('image/jpeg', 0.85);
    showGeminiModal(true);

    try {
      const res = await fetch('/api/vision/analyze-scene', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image: fullB64,
          api_key: geminiApiKey || undefined,
        }),
      });

      const data = await res.json();
      if (res.ok && data.success) {
        const scene = data.data || {};
        const items = scene.detected_products || [];
        const itemsHtml = items.length
          ? `<ul style="margin: 8px 0 0 16px; font-size:0.84rem; color:var(--text-secondary);">
              ${items.map((i) => `<li><strong>${i.name}</strong> (${i.brand || 'Khác'}) - Số lượng: ${i.count || 1} [${i.location || ''}]</li>`).join('')}
            </ul>`
          : '<p style="font-size:0.82rem; color:var(--text-muted);">Không phát hiện sản phẩm riêng biệt.</p>';

        const modalData = {
          data: {
            brand: 'Quét Toàn Cảnh',
            product_name: `Phát hiện ${scene.total_items || items.length} sản phẩm trong khung`,
            category: 'Thị giác không gian',
            specs: `Đếm tổng: ${scene.total_items || items.length}`,
            packaging_condition: 'Tốt',
            description: `${scene.scene_summary || ''}\n\nNhận xét: ${scene.inspection_notes || ''}`,
          },
          cached: data.cached,
        };
        showGeminiModal(false, modalData, fullB64);
      } else {
        alert(data.message || data.detail || 'Lỗi quét toàn cảnh.');
        showGeminiModal(false);
      }
    } catch (e) {
      alert('Lỗi quét toàn cảnh: ' + e);
      showGeminiModal(false);
    }
  }

  // Initialize
  async function init(mountId = 'visionHubMount') {
    await fetchStatus();
    renderHub(mountId);
  }

  return {
    init,
    setEngine,
    updateYoloClasses,
    inspectProductCrop,
    triggerFullSceneInspection,
    showGeminiModal,
    getActiveEngine: () => activeEngine,
  };
})();
