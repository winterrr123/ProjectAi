/* ==========================================================================
   AI VISION PRO - DASHBOARD, STATISTICS & HISTORY LOGIC
   Smooth number counting, modern Chart.js setup, realtime filtering, toasts
   ========================================================================== */

const API_BASE = '/api';

// Reusable Toast Notification System
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

// Enhanced Smooth Count-Up Animation (Preserves previous state for seamless realtime transitions)
function animateValue(element, start, end, duration = 900) {
  if (!element) return;
  const fromVal = element._currentVal !== undefined ? element._currentVal : (Number(start) || 0);
  const targetVal = Number(end) || 0;
  const range = targetVal - fromVal;
  element._currentVal = targetVal;

  if (range === 0) {
    element.textContent = targetVal.toLocaleString('vi-VN');
    return;
  }
  let startTime = null;

  function step(timestamp) {
    if (!startTime) startTime = timestamp;
    const progress = Math.min((timestamp - startTime) / duration, 1);
    // Smooth Quartic ease-out
    const easeOut = 1 - Math.pow(1 - progress, 4);
    const current = Math.floor(fromVal + range * easeOut);
    element.textContent = current.toLocaleString('vi-VN');
    if (progress < 1) {
      window.requestAnimationFrame(step);
    } else {
      element.textContent = targetVal.toLocaleString('vi-VN');
    }
  }
  window.requestAnimationFrame(step);
}

// Format Datetime
function formatDateTime(isoString) {
  if (!isoString) return 'Chưa kết thúc';
  try {
    const d = new Date(isoString);
    if (isNaN(d.getTime())) return isoString;
    return d.toLocaleString('vi-VN', {
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      day: '2-digit',
      month: '2-digit',
      year: 'numeric',
    });
  } catch {
    return isoString;
  }
}

// Fetch helper with error catching
async function fetchJson(url) {
  try {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return await response.json();
  } catch (err) {
    console.warn(`Fetch error for ${url}:`, err);
    return null;
  }
}

// Cache sessions for search/filter in history.html
let allSessions = [];
let currentFilterType = 'ALL';

// Render History Table
function renderHistoryTable(data) {
  const tbody = document.getElementById('historyRows');
  if (!tbody) return;

  if (!data || data.length === 0) {
    tbody.innerHTML = `
      <tr>
        <td colspan="6" style="text-align: center; color: var(--text-muted); padding: 32px;">
          Không tìm thấy phiên nào phù hợp.
        </td>
      </tr>
    `;
    return;
  }

  tbody.innerHTML = data
    .map((s) => {
      const isVideo = s.session_type === 'VIDEO';
      const badgeClass = isVideo ? 'badge-video' : 'badge-camera';
      const typeLabel = isVideo ? 'Video File' : 'Live Camera';
      return `
        <tr>
          <td><span style="font-family:'JetBrains Mono',monospace; color:var(--primary); font-weight:700;">#${s.id}</span></td>
          <td><span class="session-type-badge ${badgeClass}">${typeLabel}</span></td>
          <td>
            <div style="display:flex; align-items:center; gap:10px;">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="color:var(--text-muted); flex-shrink:0;">
                ${isVideo ? '<polygon points="23 7 16 12 23 17 23 7"></polygon><rect x="1" y="5" width="15" height="14" rx="2" ry="2"></rect>' : '<circle cx="12" cy="12" r="3"></circle><path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"></path>'}
              </svg>
              <span style="font-weight:600; color:var(--text-primary);" title="${s.video_name || 'Camera Stream'}">${s.video_name || 'Trực tiếp (Webcam)'}</span>
            </div>
          </td>
          <td style="font-size:0.85rem; color:var(--text-secondary);">${formatDateTime(s.started_at)}</td>
          <td style="font-size:0.85rem; color:var(--text-secondary);">${formatDateTime(s.ended_at)}</td>
          <td>
            <span style="font-family:'JetBrains Mono',monospace; font-size:1.05rem; font-weight:700; color:var(--primary);">
              ${s.total_objects || 0}
            </span>
          </td>
        </tr>
      `;
    })
    .join('');
}

// Filter history data
function applyHistoryFilter() {
  const searchInput = document.getElementById('historySearchInput');
  const query = (searchInput?.value || '').toLowerCase().trim();

  let filtered = allSessions.filter((s) => {
    const matchesType =
      currentFilterType === 'ALL' ||
      (currentFilterType === 'VIDEO' && s.session_type === 'VIDEO') ||
      (currentFilterType === 'CAMERA' && s.session_type !== 'VIDEO');

    const name = (s.video_name || '').toLowerCase();
    const idStr = String(s.id);
    const matchesSearch = query === '' || name.includes(query) || idStr.includes(query);

    return matchesType && matchesSearch;
  });

  renderHistoryTable(filtered);
  const countBadge = document.getElementById('historyCountBadge');
  if (countBadge) {
    countBadge.textContent = `${filtered.length} phiên hiển thị`;
  }
}

// Chart Instances
let barChartInstance = null;
let pieChartInstance = null;

// Palette for charts - clean SaaS indigo, emerald, amber, violet, teal
const chartColors = [
  '#2563eb', // blue
  '#059669', // emerald
  '#d97706', // amber
  '#7c3aed', // purple
  '#0284c7', // cyan
  '#db2777', // pink
  '#ea580c', // orange
  '#0d9488', // teal
];

// Main Dashboard Loader
async function loadDashboard() {
  try {
    const [statsData, sessionsData] = await Promise.all([
      fetchJson(`${API_BASE}/statistics`),
      fetchJson(`${API_BASE}/sessions`),
    ]);

    const stats = statsData?.statistics || {};
    const sessions = sessionsData?.sessions || [];
    allSessions = sessions;

    // 1. Dashboard KPI Counters
    const totalObjectsEl = document.getElementById('totalProducts');
    if (totalObjectsEl) animateValue(totalObjectsEl, 0, stats.total_objects || 0);

    const videoSessionsEl = document.getElementById('videoSessions');
    if (videoSessionsEl) animateValue(videoSessionsEl, 0, stats.total_video_sessions || 0);

    const cameraSessionsEl = document.getElementById('cameraSessions');
    if (cameraSessionsEl) animateValue(cameraSessionsEl, 0, stats.total_camera_sessions || 0);

    const classEntries = Object.entries(stats.class_counts || {});
    const totalClassesEl = document.getElementById('totalClasses');
    if (totalClassesEl) animateValue(totalClassesEl, 0, classEntries.length);

    // 2. Class Summary Pills (index.html)
    const classSummary = document.getElementById('classSummary');
    if (classSummary) {
      if (classEntries.length > 0) {
        classSummary.innerHTML = classEntries
          .map(([name, count], index) => {
            const color = chartColors[index % chartColors.length];
            return `
              <div class="class-pill">
                <span class="class-pill-dot" style="background:${color};"></span>
                <span style="font-weight:600; color:var(--text-primary);">${name}</span>
                <span class="class-pill-count" style="color:${color};">${count}</span>
              </div>
            `;
          })
          .join('');
      } else {
        classSummary.innerHTML = `
          <div style="color: var(--text-muted); font-size: 0.9rem; padding: 12px 0;">
            Chưa có nhãn sản phẩm nào được lưu trữ. Hãy tải video hoặc bật camera để bắt đầu!
          </div>
        `;
      }
    }

    // 3. Recent Sessions List (index.html)
    const recentSessionsEl = document.getElementById('recentSessions');
    if (recentSessionsEl) {
      if (sessions.length > 0) {
        recentSessionsEl.innerHTML = sessions.slice(0, 6)
          .map((s, index) => {
            const isVideo = s.session_type === 'VIDEO';
            const badgeClass = isVideo ? 'badge-video' : 'badge-camera';
            const typeLabel = isVideo ? 'VIDEO' : 'CAMERA';
            return `
              <li class="session-item enter-anim" style="animation: itemSlideUp 0.35s cubic-bezier(0.16, 1, 0.3, 1) ${index * 55}ms both;">
                <div class="session-info">
                  <span class="session-type-badge ${badgeClass}">${typeLabel}</span>
                  <div>
                    <div class="session-name" title="${s.video_name || 'Camera Stream'}">
                      ${s.video_name || 'Camera Stream'}
                    </div>
                    <div style="font-size: 0.74rem; color: var(--text-muted); margin-top: 2px;">
                      ${formatDateTime(s.started_at)}
                    </div>
                  </div>
                </div>
                <div class="session-count-box">
                  <strong>${s.total_objects || 0}</strong>
                  <span>vật thể</span>
                </div>
              </li>
            `;
          })
          .join('');
      } else {
        recentSessionsEl.innerHTML = `
          <li class="session-item" style="color: var(--text-muted); justify-content: center; padding: 24px;">
            Chưa có phiên nhận diện nào được tạo.
          </li>
        `;
      }
    }

    // 4. Statistics Page Elements (statistics.html)
    const totalSessionsEl = document.getElementById('totalSessions');
    if (totalSessionsEl) animateValue(totalSessionsEl, 0, stats.total_sessions || 0);

    const totalVideoSessionsEl = document.getElementById('totalVideoSessions');
    if (totalVideoSessionsEl) animateValue(totalVideoSessionsEl, 0, stats.total_video_sessions || 0);

    const avgConfidenceEl = document.getElementById('avgConfidence');
    if (avgConfidenceEl) {
      const conf = Number(stats.average_confidence || 0);
      avgConfidenceEl.textContent = conf > 0 ? `${(conf * 100).toFixed(1)}%` : 'N/A';
    }

    // Render Leaderboard in Statistics
    const leaderboardRows = document.getElementById('leaderboardRows');
    if (leaderboardRows) {
      const totalObj = stats.total_objects || 1;
      const sortedClasses = [...classEntries].sort((a, b) => b[1] - a[1]);

      if (sortedClasses.length > 0) {
        leaderboardRows.innerHTML = sortedClasses
          .map(([name, count], index) => {
            const pct = Math.round((count / totalObj) * 100);
            const color = chartColors[index % chartColors.length];
            return `
              <tr>
                <td>
                  <div style="display:flex; align-items:center; gap:10px;">
                    <span style="width:9px; height:9px; border-radius:50%; background:${color}; flex-shrink:0;"></span>
                    <strong style="color:var(--text-primary); font-weight:600;">${name}</strong>
                  </div>
                </td>
                <td>
                  <span style="font-weight:700; color:var(--primary); font-size:1rem;">
                    ${count}
                  </span>
                </td>
                <td>
                  <span style="font-weight:600; color:var(--text-secondary);">${pct}%</span>
                </td>
                <td style="min-width: 130px;">
                  <div style="height: 6px; background: #e2e8f0; border-radius: 999px; overflow: hidden;">
                    <div style="width: ${pct}%; height: 100%; background: ${color}; border-radius: 999px; transition: width 0.4s ease;"></div>
                  </div>
                </td>
              </tr>
            `;
          })
          .join('');
      } else {
        leaderboardRows.innerHTML = `
          <tr>
            <td colspan="4" style="text-align: center; color: var(--text-muted); padding: 24px;">
              Chưa có dữ liệu thống kê sản phẩm.
            </td>
          </tr>
        `;
      }
    }

    // 5. Chart.js Setup
    if (window.Chart) {
      Chart.defaults.color = '#64748b';
      Chart.defaults.font.family = "'Plus Jakarta Sans', system-ui, sans-serif";

      const labels = classEntries.map((c) => c[0]);
      const dataValues = classEntries.map((c) => c[1]);

      const ctxBar = document.getElementById('barChart');
      if (ctxBar) {
        if (barChartInstance) barChartInstance.destroy();
        barChartInstance = new Chart(ctxBar, {
          type: 'bar',
          data: {
            labels: labels.length ? labels : ['Chưa có dữ liệu'],
            datasets: [
              {
                label: 'Số lượng phát hiện',
                data: dataValues.length ? dataValues : [0],
                backgroundColor: labels.map((_, i) => chartColors[i % chartColors.length]),
                borderRadius: 8,
                borderSkipped: false,
                barThickness: 28,
              },
            ],
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
              legend: { display: false },
              tooltip: {
                backgroundColor: '#0f172a',
                titleColor: '#ffffff',
                bodyColor: '#93c5fd',
                borderColor: '#1e293b',
                borderWidth: 1,
                padding: 12,
                cornerRadius: 8,
              },
            },
            scales: {
              x: {
                grid: { display: false },
                ticks: { color: '#64748b', font: { weight: 600 } },
              },
              y: {
                beginAtZero: true,
                grid: { color: '#f1f5f9' },
                ticks: { precision: 0, color: '#64748b' },
              },
            },
          },
        });
      }

      const ctxPie = document.getElementById('pieChart');
      if (ctxPie) {
        if (pieChartInstance) pieChartInstance.destroy();
        pieChartInstance = new Chart(ctxPie, {
          type: 'doughnut',
          data: {
            labels: labels.length ? labels : ['Chưa có dữ liệu'],
            datasets: [
              {
                data: dataValues.length ? dataValues : [1],
                backgroundColor: labels.length
                  ? labels.map((_, i) => chartColors[i % chartColors.length])
                  : ['#e2e8f0'],
                borderColor: '#ffffff',
                borderWidth: 3,
                hoverOffset: 6,
              },
            ],
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: '70%',
            plugins: {
              legend: {
                position: 'bottom',
                labels: {
                  boxWidth: 12,
                  padding: 16,
                  color: '#334155',
                  font: { weight: 600 },
                },
              },
              tooltip: {
                backgroundColor: '#0f172a',
                titleColor: '#ffffff',
                borderColor: '#1e293b',
                borderWidth: 1,
                padding: 12,
                cornerRadius: 8,
              },
            },
          },
        });
      }
    }

    // 6. History Page Initial Render (history.html)
    if (document.getElementById('historyRows')) {
      applyHistoryFilter();

      // Hook search & filter events
      document.getElementById('historySearchInput')?.addEventListener('input', applyHistoryFilter);

      const filterAllBtn = document.getElementById('filterAllBtn');
      const filterVideoBtn = document.getElementById('filterVideoBtn');
      const filterCameraBtn = document.getElementById('filterCameraBtn');
      const refreshBtn = document.getElementById('refreshHistoryBtn');

      function updateActiveFilterBtn(activeBtn) {
        [filterAllBtn, filterVideoBtn, filterCameraBtn].forEach((btn) => {
          if (btn) {
            btn.classList.remove('btn-primary');
            btn.classList.add('btn-secondary');
          }
        });
        if (activeBtn) {
          activeBtn.classList.remove('btn-secondary');
          activeBtn.classList.add('btn-primary');
        }
      }

      filterAllBtn?.addEventListener('click', () => {
        currentFilterType = 'ALL';
        updateActiveFilterBtn(filterAllBtn);
        applyHistoryFilter();
      });

      filterVideoBtn?.addEventListener('click', () => {
        currentFilterType = 'VIDEO';
        updateActiveFilterBtn(filterVideoBtn);
        applyHistoryFilter();
      });

      filterCameraBtn?.addEventListener('click', () => {
        currentFilterType = 'CAMERA';
        updateActiveFilterBtn(filterCameraBtn);
        applyHistoryFilter();
      });

      refreshBtn?.addEventListener('click', () => {
        refreshBtn.style.opacity = '0.5';
        showToast('Đang làm mới dữ liệu lịch sử...', 'info');
        loadDashboard().then(() => {
          refreshBtn.style.opacity = '1';
          showToast('Đã cập nhật danh sách phiên thành công!', 'success');
        });
      });

      // Default highlight All
      updateActiveFilterBtn(filterAllBtn);
    }
  } catch (error) {
    console.error('Lỗi khi tải dữ liệu thống kê:', error);
    showToast('Không thể kết nối đến máy chủ API', 'error');
  }
}

// ==========================================================================
// Subtle 3D Mouse Parallax & Card Tilt Engine
// ==========================================================================
function init3DInteractions() {
  const isReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  // 1. Dynamic Topbar Scroll state
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

  if (isReducedMotion) return;

  // Check if touch device - avoid cursor parallax on mobile/tablets
  const isTouchDevice = 'ontouchstart' in window || navigator.maxTouchPoints > 0;
  if (isTouchDevice) return;

  // 2. 3D Tilt on KPI Stat Cards (subtle 2-3 degrees)
  const statCards = document.querySelectorAll('.stat-card');
  statCards.forEach((card) => {
    card.addEventListener('mousemove', (e) => {
      const rect = card.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      const centerX = rect.width / 2;
      const centerY = rect.height / 2;
      const rotateX = ((y - centerY) / centerY) * -2.4;
      const rotateY = ((x - centerX) / centerX) * 2.4;

      card.style.transform = `perspective(1000px) translateY(-6px) scale(1.012) rotateX(${rotateX.toFixed(2)}deg) rotateY(${rotateY.toFixed(2)}deg)`;
    });

    card.addEventListener('mouseleave', () => {
      card.style.transform = '';
    });
  });

  // 3. Subtle Mouse Parallax on Hero Banner (5-12px drift)
  const hero = document.getElementById('heroBanner');
  if (hero) {
    const orb1 = hero.querySelector('.orb-blue');
    const orb2 = hero.querySelector('.orb-purple');

    hero.addEventListener('mousemove', (e) => {
      const rect = hero.getBoundingClientRect();
      const normX = (e.clientX - rect.left) / rect.width - 0.5;
      const normY = (e.clientY - rect.top) / rect.height - 0.5;

      if (orb1) orb1.style.transform = `translate(${normX * -14}px, ${normY * -10}px)`;
      if (orb2) orb2.style.transform = `translate(${normX * 12}px, ${normY * 14}px)`;
    });

    hero.addEventListener('mouseleave', () => {
      if (orb1) orb1.style.transform = '';
      if (orb2) orb2.style.transform = '';
    });
  }
}

// Execute on DOM Ready
document.addEventListener('DOMContentLoaded', () => {
  loadDashboard();
  init3DInteractions();
});
