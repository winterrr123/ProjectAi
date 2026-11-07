/* ==========================================================================
   PRODUCT VISION - DASHBOARD, STATISTICS & HISTORY LOGIC
   Utility-grade data rendering, table formatters, Chart.js setup, toasts
   ========================================================================== */

const API_BASE = '/api';

// Toast Notification System (Flat, 3px left border)
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
    <div style="flex: 1; font-size: 13px;">${message}</div>
    <span style="cursor: pointer; opacity: 0.6; margin-left: 8px;" title="Đóng">✕</span>
  `;

  toast.querySelector('span').addEventListener('click', () => toast.remove());
  container.appendChild(toast);

  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transition = 'opacity 150ms ease';
    setTimeout(() => toast.remove(), 150);
  }, 3500);
}

// Gentle number transition
function animateValue(element, start, end, duration = 300) {
  if (!element) return;
  const targetVal = Number(end) || 0;
  element.textContent = targetVal.toLocaleString('vi-VN');
}

// Format Datetime (DD/MM/YYYY HH:mm:ss)
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

// Render History Table (Real table, sentence case)
function renderHistoryTable(data) {
  const tbody = document.getElementById('historyRows');
  if (!tbody) return;

  if (!data || data.length === 0) {
    tbody.innerHTML = `
      <tr>
        <td colspan="7" class="table-empty">
          Chưa có phiên nào.
        </td>
      </tr>
    `;
    return;
  }

  tbody.innerHTML = data
    .map((s) => {
      const isVideo = s.session_type === 'VIDEO';
      const badgeClass = isVideo ? 'tag-video' : 'tag-camera';
      const typeLabel = isVideo ? 'Video' : 'Camera';
      return `
        <tr>
          <td class="cell-mono">#${s.id}</td>
          <td><span class="status-tag ${badgeClass}">${typeLabel}</span></td>
          <td class="cell-name">${s.video_name || 'Camera trực tiếp'}</td>
          <td class="cell-sub">${formatDateTime(s.started_at)}</td>
          <td class="cell-sub">${formatDateTime(s.ended_at)}</td>
          <td class="cell-count">${(s.total_objects || 0).toLocaleString('vi-VN')}</td>
          <td style="text-align: center;">
            <button class="btn btn-secondary btn-sm" onclick="deleteHistorySession(${s.id})" title="Xóa phiên" style="padding: 2px 8px; font-size: 12px; color: #dc2626; border-color: #fecaca;">
              Xóa
            </button>
          </td>
        </tr>
      `;
    })
    .join('');
}

// Delete history session
async function deleteHistorySession(sessionId) {
  if (!confirm(`Bạn có chắc muốn xóa phiên #${sessionId} và các tệp báo cáo liên quan?`)) {
    return;
  }

  try {
    const res = await fetch(`${API_BASE}/sessions/${sessionId}`, { method: 'DELETE' });
    const data = await res.json();
    if (!res.ok || !data.success) {
      throw new Error(data.message || data.detail || 'Không thể xóa phiên');
    }

    allSessions = allSessions.filter((s) => s.id !== sessionId);
    applyHistoryFilter();
    showToast(`Đã xóa phiên #${sessionId} thành công.`, 'success');
  } catch (err) {
    showToast(err.message || 'Lỗi khi xóa phiên', 'error');
  }
}
window.deleteHistorySession = deleteHistorySession;

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
    countBadge.textContent = `${filtered.length} phiên`;
  }
}

// Chart Instances
let barChartInstance = null;
let pieChartInstance = null;

// Palette for charts - utility flat colors
const chartColors = [
  '#2563eb', // blue
  '#475569', // slate
  '#059669', // emerald
  '#d97706', // amber
  '#7c3aed', // purple
  '#0284c7', // cyan
  '#dc2626', // red
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

    // 2. Class Summary Table (index.html) - Real table with thin horizontal bar
    const classSummary = document.getElementById('classSummary');
    if (classSummary) {
      if (classEntries.length > 0) {
        const totalObj = stats.total_objects || 1;
        classSummary.innerHTML = `
          <div class="table-responsive">
            <table class="data-table">
              <thead>
                <tr>
                  <th>Tên nhãn</th>
                  <th style="width: 90px; text-align: right;">Số lượng</th>
                  <th style="width: 140px;">Tỉ lệ</th>
                </tr>
              </thead>
              <tbody>
                ${classEntries
                  .map(([name, count]) => {
                    const pct = Math.round((count / totalObj) * 100) || 0;
                    return `
                      <tr>
                        <td style="font-weight: 500;">${name}</td>
                        <td style="text-align: right; font-variant-numeric: tabular-nums; font-weight: 600;">
                          ${count.toLocaleString('vi-VN')}
                        </td>
                        <td>
                          <div class="progress-bar-wrap">
                            <div class="progress-bar-track">
                              <div class="progress-bar-fill" style="width: ${pct}%;"></div>
                            </div>
                            <span class="progress-bar-text">${pct}%</span>
                          </div>
                        </td>
                      </tr>
                    `;
                  })
                  .join('')}
              </tbody>
            </table>
          </div>
        `;
      } else {
        classSummary.innerHTML = `
          <div class="empty-state">Chưa có nhãn sản phẩm nào. Hãy tải video để bắt đầu.</div>
        `;
      }
    }

    // 3. Recent Sessions List (index.html)
    const recentSessionsEl = document.getElementById('recentSessions');
    if (recentSessionsEl) {
      if (sessions.length > 0) {
        recentSessionsEl.innerHTML = `
          <div class="table-responsive">
            <table class="data-table">
              <thead>
                <tr>
                  <th>Mã</th>
                  <th>Nguồn</th>
                  <th>Tên tệp</th>
                  <th>Thời gian</th>
                  <th style="text-align: right;">Số lượng</th>
                </tr>
              </thead>
              <tbody>
                ${sessions.slice(0, 6)
                  .map((s) => {
                    const isVideo = s.session_type === 'VIDEO';
                    const badgeClass = isVideo ? 'tag-video' : 'tag-camera';
                    const typeLabel = isVideo ? 'Video' : 'Camera';
                    return `
                      <tr>
                        <td class="cell-mono">#${s.id}</td>
                        <td><span class="status-tag ${badgeClass}">${typeLabel}</span></td>
                        <td class="cell-name">${s.video_name || 'Camera trực tiếp'}</td>
                        <td class="cell-sub">${formatDateTime(s.started_at)}</td>
                        <td class="cell-count" style="text-align: right;">${(s.total_objects || 0).toLocaleString('vi-VN')}</td>
                      </tr>
                    `;
                  })
                  .join('')}
              </tbody>
            </table>
          </div>
        `;
      } else {
        recentSessionsEl.innerHTML = `
          <div class="empty-state">Chưa có phiên nhận diện nào.</div>
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
      avgConfidenceEl.textContent = conf > 0 ? `${(conf * 100).toFixed(1)}%` : '0%';
    }

    // Render Leaderboard in Statistics
    const leaderboardRows = document.getElementById('leaderboardRows');
    if (leaderboardRows) {
      const totalObj = stats.total_objects || 1;
      const sortedClasses = [...classEntries].sort((a, b) => b[1] - a[1]);

      if (sortedClasses.length > 0) {
        leaderboardRows.innerHTML = sortedClasses
          .map(([name, count]) => {
            const pct = Math.round((count / totalObj) * 100) || 0;
            return `
              <tr>
                <td style="font-weight: 500;">${name}</td>
                <td style="text-align: right; font-variant-numeric: tabular-nums; font-weight: 600;">
                  ${count.toLocaleString('vi-VN')}
                </td>
                <td style="font-variant-numeric: tabular-nums;">${pct}%</td>
                <td style="min-width: 140px;">
                  <div class="progress-bar-wrap">
                    <div class="progress-bar-track">
                      <div class="progress-bar-fill" style="width: ${pct}%;"></div>
                    </div>
                  </div>
                </td>
              </tr>
            `;
          })
          .join('');
      } else {
        leaderboardRows.innerHTML = `
          <tr>
            <td colspan="4" class="table-empty">
              Chưa có dữ liệu thống kê sản phẩm.
            </td>
          </tr>
        `;
      }
    }

    // 5. Chart.js Setup
    if (window.Chart) {
      Chart.defaults.color = '#6b7280';
      Chart.defaults.font.family = "'Inter', system-ui, sans-serif";
      Chart.defaults.font.size = 12;

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
                backgroundColor: '#2563eb',
                borderRadius: 2,
                barThickness: 24,
              },
            ],
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
              legend: { display: false },
              tooltip: {
                backgroundColor: '#1d2125',
                padding: 8,
                cornerRadius: 4,
              },
            },
            scales: {
              x: {
                grid: { display: false },
                ticks: { color: '#6b7280' },
              },
              y: {
                beginAtZero: true,
                grid: { color: '#e3e5e8' },
                ticks: { precision: 0, color: '#6b7280' },
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
                  : ['#e5e7eb'],
                borderColor: '#ffffff',
                borderWidth: 2,
              },
            ],
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: '65%',
            plugins: {
              legend: {
                position: 'bottom',
                labels: {
                  boxWidth: 10,
                  padding: 12,
                  color: '#1d2125',
                },
              },
              tooltip: {
                backgroundColor: '#1d2125',
                padding: 8,
                cornerRadius: 4,
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
        refreshBtn.disabled = true;
        showToast('Đang làm mới dữ liệu lịch sử...', 'info');
        loadDashboard().then(() => {
          refreshBtn.disabled = false;
          showToast('Đã cập nhật danh sách phiên.', 'success');
        });
      });

      // Default highlight All
      updateActiveFilterBtn(filterAllBtn);
    }
  } catch (error) {
    console.error('Lỗi khi tải dữ liệu thống kê:', error);
    showToast('Không thể kết nối đến máy chủ.', 'error');
  }
}

// Execute on DOM Ready
document.addEventListener('DOMContentLoaded', () => {
  loadDashboard();
});
