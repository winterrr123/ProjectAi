/**
 * ============================================================================
 * PRODUCT VISION - REALTIME 3D SPATIAL COMPUTER VISION ANIMATION ENGINE
 * True 3D Mathematical Projection, Dynamic Particle Mesh, Tumbling Polyhedra,
 * Mouse 3D Gyroscopic Parallax & Live Backend API Pulse Synchronization.
 * 60FPS Hardware-Accelerated (Zero External Dependencies)
 * ============================================================================
 */

(function () {
  'use strict';

  // Config
  const CONFIG = {
    fov: 460,                 // 3D Field of view
    gridRows: 14,             // Matrix depth points
    gridCols: 26,             // Matrix width points
    gridSpacing: 74,          // Distance between vertices
    camElevation: 220,        // Camera Y height
    camPitch: 0.38,           // Downward tilt angle
    connectionDist: 85,       // Max distance to draw filaments
    polyhedraCount: 4,        // Floating geometric 3D shapes
  };

  let canvas = null;
  let ctx = null;
  let width = 0;
  let height = 0;
  let dpr = 1;
  let rafId = null;

  // 3D Camera & Mouse State
  let targetRotY = 0;
  let targetRotX = 0;
  let currentRotY = 0;
  let currentRotX = 0;
  let time = 0;
  let lastTime = performance.now();

  // API State Synchronization
  let lastTotalProducts = 0;
  let apiPulseStrength = 0;    // Triggered on product count updates
  let isAiActive = false;       // Live Camera / Video active state
  let apiCheckTimer = null;

  // 3D Vertices Cache
  let waveNodes = [];
  let polyhedra = [];

  // ==========================================================================
  // 1. 3D Mathematics & Matrix Utilities
  // ==========================================================================
  function project3D(x, y, z, cx, cy) {
    // 3D Perspective Projection
    const depth = z + CONFIG.fov;
    if (depth <= 10) return null;
    const scale = CONFIG.fov / depth;
    return {
      x: cx + x * scale,
      y: cy + y * scale,
      scale: scale,
      depth: depth,
    };
  }

  function rotateX(y, z, angle) {
    const cos = Math.cos(angle);
    const sin = Math.sin(angle);
    return {
      y: y * cos - z * sin,
      z: y * sin + z * cos,
    };
  }

  function rotateY(x, z, angle) {
    const cos = Math.cos(angle);
    const sin = Math.sin(angle);
    return {
      x: x * cos + z * sin,
      z: -x * sin + z * cos,
    };
  }

  // ==========================================================================
  // 2. 3D Polyhedra Models (Floating Octahedron & Diamond Prisms)
  // ==========================================================================
  function createOctahedron(size, posX, posY, posZ) {
    const vertices = [
      { x: 0, y: -size, z: 0 },
      { x: size, y: 0, z: 0 },
      { x: 0, y: 0, z: size },
      { x: -size, y: 0, z: 0 },
      { x: 0, y: 0, z: -size },
      { x: 0, y: size, z: 0 },
    ];
    const edges = [
      [0, 1], [0, 2], [0, 3], [0, 4],
      [5, 1], [5, 2], [5, 3], [5, 4],
      [1, 2], [2, 3], [3, 4], [4, 1],
    ];
    return {
      vertices,
      edges,
      pos: { x: posX, y: posY, z: posZ },
      rot: { x: Math.random() * Math.PI, y: Math.random() * Math.PI, z: Math.random() * Math.PI },
      spinSpeed: {
        x: (Math.random() - 0.5) * 0.012,
        y: (Math.random() - 0.5) * 0.015,
        z: (Math.random() - 0.5) * 0.008,
      },
      floatOffset: Math.random() * 10,
    };
  }

  // ==========================================================================
  // 3. World Initialization
  // ==========================================================================
  function initWorld() {
    waveNodes = [];
    const totalW = (CONFIG.gridCols - 1) * CONFIG.gridSpacing;
    const totalD = (CONFIG.gridRows - 1) * CONFIG.gridSpacing;

    for (let r = 0; r < CONFIG.gridRows; r++) {
      for (let c = 0; c < CONFIG.gridCols; c++) {
        waveNodes.push({
          baseX: c * CONFIG.gridSpacing - totalW / 2,
          baseZ: r * CONFIG.gridSpacing - totalD / 2,
          row: r,
          col: c,
          phase: Math.sqrt(r * r + c * c) * 0.35,
        });
      }
    }

    polyhedra = [
      createOctahedron(42, -width * 0.35, -height * 0.15, 80),
      createOctahedron(54, width * 0.38, -height * 0.08, 140),
      createOctahedron(36, -width * 0.28, height * 0.25, 20),
      createOctahedron(48, width * 0.32, height * 0.22, 110),
    ];
  }

  function resizeCanvas() {
    if (!canvas) return;
    dpr = Math.min(window.devicePixelRatio || 1, 1.5);
    width = window.innerWidth;
    height = window.innerHeight;
    canvas.width = Math.floor(width * dpr);
    canvas.height = Math.floor(height * dpr);
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    if (ctx) ctx.scale(dpr, dpr);
    initWorld();
  }

  // ==========================================================================
  // 4. Backend API Integration (Live Data Pulse)
  // ==========================================================================
  async function syncWithBackendAPI() {
    try {
      const res = await fetch('/api/statistics');
      if (!res.ok) return;
      const data = await res.json();
      const newTotal = data.total_products || 0;

      if (lastTotalProducts > 0 && newTotal > lastTotalProducts) {
        // Trigger 3D pulse wave on new detection
        trigger3DPulse();
      }
      lastTotalProducts = newTotal;
    } catch {
      // Offline safe
    }
  }

  function trigger3DPulse() {
    apiPulseStrength = 1.0;
  }

  // Expose global controller
  window.setAiActive = function (active) {
    isAiActive = !!active;
    if (isAiActive) {
      document.body.classList.add('ai-active');
      trigger3DPulse();
    } else {
      document.body.classList.remove('ai-active');
    }
  };

  window.trigger3DPulse = trigger3DPulse;

  // ==========================================================================
  // 5. 3D Render Loop (60FPS)
  // ==========================================================================
  function render(now) {
    const dt = Math.min((now - lastTime) / 1000, 0.1);
    lastTime = now;
    time += dt * (isAiActive ? 1.6 : 1.0);

    // Fade pulse strength
    if (apiPulseStrength > 0) {
      apiPulseStrength = Math.max(0, apiPulseStrength - dt * 0.9);
    }

    // Smooth camera damping
    currentRotY += (targetRotY - currentRotY) * 0.055;
    currentRotX += (targetRotX - currentRotX) * 0.055;

    // Clear Canvas
    ctx.clearRect(0, 0, width, height);

    const cx = width / 2;
    const cy = height * 0.44;
    const pitch = CONFIG.camPitch + currentRotX;
    const yaw = currentRotY;

    // ------------------------------------------------------------------------
    // Render Layer 1: Soft Ambient Background Glows
    // ------------------------------------------------------------------------
    const radial1 = ctx.createRadialGradient(cx * 0.4, cy * 0.6, 50, cx * 0.4, cy * 0.6, width * 0.55);
    radial1.addColorStop(0, 'rgba(59, 130, 246, 0.065)');
    radial1.addColorStop(0.5, 'rgba(37, 99, 235, 0.025)');
    radial1.addColorStop(1, 'rgba(255, 255, 255, 0)');
    ctx.fillStyle = radial1;
    ctx.fillRect(0, 0, width, height);

    const radial2 = ctx.createRadialGradient(cx * 1.5, cy * 1.2, 80, cx * 1.5, cy * 1.2, width * 0.6);
    radial2.addColorStop(0, isAiActive ? 'rgba(6, 182, 212, 0.08)' : 'rgba(139, 92, 246, 0.05)');
    radial2.addColorStop(0.6, 'rgba(16, 185, 129, 0.02)');
    radial2.addColorStop(1, 'rgba(255, 255, 255, 0)');
    ctx.fillStyle = radial2;
    ctx.fillRect(0, 0, width, height);

    // ------------------------------------------------------------------------
    // Render Layer 2: 3D Topological Undulating Spatial Mesh
    // ------------------------------------------------------------------------
    const projectedGrid = [];
    const pulseBoost = apiPulseStrength * 55;

    for (let i = 0; i < waveNodes.length; i++) {
      const node = waveNodes[i];

      // Undulating double sine wave elevation
      const waveY =
        Math.sin(time * 1.1 + node.phase) * 22 +
        Math.cos(time * 0.85 + node.baseX * 0.008) * 16 +
        Math.sin(time * 1.5 + node.baseZ * 0.008) * (pulseBoost + 12);

      let x = node.baseX;
      let y = waveY + CONFIG.camElevation;
      let z = node.baseZ;

      // Camera yaw (Y-axis) rotation
      const rotYRes = rotateY(x, z, yaw);
      x = rotYRes.x;
      z = rotYRes.z;

      // Camera pitch (X-axis) tilt
      const rotXRes = rotateX(y, z, pitch);
      y = rotXRes.y;
      z = rotXRes.z;

      const p = project3D(x, y, z, cx, cy);
      projectedGrid.push(p);
    }

    // Draw Mesh Connecting Filaments
    ctx.lineWidth = 1;
    for (let r = 0; r < CONFIG.gridRows; r++) {
      for (let c = 0; c < CONFIG.gridCols; c++) {
        const idx = r * CONFIG.gridCols + c;
        const p1 = projectedGrid[idx];
        if (!p1) continue;

        // Connect right neighbor
        if (c < CONFIG.gridCols - 1) {
          const p2 = projectedGrid[idx + 1];
          if (p2) {
            const alpha = Math.max(0.015, Math.min(0.22, p1.scale * 0.28 + (isAiActive ? 0.08 : 0)));
            ctx.strokeStyle = isAiActive
              ? `rgba(6, 182, 212, ${alpha.toFixed(3)})`
              : `rgba(59, 130, 246, ${alpha.toFixed(3)})`;
            ctx.beginPath();
            ctx.moveTo(p1.x, p1.y);
            ctx.lineTo(p2.x, p2.y);
            ctx.stroke();
          }
        }

        // Connect down neighbor
        if (r < CONFIG.gridRows - 1) {
          const p3 = projectedGrid[idx + CONFIG.gridCols];
          if (p3) {
            const alpha = Math.max(0.015, Math.min(0.20, p1.scale * 0.25 + (isAiActive ? 0.08 : 0)));
            ctx.strokeStyle = isAiActive
              ? `rgba(16, 185, 129, ${alpha.toFixed(3)})`
              : `rgba(99, 102, 241, ${alpha.toFixed(3)})`;
            ctx.beginPath();
            ctx.moveTo(p1.x, p1.y);
            ctx.lineTo(p3.x, p3.y);
            ctx.stroke();
          }
        }

        // Draw Spatial Node Dots
        const dotRadius = Math.max(1.2, p1.scale * (isAiActive ? 3.4 : 2.6));
        const dotAlpha = Math.max(0.04, Math.min(0.48, p1.scale * 0.45 + apiPulseStrength * 0.35));
        ctx.fillStyle = isAiActive
          ? `rgba(16, 185, 129, ${dotAlpha.toFixed(3)})`
          : `rgba(37, 99, 235, ${dotAlpha.toFixed(3)})`;
        ctx.beginPath();
        ctx.arc(p1.x, p1.y, dotRadius, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    // ------------------------------------------------------------------------
    // Render Layer 3: Tumbling 3D Floating Polyhedra
    // ------------------------------------------------------------------------
    for (let i = 0; i < polyhedra.length; i++) {
      const poly = polyhedra[i];

      // Update rotation
      poly.rot.x += poly.spinSpeed.x;
      poly.rot.y += poly.spinSpeed.y;
      poly.rot.z += poly.spinSpeed.z;

      const floatY = Math.sin(time * 0.8 + poly.floatOffset) * 14;

      // Project vertices
      const projVerts = [];
      for (let v = 0; v < poly.vertices.length; v++) {
        let vx = poly.vertices[v].x;
        let vy = poly.vertices[v].y;
        let vz = poly.vertices[v].z;

        // Local rotation
        let r = rotateX(vy, vz, poly.rot.x);
        vy = r.y; vz = r.z;
        r = rotateY(vx, vz, poly.rot.y);
        vx = r.x; vz = r.z;

        // Translate to world position
        let wx = vx + poly.pos.x;
        let wy = vy + poly.pos.y + floatY;
        let wz = vz + poly.pos.z;

        // Camera rotation
        let cr = rotateY(wx, wz, yaw);
        wx = cr.x; wz = cr.z;
        cr = rotateX(wy, wz, pitch);
        wy = cr.y; wz = cr.z;

        projVerts.push(project3D(wx, wy, wz, cx, cy));
      }

      // Draw wireframe edges
      ctx.lineWidth = 1.2;
      const polyAlpha = isAiActive ? 0.22 : 0.12;
      ctx.strokeStyle = i % 2 === 0
        ? `rgba(37, 99, 235, ${polyAlpha})`
        : `rgba(124, 58, 237, ${polyAlpha})`;

      for (let e = 0; e < poly.edges.length; e++) {
        const [v1, v2] = poly.edges[e];
        const p1 = projVerts[v1];
        const p2 = projVerts[v2];
        if (p1 && p2) {
          ctx.beginPath();
          ctx.moveTo(p1.x, p1.y);
          ctx.lineTo(p2.x, p2.y);
          ctx.stroke();
        }
      }

      // Draw vertex hubs
      for (let v = 0; v < projVerts.length; v++) {
        const p = projVerts[v];
        if (p) {
          ctx.fillStyle = 'rgba(59, 130, 246, 0.28)';
          ctx.beginPath();
          ctx.arc(p.x, p.y, Math.max(1.5, p.scale * 2.8), 0, Math.PI * 2);
          ctx.fill();
        }
      }
    }

    rafId = requestAnimationFrame(render);
  }

  // ==========================================================================
  // 6. Interactive Mouse & Lifecycle Handlers
  // ==========================================
  function onMouseMove(e) {
    const normX = (e.clientX / width) - 0.5;
    const normY = (e.clientY / height) - 0.5;
    targetRotY = normX * 0.42;  // Gentle horizontal 3D tilt
    targetRotX = normY * 0.18;  // Gentle vertical 3D pitch
  }

  function onMouseLeave() {
    targetRotY = 0;
    targetRotX = 0;
  }

  function handleVisibilityChange() {
    if (document.hidden) {
      if (rafId) {
        cancelAnimationFrame(rafId);
        rafId = null;
      }
    } else {
      if (!rafId) {
        lastTime = performance.now();
        rafId = requestAnimationFrame(render);
      }
    }
  }

  // ==========================================================================
  // 7. Initialization
  // ==========================================================================
  function init() {
    // Ensure canvas exists
    canvas = document.getElementById('bg3dCanvas');
    if (!canvas) {
      canvas = document.createElement('canvas');
      canvas.id = 'bg3dCanvas';
      canvas.className = 'bg-3d-canvas';
      canvas.setAttribute('aria-hidden', 'true');
      document.body.insertBefore(canvas, document.body.firstChild);
    }

    ctx = canvas.getContext('2d', { alpha: true });
    resizeCanvas();

    window.addEventListener('resize', resizeCanvas, { passive: true });
    window.addEventListener('mousemove', onMouseMove, { passive: true });
    document.addEventListener('mouseleave', onMouseLeave, { passive: true });
    document.addEventListener('visibilitychange', handleVisibilityChange);

    // Initial API Sync and interval
    syncWithBackendAPI();
    apiCheckTimer = setInterval(syncWithBackendAPI, 6000);

    // Start 60fps loop
    lastTime = performance.now();
    rafId = requestAnimationFrame(render);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
