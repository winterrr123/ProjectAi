// auth.js - User authentication helper and sidebar profile renderer

async function initAuthNavbar() {
    let profileBadge = document.getElementById('userProfileNav');
    if (!profileBadge) {
        profileBadge = document.createElement('div');
        profileBadge.id = 'userProfileNav';
        profileBadge.className = 'user-profile-slot';
        const targetContainer = document.querySelector('.sidebar-footer') || document.querySelector('.app-sidebar') || document.body;
        targetContainer.appendChild(profileBadge);
    }

    try {
        const res = await fetch('/api/auth/me');
        if (res.ok) {
            const data = await res.json();
            if (data.success && data.user) {
                const user = data.user;
                const displayName = user.full_name || user.email.split('@')[0];
                const initialLetter = (displayName ? displayName.charAt(0) : 'U').toUpperCase();
                
                const avatarHtml = user.avatar_url
                    ? `<img src="${user.avatar_url}" class="user-avatar-img" width="26" height="26" alt="${displayName}" referrerpolicy="no-referrer" onerror="this.onerror=null; this.parentElement.innerHTML='<div class=\\'user-avatar-circle\\'>${initialLetter}</div>';">`
                    : `<div class="user-avatar-circle">${initialLetter}</div>`;

                profileBadge.innerHTML = `
                    <div class="user-profile-box">
                        <div class="user-profile-info">
                            ${avatarHtml}
                            <div class="user-details">
                                <span class="user-name" title="${user.email}">${displayName}</span>
                                <span class="user-status-line"><span class="status-dot"></span> Trực tuyến</span>
                            </div>
                        </div>
                        <button type="button" class="btn-logout" onclick="logoutUser()" title="Đăng xuất">
                            Thoát
                        </button>
                    </div>
                `;
                return;
            }
        }
    } catch (err) {
        console.warn('Không thể kiểm tra phiên đăng nhập:', err);
    }

    // Unauthenticated fallback
    profileBadge.innerHTML = `
        <a href="/login" class="btn-login-nav">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M15 3h4a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2h-4"></path>
                <polyline points="10 17 15 12 10 7"></polyline>
                <line x1="15" y1="12" x2="3" y2="12"></line>
            </svg>
            <span>Đăng nhập</span>
        </a>
    `;
}

async function logoutUser() {
    try {
        await fetch('/api/auth/logout', { method: 'POST' });
    } catch (e) {
        console.error(e);
    }
    window.location.href = '/login';
}

document.addEventListener('DOMContentLoaded', initAuthNavbar);
