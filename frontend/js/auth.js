// auth.js - User authentication helper and navbar profile renderer

async function initAuthNavbar() {
    const topbar = document.querySelector('.topbar');
    if (!topbar) return;

    // Purge any rogue badge elements accidentally inside .nav-actions
    const rogueBadges = document.querySelectorAll('.nav-actions #userProfileNav, .nav-actions .user-profile-badge, .nav-actions .user-profile-header');
    rogueBadges.forEach(el => el.remove());

    let profileBadge = document.getElementById('userProfileNav');
    if (!profileBadge) {
        profileBadge = document.createElement('div');
        profileBadge.id = 'userProfileNav';
        profileBadge.className = 'user-profile-header';
        topbar.appendChild(profileBadge);
    } else {
        // Ensure it is a direct child of topbar (outside nav-actions)
        if (profileBadge.parentElement && profileBadge.parentElement !== topbar) {
            topbar.appendChild(profileBadge);
        }
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
                    ? `<div class="user-avatar-wrap"><img src="${user.avatar_url}" class="user-avatar-img" width="32" height="32" alt="${displayName}" referrerpolicy="no-referrer" onerror="this.onerror=null; this.parentElement.innerHTML='<div class=\\'user-avatar-fallback\\'>${initialLetter}</div>';"></div>`
                    : `<div class="user-avatar-wrap"><div class="user-avatar-fallback">${initialLetter}</div></div>`;

                profileBadge.innerHTML = `
                    <div class="user-profile-badge">
                        ${avatarHtml}
                        <div class="user-info-text">
                            <span class="user-name-text" title="${user.email}">${displayName}</span>
                            <span class="user-status-text"><span class="user-status-dot"></span> Đang hoạt động</span>
                        </div>
                        <button type="button" class="btn-logout" onclick="logoutUser()" title="Đăng xuất khỏi tài khoản">
                            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                                <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"></path>
                                <polyline points="16 17 21 12 16 7"></polyline>
                                <line x1="21" y1="12" x2="9" y2="12"></line>
                            </svg>
                            <span>Thoát</span>
                        </button>
                    </div>
                `;

                // Remove any leftover greeting element if present in DOM
                const existingGreeting = document.getElementById('userGreeting');
                if (existingGreeting) existingGreeting.remove();

                return;
            }
        }
    } catch (err) {
        console.warn('Không thể kiểm tra phiên đăng nhập:', err);
    }

    // Unauthenticated fallback
    profileBadge.innerHTML = `
        <a href="/login" class="btn-login-nav">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
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
