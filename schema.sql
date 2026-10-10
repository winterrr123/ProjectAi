CREATE DATABASE IF NOT EXISTS product_detection;
USE product_detection;

CREATE TABLE IF NOT EXISTS products (
    id INT AUTO_INCREMENT PRIMARY KEY,
    product_name VARCHAR(255) NOT NULL UNIQUE,
    class_id INT NOT NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    google_id VARCHAR(255) NOT NULL UNIQUE,
    email VARCHAR(255) NOT NULL UNIQUE,
    full_name VARCHAR(255) NULL,
    avatar_url VARCHAR(1024) NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS detection_sessions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    session_type VARCHAR(50) NOT NULL,
    video_name VARCHAR(255),
    started_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    ended_at DATETIME NULL,
    total_objects INT NOT NULL DEFAULT 0,
    CONSTRAINT fk_session_user FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS detection_results (
    id INT AUTO_INCREMENT PRIMARY KEY,
    session_id INT NOT NULL,
    tracking_id INT NULL,
    class_id INT NOT NULL,
    class_name VARCHAR(255) NOT NULL,
    confidence DOUBLE NOT NULL,
    x1 DOUBLE NOT NULL,
    y1 DOUBLE NOT NULL,
    x2 DOUBLE NOT NULL,
    y2 DOUBLE NOT NULL,
    frame_number INT NOT NULL,
    timestamp DOUBLE NOT NULL,
    CONSTRAINT fk_session_result FOREIGN KEY (session_id) REFERENCES detection_sessions(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS product_counts (
    id INT AUTO_INCREMENT PRIMARY KEY,
    session_id INT NOT NULL,
    class_name VARCHAR(255) NOT NULL,
    count INT NOT NULL DEFAULT 0,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_session_count FOREIGN KEY (session_id) REFERENCES detection_sessions(id) ON DELETE CASCADE
);
