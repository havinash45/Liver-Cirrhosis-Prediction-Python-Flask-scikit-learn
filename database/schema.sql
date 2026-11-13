-- =====================================================================
-- Liver Cirrhosis Prediction System - MySQL Database Schema
-- Database: liver_cirrhosis_db
-- =====================================================================

-- 1. Create Database if not exists
CREATE DATABASE IF NOT EXISTS liver_cirrhosis_db
CHARACTER SET utf8mb4
COLLATE utf8mb4_unicode_ci;

USE liver_cirrhosis_db;

-- 2. Patients Table (Stores registered / recurring clinical patients)
CREATE TABLE IF NOT EXISTS patients (
    id INT AUTO_INCREMENT PRIMARY KEY,
    patient_code VARCHAR(50) UNIQUE NOT NULL,
    full_name VARCHAR(100) NOT NULL,
    age FLOAT NOT NULL,
    gender VARCHAR(20) NOT NULL,
    contact_phone VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_patient_code (patient_code),
    INDEX idx_created_at (created_at)
) ENGINE=InnoDB;

-- 3. Predictions Table (Logs every stage classification and risk check)
CREATE TABLE IF NOT EXISTS predictions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    patient_id INT NULL,
    patient_name VARCHAR(100) DEFAULT 'Anonymous Patient',
    model_type ENUM('stage', 'risk') NOT NULL,
    prediction_result VARCHAR(100) NOT NULL,
    stage_number INT NULL,
    risk_label VARCHAR(10) NULL,
    confidence FLOAT NULL,
    probability_scores JSON NULL,
    inputs JSON NOT NULL,
    doctor_notes TEXT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_prediction_patient
        FOREIGN KEY (patient_id) REFERENCES patients(id)
        ON DELETE SET NULL,
    INDEX idx_model_type (model_type),
    INDEX idx_pred_created_at (created_at)
) ENGINE=InnoDB;

-- 4. Model Training & Audit Logs Table
CREATE TABLE IF NOT EXISTS model_metrics_logs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    model_type ENUM('stage', 'risk') NOT NULL,
    algorithm_name VARCHAR(100) NOT NULL,
    accuracy FLOAT NULL,
    precision_macro FLOAT NULL,
    recall_macro FLOAT NULL,
    f1_macro FLOAT NULL,
    cv_mean_f1 FLOAT NULL,
    cv_std_f1 FLOAT NULL,
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB;

-- Sample patient for demonstration
INSERT IGNORE INTO patients (patient_code, full_name, age, gender, contact_phone)
VALUES ('PT-DEMO-001', 'Demo Patient (PBC ID 2)', 56.4, 'Female', '+1-555-0199');
