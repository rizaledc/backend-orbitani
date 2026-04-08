-- =============================================================================
-- Orbitani SaaS — RBAC & Multi-Tenancy Migration
-- Jalankan di Supabase Dashboard → SQL Editor
-- =============================================================================

-- -----------------------------------------------------------------------------
-- 1. Tabel organizations
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS organizations (
    id        BIGSERIAL PRIMARY KEY,
    name      TEXT NOT NULL UNIQUE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- -----------------------------------------------------------------------------
-- 2. ALTER users — tambah organization_id (NULL = superadmin)
-- -----------------------------------------------------------------------------
ALTER TABLE users
    ADD COLUMN IF NOT EXISTS organization_id BIGINT REFERENCES organizations(id) ON DELETE SET NULL;

-- Rule: admin & user WAJIB punya organization_id
-- (enforced di application layer via security middleware)

-- -----------------------------------------------------------------------------
-- 3. ALTER lahan — tambah organization_id
-- -----------------------------------------------------------------------------
ALTER TABLE lahan
    ADD COLUMN IF NOT EXISTS organization_id BIGINT REFERENCES organizations(id) ON DELETE SET NULL;

-- -----------------------------------------------------------------------------
-- 4. Tabel ml_feedback (Ground Truth dari petani)
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ml_feedback (
    id                 BIGSERIAL PRIMARY KEY,
    lahan_id           BIGINT REFERENCES lahan(id) ON DELETE SET NULL,
    n                  FLOAT,
    p                  FLOAT,
    k                  FLOAT,
    temperature        FLOAT,
    humidity           FLOAT,
    ph                 FLOAT,
    rainfall           FLOAT,
    ai_recommendation  TEXT,
    actual_crop        TEXT NOT NULL,
    submitted_by       BIGINT REFERENCES users(id) ON DELETE SET NULL,
    created_at         TIMESTAMPTZ DEFAULT NOW()
);

-- -----------------------------------------------------------------------------
-- 5. Index untuk query performa
-- -----------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_users_org      ON users(organization_id);
CREATE INDEX IF NOT EXISTS idx_lahan_org      ON lahan(organization_id);
CREATE INDEX IF NOT EXISTS idx_feedback_lahan ON ml_feedback(lahan_id);
