import pytest
from app.database.mysql import SessionLocal, init_db
from app.models.user import User
from app.models.sample import SampleProduct
from app.database.crud import SampleCRUD

def test_sample_product_crud_and_isolation():
    init_db()
    db = SessionLocal()
    try:
        # Create test users
        u1 = db.query(User).filter_by(google_id="test_sample_u1").first()
        if not u1:
            u1 = User(google_id="test_sample_u1", email="u1@test.com", full_name="User 1")
            db.add(u1)
            db.commit()
            db.refresh(u1)

        u2 = db.query(User).filter_by(google_id="test_sample_u2").first()
        if not u2:
            u2 = User(google_id="test_sample_u2", email="u2@test.com", full_name="User 2")
            db.add(u2)
            db.commit()
            db.refresh(u2)

        # Create sample for u1
        sample1 = SampleCRUD.create_sample(
            db,
            user_id=u1.id,
            product_name="Lon Coca-Cola 330ml",
            brand="Coca-Cola",
            category="Nước giải khát",
            visual_class="can",
            sample_image_url="data:image/jpeg;base64,sample123",
            specs="330ml"
        )
        assert sample1.id is not None
        assert sample1.user_id == u1.id
        assert sample1.visual_class == "can"

        # Check isolation
        u1_samples = SampleCRUD.get_user_samples(db, user_id=u1.id)
        assert any(s.id == sample1.id for s in u1_samples)

        u2_samples = SampleCRUD.get_user_samples(db, user_id=u2.id)
        assert not any(s.id == sample1.id for s in u2_samples)

        # Delete sample
        deleted = SampleCRUD.delete_sample(db, sample_id=sample1.id, user_id=u1.id)
        assert deleted is True
        assert not any(s.id == sample1.id for s in SampleCRUD.get_user_samples(db, user_id=u1.id))
    finally:
        db.close()
