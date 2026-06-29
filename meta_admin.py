import argparse
import json
import re
import shutil
from pathlib import Path
from datetime import datetime, timezone

import joblib
import pandas as pd
from dotenv import load_dotenv

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Float,
    DateTime,
    Boolean,
    Text,
)
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import sessionmaker, declarative_base

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = BASE_DIR / "data" / "meta_dataset_final_ordered.csv"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

MODEL_PATH = ARTIFACTS_DIR / "rfc_model.pkl"
SCALER_PATH = ARTIFACTS_DIR / "scaler.pkl"
FEATURE_COLUMNS_PATH = ARTIFACTS_DIR / "feature_columns.pkl"
MODEL_INFO_PATH = ARTIFACTS_DIR / "model_info.txt"


# ============================================================
# Database setup
# ============================================================

load_dotenv(BASE_DIR / ".env")

import os

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL is missing from .env")

engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)

Base = declarative_base()


class MetaClassifierLog(Base):
    __tablename__ = "meta_classifier_logs"

    id = Column(Integer, primary_key=True, index=True)

    url = Column(Text, nullable=True)
    final_url = Column(Text, nullable=True)

    is_web = Column(Integer, default=0)
    is_payment = Column(Integer, default=0)
    is_download = Column(Integer, default=0)

    features_json = Column(JSONB, nullable=False)
    feature_hash = Column(String, unique=True, index=True, nullable=False)

    predicted_label = Column(Integer, nullable=False)
    predicted_probability = Column(Float, nullable=True)

    true_label = Column(Integer, nullable=True)

    review_status = Column(String, default="pending")
    label_source = Column(String, nullable=True)
    review_notes = Column(Text, nullable=True)

    model_version = Column(String, nullable=True)

    used_for_training = Column(Boolean, default=False)
    training_version = Column(String, nullable=True)

    scan_count = Column(Integer, default=1)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_seen_at = Column(DateTime(timezone=True), nullable=True)
    reviewed_at = Column(DateTime(timezone=True), nullable=True)


# ============================================================
# Backend feature names
# These must match features_extractor.py exactly.
# ============================================================

BACKEND_FEATURE_NAMES = [
    "is_top1m",
    "url_length",
    "path_length",
    "host_name_length",
    "no_of_digits",
    "has_ip_host",
    "is_https",
    "domain_age_years",
    "years_to_expire",
    "symbol_count",
    "is_shortened",
    "has_obfuscation",

    "ti_providers_count",
    "ti_malicious_votes",
    "ti_harmless_votes",
    "ti_undetected_votes",
    "ti_any_malicious",
    "ti_unknown",

    "redirect_hop_count",
    "redirect_cross_domain",
    "redirect_stage_score",
    "redirect_stage_suspicious",
    "redirect_heavy",
    "redirect_to_ip",
    "final_url_present",

    "cred_risk",
    "payment_risk",

    "download_detected",
    "download_stage_score",
    "download_stage_suspicious",
    "download_type_direct",
    "download_type_endpoint",
    "download_type_indirect",

    "download_ext_executable",
    "download_ext_script",
    "download_ext_archive",
    "download_ext_document",
    "download_ext_image",
    "download_ext_unknown",

    "dynamic_stage_score",
    "dynamic_stage_suspicious",
    "dynamic_non_http_scheme",
    "dynamic_has_fragment",

    "sandbox_required",

    "file_decision_score",
    "file_max_allowed_mb",

    "file_action_static_scan",
    "file_action_allow_warn",

    "file_priority_high",
    "file_priority_medium",
    "file_priority_low",

    "downloads_new_count",
    "file_size_bytes",
    "open_status_numeric",
    "duration_seconds",
    "vt_malicious_count",
    "vt_undetected_count",

    "is_download",
    "is_payment",
    "is_web",
]


CONTINUOUS_COLS = [
    "url_length",
    "path_length",
    "host_name_length",
    "no_of_digits",
    "domain_age_years",
    "years_to_expire",
    "symbol_count",

    "ti_providers_count",
    "ti_malicious_votes",
    "ti_harmless_votes",
    "ti_undetected_votes",

    "redirect_hop_count",
    "redirect_cross_domain",
    "redirect_stage_score",

    "cred_risk",
    "payment_risk",

    "download_stage_score",
    "dynamic_stage_score",

    "file_decision_score",
    "file_max_allowed_mb",

    "downloads_new_count",
    "file_size_bytes",
    "duration_seconds",
    "vt_malicious_count",
    "vt_undetected_count",
]


OLD_FEATURE_NAMES = [
    "url_len",
    "path_len",
    "host_name_len",
    "having_ip_address",
    "ti_engines_count",
]


# ============================================================
# Utility functions
# ============================================================

def init_db():
    print("Creating database tables...")
    Base.metadata.create_all(bind=engine)
    print("Done.")


def print_record_short(row: MetaClassifierLog):
    risk_score = None

    if row.predicted_probability is not None:
        risk_score = round(float(row.predicted_probability) * 100, 2)

    print(
        f"ID={row.id} | "
        f"url={row.url} | "
        f"pred={row.predicted_label} | "
        f"risk={risk_score} | "
        f"true={row.true_label} | "
        f"status={row.review_status} | "
        f"used={row.used_for_training} | "
        f"model={row.model_version} | "
        f"scans={row.scan_count}"
    )


def load_current_model_version() -> str:
    if not MODEL_INFO_PATH.exists():
        return "meta_rf_v0"

    with open(MODEL_INFO_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("model_version="):
                return line.strip().split("=", 1)[1]

    return "meta_rf_v0"


def next_model_version(current_version: str) -> str:
    match = re.search(r"v(\d+)$", current_version)

    if not match:
        return "meta_rf_v1"

    current_number = int(match.group(1))
    return re.sub(r"v\d+$", f"v{current_number + 1}", current_version)


def validate_no_old_names(df: pd.DataFrame):
    found = [col for col in OLD_FEATURE_NAMES if col in df.columns]

    if found:
        raise ValueError(
            f"Dataset contains old feature names: {found}. "
            "Use backend feature names before retraining."
        )


def align_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if "label" not in df.columns:
        raise ValueError("Data must contain label column.")

    validate_no_old_names(df)

    df = df.copy()

    for col in BACKEND_FEATURE_NAMES:
        if col not in df.columns:
            df[col] = 0

    df = df[["label"] + BACKEND_FEATURE_NAMES]
    df["label"] = df["label"].astype(int)

    return df


# ============================================================
# Database admin commands
# ============================================================

def list_records(limit: int = 20, pending_only: bool = False, reviewed_only: bool = False):
    db = SessionLocal()

    try:
        query = db.query(MetaClassifierLog)

        if pending_only:
            query = query.filter(MetaClassifierLog.true_label.is_(None))

        if reviewed_only:
            query = query.filter(MetaClassifierLog.true_label.isnot(None))

        rows = query.order_by(MetaClassifierLog.id.desc()).limit(limit).all()

        if not rows:
            print("No records found.")
            return

        for row in rows:
            print_record_short(row)

    finally:
        db.close()


def show_record(record_id: int):
    db = SessionLocal()

    try:
        row = db.query(MetaClassifierLog).filter(
            MetaClassifierLog.id == record_id
        ).first()

        if row is None:
            print(f"No record found with ID {record_id}")
            return

        print("\n=== Basic Info ===")
        print(f"ID: {row.id}")
        print(f"URL: {row.url}")
        print(f"Final URL: {row.final_url}")
        print(f"is_web: {row.is_web}")
        print(f"is_payment: {row.is_payment}")
        print(f"is_download: {row.is_download}")

        print("\n=== Prediction ===")
        print(f"Predicted label: {row.predicted_label}")
        print(f"Predicted probability: {row.predicted_probability}")

        if row.predicted_probability is not None:
            print(f"Risk score: {round(float(row.predicted_probability) * 100, 2)}")

        print(f"Model version: {row.model_version}")

        print("\n=== Review ===")
        print(f"True label: {row.true_label}")
        print(f"Review status: {row.review_status}")
        print(f"Label source: {row.label_source}")
        print(f"Review notes: {row.review_notes}")
        print(f"Reviewed at: {row.reviewed_at}")

        print("\n=== Training ===")
        print(f"Used for training: {row.used_for_training}")
        print(f"Training version: {row.training_version}")

        print("\n=== Tracking ===")
        print(f"Scan count: {row.scan_count}")
        print(f"Created at: {row.created_at}")
        print(f"Last seen at: {row.last_seen_at}")

        print("\n=== Features JSON ===")
        print(json.dumps(row.features_json, indent=4, ensure_ascii=False))

    finally:
        db.close()


def review_record(record_id: int, label: int, notes: str | None):
    if label not in [0, 1]:
        print("Label must be 0 or 1.")
        return

    db = SessionLocal()

    try:
        row = db.query(MetaClassifierLog).filter(
            MetaClassifierLog.id == record_id
        ).first()

        if row is None:
            print(f"No record found with ID {record_id}")
            return

        row.true_label = label
        row.review_status = "reviewed"
        row.label_source = "manual"
        row.review_notes = notes
        row.reviewed_at = datetime.now(timezone.utc)

        db.commit()
        db.refresh(row)

        print("Record reviewed.")
        print_record_short(row)

    finally:
        db.close()


def reset_record(record_id: int):
    db = SessionLocal()

    try:
        row = db.query(MetaClassifierLog).filter(
            MetaClassifierLog.id == record_id
        ).first()

        if row is None:
            print(f"No record found with ID {record_id}")
            return

        row.true_label = None
        row.review_status = "pending"
        row.label_source = None
        row.review_notes = None
        row.reviewed_at = None
        row.used_for_training = False
        row.training_version = None

        db.commit()
        db.refresh(row)

        print("Record reset.")
        print_record_short(row)

    finally:
        db.close()


def delete_record(record_id: int):
    db = SessionLocal()

    try:
        row = db.query(MetaClassifierLog).filter(
            MetaClassifierLog.id == record_id
        ).first()

        if row is None:
            print(f"No record found with ID {record_id}")
            return

        db.delete(row)
        db.commit()

        print(f"Deleted record ID {record_id}")

    finally:
        db.close()


def clear_records(confirm: bool):
    if not confirm:
        print("This will delete all records.")
        print("Run with --confirm if you are sure.")
        return

    db = SessionLocal()

    try:
        count = db.query(MetaClassifierLog).delete()
        db.commit()

        print(f"Deleted {count} records.")

    finally:
        db.close()


def stats():
    db = SessionLocal()

    try:
        total = db.query(MetaClassifierLog).count()
        pending = db.query(MetaClassifierLog).filter(
            MetaClassifierLog.true_label.is_(None)
        ).count()
        reviewed = db.query(MetaClassifierLog).filter(
            MetaClassifierLog.true_label.isnot(None)
        ).count()
        used = db.query(MetaClassifierLog).filter(
            MetaClassifierLog.used_for_training.is_(True)
        ).count()

        safe = db.query(MetaClassifierLog).filter(
            MetaClassifierLog.true_label == 0
        ).count()
        suspicious = db.query(MetaClassifierLog).filter(
            MetaClassifierLog.true_label == 1
        ).count()

        print("Database statistics:")
        print(f"Total records: {total}")
        print(f"Pending records: {pending}")
        print(f"Reviewed records: {reviewed}")
        print(f"Used for training: {used}")
        print(f"Reviewed safe labels: {safe}")
        print(f"Reviewed suspicious/malicious labels: {suspicious}")

    finally:
        db.close()


def export_records(output_path: str):
    db = SessionLocal()

    try:
        rows = db.query(MetaClassifierLog).order_by(MetaClassifierLog.id.asc()).all()

        data = []

        for row in rows:
            data.append({
                "id": row.id,
                "url": row.url,
                "final_url": row.final_url,
                "predicted_label": row.predicted_label,
                "predicted_probability": row.predicted_probability,
                "true_label": row.true_label,
                "review_status": row.review_status,
                "model_version": row.model_version,
                "used_for_training": row.used_for_training,
                "training_version": row.training_version,
                "scan_count": row.scan_count,
                "features_json": row.features_json,
            })

        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)

        print(f"Exported {len(df)} records to {output_path}")

    finally:
        db.close()


# ============================================================
# Retraining commands
# ============================================================

def load_base_dataset() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)
    df = align_dataframe(df)

    return df


def load_reviewed_feedback(include_used: bool):
    db = SessionLocal()

    try:
        query = db.query(MetaClassifierLog).filter(
            MetaClassifierLog.true_label.isnot(None)
        )

        if not include_used:
            query = query.filter(
                MetaClassifierLog.used_for_training.is_(False)
            )

        rows = query.order_by(MetaClassifierLog.id.asc()).all()

        feedback_records = []

        for row in rows:
            features = dict(row.features_json or {})
            features["label"] = int(row.true_label)
            feedback_records.append(features)

        return rows, feedback_records

    finally:
        db.close()


def build_feedback_dataframe(records: list[dict]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=["label"] + BACKEND_FEATURE_NAMES)

    df = pd.DataFrame(records)
    df = align_dataframe(df)

    return df


def train_model(combined_df: pd.DataFrame, model_version: str):
    X = combined_df[BACKEND_FEATURE_NAMES]
    y = combined_df["label"].astype(int)

    if y.nunique() < 2:
        raise ValueError("Training data must contain both labels: 0 and 1.")

    x_train, x_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    scaler_cols = [
        col for col in CONTINUOUS_COLS
        if col in x_train.columns
    ]

    scaler = StandardScaler()

    x_train = x_train.copy()
    x_test = x_test.copy()

    x_train[scaler_cols] = scaler.fit_transform(x_train[scaler_cols])
    x_test[scaler_cols] = scaler.transform(x_test[scaler_cols])

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=4,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)

    print("\nAccuracy:", accuracy)
    print("\nClassification report:")
    print(classification_report(y_test, predictions))
    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, predictions))

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(BACKEND_FEATURE_NAMES, FEATURE_COLUMNS_PATH)

    with open(MODEL_INFO_PATH, "w", encoding="utf-8") as f:
        f.write(f"model_version={model_version}\n")
        f.write(f"created_at={datetime.now(timezone.utc)}\n")
        f.write(f"accuracy={accuracy}\n")
        f.write(f"training_rows={len(combined_df)}\n")
        f.write(f"features_count={len(BACKEND_FEATURE_NAMES)}\n")

    print("\nSaved artifacts:")
    print(MODEL_PATH)
    print(SCALER_PATH)
    print(FEATURE_COLUMNS_PATH)
    print(MODEL_INFO_PATH)


def mark_rows_used(rows, training_version: str):
    if not rows:
        return

    db = SessionLocal()

    try:
        row_ids = [row.id for row in rows]

        db.query(MetaClassifierLog).filter(
            MetaClassifierLog.id.in_(row_ids)
        ).update(
            {
                MetaClassifierLog.used_for_training: True,
                MetaClassifierLog.training_version: training_version,
            },
            synchronize_session=False,
        )

        db.commit()

    finally:
        db.close()


def copy_artifacts_to_backend(backend_artifacts_path: str):
    backend_dir = Path(backend_artifacts_path)
    backend_dir.mkdir(parents=True, exist_ok=True)

    for file_name in [
        "rfc_model.pkl",
        "scaler.pkl",
        "feature_columns.pkl",
        "model_info.txt",
    ]:
        src = ARTIFACTS_DIR / file_name
        dst = backend_dir / file_name

        if not src.exists():
            raise FileNotFoundError(f"Missing artifact: {src}")

        shutil.copy2(src, dst)

    print(f"Copied artifacts to backend: {backend_dir}")


def retrain(include_used: bool, min_reviewed: int, dry_run: bool, backend_artifacts: str | None):
    print("Loading base dataset...")
    base_df = load_base_dataset()

    reviewed_rows, feedback_records = load_reviewed_feedback(include_used=include_used)
    feedback_df = build_feedback_dataframe(feedback_records)

    print(f"Base dataset rows: {len(base_df)}")
    print(f"Reviewed feedback rows: {len(feedback_df)}")

    if len(feedback_df) < min_reviewed:
        print(f"Not enough reviewed rows. Required {min_reviewed}, found {len(feedback_df)}.")
        return

    combined_df = pd.concat(
        [base_df, feedback_df],
        ignore_index=True,
    )

    combined_df = combined_df.sample(
        frac=1,
        random_state=42,
    ).reset_index(drop=True)

    print(f"Combined training rows: {len(combined_df)}")
    print("\nLabel counts:")
    print(combined_df["label"].value_counts())

    current_version = load_current_model_version()
    new_version = next_model_version(current_version)

    print(f"\nCurrent model version: {current_version}")
    print(f"New model version: {new_version}")

    if dry_run:
        print("Dry run only. No model saved.")
        return

    train_model(combined_df, new_version)
    mark_rows_used(reviewed_rows, new_version)

    if backend_artifacts:
        copy_artifacts_to_backend(backend_artifacts)

    print("Retraining complete.")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Meta database system: admin, monitoring, review, and retraining"
    )

    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("init-db")

    list_parser = subparsers.add_parser("list")
    list_parser.add_argument("--limit", type=int, default=20)
    list_parser.add_argument("--pending", action="store_true")
    list_parser.add_argument("--reviewed", action="store_true")

    show_parser = subparsers.add_parser("show")
    show_parser.add_argument("id", type=int)

    review_parser = subparsers.add_parser("review")
    review_parser.add_argument("id", type=int)
    review_parser.add_argument("--label", type=int, required=True)
    review_parser.add_argument("--notes", type=str, default=None)

    reset_parser = subparsers.add_parser("reset")
    reset_parser.add_argument("id", type=int)

    delete_parser = subparsers.add_parser("delete")
    delete_parser.add_argument("id", type=int)

    clear_parser = subparsers.add_parser("clear")
    clear_parser.add_argument("--confirm", action="store_true")

    subparsers.add_parser("stats")

    export_parser = subparsers.add_parser("export")
    export_parser.add_argument("--out", type=str, default="meta_logs_export.csv")

    retrain_parser = subparsers.add_parser("retrain")
    retrain_parser.add_argument("--include-used", action="store_true")
    retrain_parser.add_argument("--min-reviewed", type=int, default=1)
    retrain_parser.add_argument("--dry-run", action="store_true")
    retrain_parser.add_argument("--backend-artifacts", type=str, default=None)

    args = parser.parse_args()

    if args.command == "init-db":
        init_db()

    elif args.command == "list":
        list_records(
            limit=args.limit,
            pending_only=args.pending,
            reviewed_only=args.reviewed,
        )

    elif args.command == "show":
        show_record(args.id)

    elif args.command == "review":
        review_record(
            record_id=args.id,
            label=args.label,
            notes=args.notes,
        )

    elif args.command == "reset":
        reset_record(args.id)

    elif args.command == "delete":
        delete_record(args.id)

    elif args.command == "clear":
        clear_records(confirm=args.confirm)

    elif args.command == "stats":
        stats()

    elif args.command == "export":
        export_records(args.out)

    elif args.command == "retrain":
        retrain(
            include_used=args.include_used,
            min_reviewed=args.min_reviewed,
            dry_run=args.dry_run,
            backend_artifacts=args.backend_artifacts,
        )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()