#!/usr/bin/env python3
"""
Configuration Migration Script
Migrates from old flat configuration files to new inheritance-based structure.
"""

import shutil
import sys
from pathlib import Path


def migrate_configs():
    """Migrate configuration files to new inheritance structure."""
    config_dir = Path(__file__).parent.parent / "config"

    migrations = [
        ("default.yaml", "default_old.yaml", "default_new.yaml"),
        ("sitl.yaml", "sitl_old.yaml", "sitl_new.yaml"),
        ("field_test.yaml", "field_test_old.yaml", "field_test_new.yaml"),
    ]

    profile_migrations = [
        (
            "profiles/wifi_beacon.yaml",
            "profiles/wifi_beacon_old.yaml",
            "profiles/wifi_beacon_new.yaml",
        ),
        (
            "profiles/lora_beacon.yaml",
            "profiles/lora_beacon_old.yaml",
            "profiles/lora_beacon_new.yaml",
        ),
    ]

    print("🔧 PISAD Configuration Migration")
    print("=" * 50)

    # Backup and rename main configs
    for original, backup, new in migrations:
        orig_path = config_dir / original
        backup_path = config_dir / backup
        new_path = config_dir / new

        if orig_path.exists() and not backup_path.exists():
            shutil.copy2(orig_path, backup_path)
            print(f"✅ Backed up {original} → {backup}")

        if new_path.exists():
            shutil.copy2(new_path, orig_path)
            print(f"✅ Activated {new} → {original}")

    # Backup and rename profile configs
    for original, backup, new in profile_migrations:
        orig_path = config_dir / original
        backup_path = config_dir / backup
        new_path = config_dir / new

        if orig_path.exists() and not backup_path.exists():
            shutil.copy2(orig_path, backup_path)
            print(f"✅ Backed up {original} → {backup}")

        if new_path.exists():
            shutil.copy2(new_path, orig_path)
            print(f"✅ Activated {new} → {original}")

    print("\n📊 Migration Results:")
    print("-" * 50)

    # Calculate reduction
    old_total = 0
    new_total = 0

    for _, backup, _ in migrations:
        path = config_dir / backup
        if path.exists():
            old_total += len(path.read_text().splitlines())

    base_path = config_dir / "base.yaml"
    if base_path.exists():
        new_total += len(base_path.read_text().splitlines())

    for original, _, _ in migrations:
        path = config_dir / original
        if path.exists():
            new_total += len(path.read_text().splitlines())

    if old_total > 0:
        reduction = (old_total - new_total) / old_total * 100
        print(f"Lines before: {old_total}")
        print(f"Lines after:  {new_total}")
        print(f"Reduction:    {reduction:.1f}%")
        print("\n✨ Configuration consolidation complete!")
        print("   - YAML inheritance enabled")
        print(f"   - {reduction:.1f}% duplication removed")
        print("   - Base template: config/base.yaml")
    else:
        print("⚠️  Could not calculate reduction metrics")

    print("\n💡 To revert: rename *_old.yaml files back to original names")

    return 0


if __name__ == "__main__":
    sys.exit(migrate_configs())
