# scripts/migrate_add_position_fields.py
"""
Migration script to add max_thickness_position and max_camber_position fields.
Run this ONCE to update your existing database schema.
"""

import sqlite3
import logging

logging.basicConfig(level=logging.INFO)


def migrate_database(db_path="airfoils_preprocessed.db"):
    """Add position fields to airfoil_geometry table."""

    print("=" * 70)
    print("DATABASE MIGRATION: Adding Position Fields")
    print("=" * 70)

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Check if columns already exist
        cursor.execute("PRAGMA table_info(airfoil_geometry)")
        columns = [col[1] for col in cursor.fetchall()]

        print(f"\nCurrent columns in airfoil_geometry:")
        for col in columns:
            print(f"  - {col}")

        # Add max_thickness_position if it doesn't exist
        if "max_thickness_position" not in columns:
            print("\n✨ Adding column: max_thickness_position")
            cursor.execute(
                """
                ALTER TABLE airfoil_geometry 
                ADD COLUMN max_thickness_position REAL
            """
            )
            print("   ✅ Added successfully")
        else:
            print("\n✓ Column max_thickness_position already exists")

        # Add max_camber_position if it doesn't exist
        if "max_camber_position" not in columns:
            print("\n✨ Adding column: max_camber_position")
            cursor.execute(
                """
                ALTER TABLE airfoil_geometry 
                ADD COLUMN max_camber_position REAL
            """
            )
            print("   ✅ Added successfully")
        else:
            print("\n✓ Column max_camber_position already exists")

        conn.commit()

        # Verify
        cursor.execute("PRAGMA table_info(airfoil_geometry)")
        new_columns = [col[1] for col in cursor.fetchall()]

        print(f"\nUpdated columns in airfoil_geometry:")
        for col in new_columns:
            print(f"  - {col}")

        conn.close()

        print("\n" + "=" * 70)
        print("✅ MIGRATION COMPLETE")
        print("=" * 70)
        print("\nNext steps:")
        print("1. Recompute geometry metrics with updated analyzer")
        print("2. Run similarity search with position features included")

        return True

    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
        return False


if __name__ == "__main__":
    import sys

    db_path = sys.argv[1] if len(sys.argv) > 1 else "airfoils_preprocessed.db"

    print(f"Migrating database: {db_path}")
    success = migrate_database(db_path)

    sys.exit(0 if success else 1)
