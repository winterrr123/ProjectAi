import os
import sqlite3
import pymysql
from urllib.parse import urlparse

def run_migration():
    print("Step 1: Connecting to MySQL server...")
    db_url = os.getenv("DATABASE_URL", "")
    host = os.getenv("MYSQL_HOST", "localhost")
    port = int(os.getenv("MYSQL_PORT", "3306"))
    user = os.getenv("MYSQL_USER", "root")
    password = os.getenv("MYSQL_PASSWORD", "1234")

    if db_url and "mysql" in db_url:
        try:
            parsed = urlparse(db_url.replace("mysql+pymysql://", "mysql://"))
            if parsed.hostname: host = parsed.hostname
            if parsed.port: port = parsed.port
            if parsed.username: user = parsed.username
            if parsed.password: password = parsed.password
        except Exception:
            pass

    conn_mysql = pymysql.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        charset="utf8mb4",
        autocommit=True
    )
    cur_mysql = conn_mysql.cursor()

    print("Step 2: Creating database product_detection if not exists...")
    cur_mysql.execute("CREATE DATABASE IF NOT EXISTS product_detection CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;")
    cur_mysql.execute("USE product_detection;")

    print("Step 3: Creating tables from schema.sql...")
    with open("schema.sql", "r", encoding="utf-8") as f:
        sql_script = f.read()

    statements = [stmt.strip() for stmt in sql_script.split(";") if stmt.strip()]
    for stmt in statements:
        cur_mysql.execute(stmt)

    print("Step 4: Connecting to SQLite database...")
    conn_sqlite = sqlite3.connect("product_detection.db")
    cur_sqlite = conn_sqlite.cursor()

    tables_in_order = [
        "users",
        "sample_products",
        "detection_sessions",
        "product_counts",
        "detection_results",
        "products"
    ]

    cur_mysql.execute("SET FOREIGN_KEY_CHECKS = 0;")

    for table in tables_in_order:
        cur_sqlite.execute(f"PRAGMA table_info({table})")
        columns = [col[1] for col in cur_sqlite.fetchall()]
        col_str = ", ".join([f"`{c}`" for c in columns])
        placeholder_str = ", ".join(["%s" for _ in columns])

        cur_sqlite.execute(f"SELECT {col_str} FROM `{table}`")
        rows = cur_sqlite.fetchall()

        if rows:
            cur_mysql.execute(f"DELETE FROM `{table}`")
            insert_sql = f"INSERT INTO `{table}` ({col_str}) VALUES ({placeholder_str})"
            cur_mysql.executemany(insert_sql, rows)
            print(f"Migrated {len(rows)} rows into {table}")
        else:
            print(f"No rows to migrate for {table}")

    cur_mysql.execute("SET FOREIGN_KEY_CHECKS = 1;")

    print("Step 5: Verifying row counts...")
    all_matched = True
    for table in tables_in_order:
        cur_mysql.execute(f"SELECT COUNT(*) FROM `{table}`")
        mysql_cnt = cur_mysql.fetchone()[0]
        cur_sqlite.execute(f"SELECT COUNT(*) FROM `{table}`")
        sqlite_cnt = cur_sqlite.fetchone()[0]
        status = "MATCH" if mysql_cnt == sqlite_cnt else "MISMATCH"
        print(f"[{status}] {table}: SQLite = {sqlite_cnt}, MySQL = {mysql_cnt}")
        if mysql_cnt != sqlite_cnt:
            all_matched = False

    conn_mysql.close()
    conn_sqlite.close()

    if all_matched:
        print("ALL DATA MIGRATED AND VERIFIED SUCCESSFULLY!")
    else:
        raise RuntimeError("Data verification failed!")

if __name__ == "__main__":
    run_migration()
