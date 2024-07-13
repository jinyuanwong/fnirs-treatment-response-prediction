CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_file TEXT NOT NULL,
    config_file TEXT NOT NULL,
    seed INTEGER NOT NULL,
    run_itr TEXT NOT NULL,
    result TEXT,
    status TEXT NOT NULL,
    start_time DATETIME DEFAULT CURRENT_TIMESTAMP,
    end_time DATETIME
);
