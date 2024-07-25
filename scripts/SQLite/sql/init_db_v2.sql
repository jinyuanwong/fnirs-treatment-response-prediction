CREATE TABLE IF NOT EXISTS experiments (
    experiment_id TEXT PRIMARY KEY,
    task_name TEXT NOT NULL,
    config_name TEXT NOT NULL,
    model_name TEXT NOT NULL,
    seed INTEGER NOT NULL,
    run_itr TEXT NOT NULL,
    launcher_name TEXT NOT NULL,
    status TEXT NOT NULL,
    start_time DATETIME DEFAULT CURRENT_TIMESTAMP,
    end_time DATETIME,
    result_id INTEGER,
    FOREIGN KEY (result_id) REFERENCES results(result_id)
);

CREATE TABLE IF NOT EXISTS results (
    result_id TEXT PRIMARY KEY,
    config_name TEXT NOT NULL,
    model_name TEXT NOT NULL,
    launcher_name TEXT NOT NULL,
    run_itr TEXT NOT NULL,
    performance_ids TEXT, -- Comma-separated list or JSON array of performance IDs
    FOREIGN KEY (performance_ids) REFERENCES performance(performance_id)
);


CREATE TABLE IF NOT EXISTS performances (
    performance_id TEXT PRIMARY KEY,
    seed INTEGER NOT NULL,
    fold_name TEXT NOT NULL,
    history TEXT,
    y_val_true TEXT,
    y_val_pred TEXT,
    y_test_true TEXT,
    y_test_pred TEXT,
    duration REAL,
    name_performance_metrics TEXT,
    val_performance_metrics TEXT,
    test_performance_metrics TEXT,
    checkpoint_path BLOB,
    others TEXT
);

CREATE TABLE IF NOT EXISTS tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_name TEXT NOT NULL,
    task_file BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS configs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    config_name TEXT NOT NULL,
    config_file BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS models (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT NOT NULL,
    model_file BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS launchers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    launcher_name TEXT NOT NULL,
    launcher_file BLOB NOT NULL
);



