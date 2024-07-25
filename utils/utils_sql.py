import sqlite3

def insert_record_into_db(table_name, record, db_path):
    
    # Connect to SQLite database (create if not exists)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Construct INSERT INTO statement dynamically
    columns = ', '.join(record.keys())
    placeholders = ', '.join([':' + key for key in record.keys()])

    # this line of code is fabulous!
    sql = f'''
        INSERT INTO {table_name} ({columns})
        VALUES ({placeholders})
    '''
    # Execute the INSERT statement with performance_output as parameters
    cursor.execute(sql, record)

    # Commit changes and close connection
    conn.commit()
    conn.close()  
    

def update_result_id_in_experiment(experiment_id, result_id, db_path):
    
    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()


    # Update the experiment record with the result_id
    cursor.execute('''
        UPDATE experiments 
        SET result_id = ? 
        WHERE experiment_id = ?
    ''', (result_id, experiment_id))

    # Commit the changes
    conn.commit()

    # Close the connection
    conn.close()    