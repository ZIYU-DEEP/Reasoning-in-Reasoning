import os
import logging
import datetime
import re

def setup_logging_environment(log_directory='logs', log_level=logging.DEBUG):
    """
    Sets up the logging environment by ensuring the log directory exists and configuring
    the root logger to use a FileHandler with a unique filename. This setup affects all
    loggers created in the application.

    :param log_directory: The directory where log files will be stored.
    :param log_level: The logging level for the handler.
    """

    # Ensure the logs directory exists
    os.makedirs(log_directory, exist_ok=True)

    # Create a unique filename with the current date and time
    filename = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S.log')
    full_log_path = os.path.join(log_directory, filename)

    # Configure the root logger to use a FileHandler with the unique filename
    logging.basicConfig(level=log_level,
                        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        handlers=[logging.FileHandler(full_log_path)])

def filter_log_entries(log_file_path, logger_name_to_filter):
    """
    Filters and prints multi-line log entries from a specified log file based on a logger name.
    
    :param log_file_path: Path to the log file to be filtered.
    :param logger_name_to_filter: Logger name to filter the log entries by.

    # Example usage
    log_file_path = "path/to/your/logfile.log"
    logger_name_to_filter = "YourLoggerNameHere"
    filter_log_entries(log_file_path, logger_name_to_filter)
    """
    def is_new_entry(line):
        # Adjust the regex according to the actual timestamp format in your log entries
        return re.match(r'\d{4}-\d{2}-\d{2}', line) is not None

    with open(log_file_path, 'r') as log_file:
        buffer = ""  # Buffer to hold multi-line log entries
        include_buffer = False  # Whether to include the buffered log entry in the output
        
        for line in log_file:
            if is_new_entry(line):
                # When we reach a new log entry, decide whether to print the buffered entry
                if include_buffer:
                    print(buffer, end='')
                
                # Reset buffer and include flag for the new log entry
                buffer = line
                include_buffer = logger_name_to_filter in line
            else:
                # If not a new entry, continue buffering lines
                buffer += line
        
        # Print the last buffered entry if it matches the filter
        if include_buffer:
            print(buffer, end='')