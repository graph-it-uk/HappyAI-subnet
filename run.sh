#!/bin/bash

# Initialize variables
worker_script="uvicorn app.main:app --host 0.0.0.0 --port 1235"
validator_script="app/neurons/validator.py"
proc_name_worker="validator_worker_process"
proc_name_validator="validator_validator_process"
worker_args=()
validator_args=()
version_location="./__init__.py"
version="__version__"

# Check if pm2 is installed
if ! command -v pm2 &> /dev/null
then
    echo "pm2 could not be found. To install: sudo npm install -g pm2"
    exit 1
fi

# Parse command line arguments for validator
parse_validator_args() {
    local args=()
    
    # Loop through all command line arguments
    while [[ $# -gt 0 ]]; do
        arg="$1"

        # Check if the argument starts with a hyphen (flag)
        if [[ "$arg" == -* ]]; then
            # Check if the argument has a value
            if [[ $# -gt 1 && "$2" != -* ]]; then
                # Add flag and value
                args+=("'$arg'")
                args+=("'$2'")
                shift 2
            else
                # Add flag only
                args+=("'$arg'")
                shift
            fi
        else
            # Argument is not a flag, add it as it is
            args+=("'$arg'")
            shift
        fi
    done
    
    # Return the joined arguments
    printf "%s," "${args[@]}"
}

# Function to start worker process
start_worker() {
    echo "Starting worker process..."
    
    # Check if worker is already running
    if pm2 status | grep -q $proc_name_worker; then
        echo "Worker is already running. Stopping and restarting..."
        pm2 delete $proc_name_worker
    fi

    # Create PM2 config for worker
    echo "module.exports = {
      apps : [{
        name   : '$proc_name_worker',
        script : 'python3',
        cwd    : './worker',
        args   : '-m uvicorn app.main:app --host 0.0.0.0 --port 1235',
        interpreter: 'python3',
        min_uptime: '5m',
        max_restarts: '5',
        env: {
          PYTHONPATH: '${PWD}:${PYTHONPATH}'
        }
      }]
    }" > worker.config.js

    echo "Worker PM2 config:"
    cat worker.config.js
    
    PYTHONPATH="${PWD}:${PYTHONPATH}" pm2 start "uvicorn app.main:app --host 0.0.0.0 --port 1235" --name validator_worker_process --cwd ./worker
}

# Function to start validator process
start_validator() {
    local validator_args_joined="$1"
    
    echo "Starting validator process..."
    
    # Check if validator is already running
    if pm2 status | grep -q $proc_name_validator; then
        echo "Validator is already running. Stopping and restarting..."
        pm2 delete $proc_name_validator
    fi

    # Remove trailing comma from args
    validator_args_joined=${validator_args_joined%,}

    # Create PM2 config for validator
    echo "module.exports = {
      apps : [{
        name   : '$proc_name_validator',
        script : '$validator_script',
        cwd    : './',
        interpreter: 'python3',
        min_uptime: '5m',
        max_restarts: '5',
        args: [$validator_args_joined],
        env: {
          PYTHONPATH: '${PWD}:${PYTHONPATH}'
        }
      }]
    }" > validator.config.js

    echo "Validator PM2 config:"
    cat validator.config.js
    echo $validator_args_joined
    PYTHONPATH="${PWD}:${PYTHONPATH}" pm2 start "python app.neurons.validator $validator_args_joined" --name validator_validator_process
}

# Function to restart both processes
restart_processes() {
    local validator_args_joined="$1"
    
    echo "Restarting both worker and validator processes..."
    
    # Stop existing processes
    if pm2 status | grep -q $proc_name_worker; then
        pm2 delete $proc_name_worker
    fi
    
    if pm2 status | grep -q $proc_name_validator; then
        pm2 delete $proc_name_validator
    fi
    
    # Start both processes
    start_worker
    start_validator "$validator_args_joined"
}

# Parse validator arguments
validator_args_joined=$(parse_validator_args "$@")

branch=$(git branch --show-current)
echo "Watching branch: $branch"
echo "Worker PM2 process name: $proc_name_worker"
echo "Validator PM2 process name: $proc_name_validator"


# Start both processes
restart_processes "$validator_args_joined"
