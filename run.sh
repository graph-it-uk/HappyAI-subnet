#!/bin/bash

# Initialize variables
worker_script="uvicorn app.main:app --host 0.0.0.0 --port 1235"
validator_script="app/neurons/validator.py"
autoRunLoc=$(readlink -f "$0")
proc_name_worker="validator_worker_process"
proc_name_validator="validator_validator_process"
worker_args=()
validator_args=()
version_location="./__init__.py"
version="__version__"

old_args=$@

# Check if pm2 is installed
if ! command -v pm2 &> /dev/null
then
    echo "pm2 could not be found. To install see: https://pm2.keymetrics.io/docs/usage/quick-start/"
    exit 1
fi

# Checks if $1 is smaller than $2
# If $1 is smaller than or equal to $2, then true.
# else false.
version_less_than_or_equal() {
    [  "$1" = "`echo -e "$1\n$2" | sort -V | head -n1`" ]
}

# Checks if $1 is smaller than $2
# If $1 is smaller than $2, then true.
# else false.
version_less_than() {
    [ "$1" = "$2" ] && return 1 || version_less_than_or_equal $1 $2
}

# Returns the difference between
# two versions as a numerical value.
get_version_difference() {
    local tag1="$1"
    local tag2="$2"

    # Extract the version numbers from the tags
    local version1=$(echo "$tag1" | sed 's/v//')
    local version2=$(echo "$tag2" | sed 's/v//')

    # Split the version numbers into an array
    IFS='.' read -ra version1_arr <<< "$version1"
    IFS='.' read -ra version2_arr <<< "$version2"

    # Calculate the numerical difference
    local diff=0
    for i in "${!version1_arr[@]}"; do
        local num1=${version1_arr[$i]}
        local num2=${version2_arr[$i]}

        # Compare the numbers and update the difference
        if (( num1 > num2 )); then
            diff=$((diff + num1 - num2))
        elif (( num1 < num2 )); then
            diff=$((diff + num2 - num1))
        fi
    done

    strip_quotes $diff
}

read_version_value() {
    # Read each line in the file
    while IFS= read -r line; do
        # Check if the line contains the variable name
        if [[ "$line" == *"$version"* ]]; then
            # Extract the value of the variable
            local value=$(echo "$line" | awk -F '=' '{print $2}' | tr -d ' ')
            strip_quotes $value
            return 0
        fi
    done < "$version_location"

    echo ""
}

check_package_installed() {
    local package_name="$1"
    os_name=$(uname -s)

    if [[ "$os_name" == "Linux" ]]; then
        # Use dpkg-query to check if the package is installed
        if dpkg-query -W -f='${Status}' "$package_name" 2>/dev/null | grep -q "installed"; then
            return 1
        else
            return 0
        fi
    elif [[ "$os_name" == "Darwin" ]]; then
         if brew list --formula | grep -q "^$package_name$"; then
            return 1
        else
            return 0
        fi
    else
        echo "Unknown operating system"
        return 0
    fi
}

check_variable_value_on_github() {
    local repo="$1"
    local file_path="$2"
    local variable_name="$3"

    local url="https://api.github.com/repos/$repo/contents/$file_path"
    local response=$(curl -s "$url")

    # Check if the response contains an error message
    if [[ $response =~ "message" ]]; then
        echo "Error: Failed to retrieve file contents from GitHub."
        return 1
    fi

    # Extract the content from the response
    local content=$(echo "$response" | tr -d '\n' | jq -r '.content')

    if [[ "$content" == "null" ]]; then
        echo "File '$file_path' not found in the repository."
        return 1
    fi

    # Decode the Base64-encoded content
    local decoded_content=$(echo "$content" | base64 --decode)

    # Extract the variable value from the content
    local variable_value=$(echo "$decoded_content" | grep "$variable_name" | awk -F '=' '{print $2}' | tr -d ' ')

    if [[ -z "$variable_value" ]]; then
        echo "Variable '$variable_name' not found in the file '$file_path'."
        return 1
    fi

    strip_quotes $variable_value
}

strip_quotes() {
    local input="$1"

    # Remove leading and trailing quotes using parameter expansion
    local stripped="${input#\"}"
    stripped="${stripped%\"}"

    echo "$stripped"
}

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
        script : 'uvicorn',
        cwd    : './worker',
        args   : 'app.main:app --host 0.0.0.0 --port 1235',
        interpreter: 'python3',
        min_uptime: '5m',
        max_restarts: '5'
      }]
    }" > worker.config.js

    echo "Worker PM2 config:"
    cat worker.config.js
    
    pm2 start worker.config.js
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
        args: [$validator_args_joined]
      }]
    }" > validator.config.js

    echo "Validator PM2 config:"
    cat validator.config.js
    
    pm2 start validator.config.js
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

# Get the current version locally
current_version=$(read_version_value)

# Start both processes initially
start_worker
start_validator "$validator_args_joined"

# Check if packages are installed
check_package_installed "jq"
if [ "$?" -eq 1 ]; then
    while true; do

        # First ensure that this is a git installation
        if [ -d "./.git" ]; then

            # check value on github remotely
            latest_version=$(check_variable_value_on_github "graph-it-uk/HappyAI-subnet" "__init__.py" "__version__ ")

            # If the file has been updated
            if version_less_than $current_version $latest_version; then
                echo "Latest version: $latest_version"
                echo "Current version: $current_version"
                diff=$(get_version_difference $latest_version $current_version)
                if [ "$diff" -eq 1 ]; then
                    echo "Current validator version: $current_version"
                    echo "Latest validator version: $latest_version"

                    # Pull latest changes
                    # Failed git pull will return a non-zero output
                    if git pull origin $branch; then
                        # latest_version is newer than current_version, should download and reinstall.
                        echo "New version published. Updating the local copy."

                        # Install latest changes for both worker and validator
                        echo "Installing main requirements..."
                        pip install -e .
                        
                        echo "Installing worker requirements..."
                        cd worker && pip install -r requirements.txt && cd ..

                        echo "Restarting both PM2 processes..."
                        restart_processes "$validator_args_joined"

                        # Update current version:
                        current_version=$(read_version_value)
                        echo "Updated to version: $current_version"

                        # Restart autorun script
                        echo "Restarting script..."
                        ./$(basename $0) $old_args && exit
                    else
                        echo "**Will not update**"
                        echo "It appears you have made changes on your local copy. Please stash your changes using git stash."
                    fi
                else
                    # current version is newer than the latest on git. This is likely a local copy, so do nothing.
                    echo "**Will not update**"
                    echo "The local version is $diff versions behind. Please manually update to the latest version and re-run this script."
                fi
            else
                echo "**Skipping update**"
                echo "$current_version is the same as or more than $latest_version. You are likely running locally."
            fi
        else
            echo "The installation does not appear to be done through Git."
        fi

        # Wait about 30 minutes
        # This should be plenty of time for validators to catch up
        # and should prevent any rate limitations by GitHub.
        sleep 1800
    done
else
    echo "Missing package 'jq'. Please install it for your system first."
fi